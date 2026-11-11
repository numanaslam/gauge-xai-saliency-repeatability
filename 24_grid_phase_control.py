#!/usr/bin/env python3
"""
GAUGE-XAI step 24 (Diagnostics review round, experiment 4): is the loss of voxel-level IG agreement a stride-grid /
resampling artefact, or acquisition-related?

The same CT patch (no second scan) is translated by s voxels, IG (air baseline, same settings as 06) is recomputed with
the centre-prior channel translated with the image (so the whole input is translated; only the stride grid phase and the
zero padding differ), the map is translated back and compared with the unshifted map inside the ROI.
The network's total stride is 16 (stem, layer2, layer3, layer4 each /2): a translation by a multiple of 16 voxels keeps
the grid phase, any other integer translation changes it.
Variants per cell x run:
  diag{s}   translation (s, s, s), s in {1, 2, 3, 4, 8, 16} voxels
  x1, x16   translation of 1 or 16 voxels along x only
  sub0.5    translation of (0.5, 0.5, 0.5) voxel by trilinear resampling of the input (map resampled back)
  mapinterp no new IG: the unshifted map itself translated by 0.5 voxel and back (isolates resampling of the map)
Measures vs the unshifted map (positive attribution in the valid ROI): Spearman at 1 mm, Spearman pooled to 16 mm cells,
top-10% Dice, and the logit change. Reference from the acquisition comparison (manuscript): Spearman 0.49 (ResNet) and
0.54 (DenseNet) between scans, 0.25 / 0.29 after local registration; pooled 0.96.

Decision rule (fixed before running), on the median Spearman over cells and runs:
  grid-locked    median(s in {1,2,3,4,8}) < 0.6 AND median(s = 16) > 0.9
  interpolation  all integer shifts > 0.9 AND sub0.5 < 0.6
  otherwise      mixed or acquisition-driven: the manuscript keeps "consistent with, not isolated".

Needs 06 run with --save-patches (patches/) and the models (--runs/seed*/best.pt).
Usage (server, from code/):
  python 24_grid_phase_control.py --saliency D:\\paper8\\results\\journal\\saliency_resnet --runs D:\\paper8\\results\\journal\\runs_resnet --out D:\\paper8\\results\\journal\\review_grid_phase\\resnet --n-runs 5
  python 24_grid_phase_control.py --self-test      (no data: checks the shift/compare logic on synthetic arrays)
"""
import argparse
import importlib.util
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import shift as nd_shift
from scipy.stats import spearmanr

N = 128
INT_SHIFTS = (1, 2, 3, 4, 8, 16)
VARIANTS = [f"diag{s}" for s in INT_SHIFTS] + ["x1", "x16", "sub0.5", "mapinterp"]


def shift_vec(variant):
    if variant.startswith("diag"):
        s = int(variant[4:])
        return (s, s, s)
    if variant in ("x1", "x16"):
        return (0, 0, int(variant[1:]))
    if variant == "sub0.5":
        return (0.5, 0.5, 0.5)
    if variant == "mapinterp":
        return (0.5, 0.5, 0.5)
    raise ValueError(variant)


def translate_int(a, sv, fill=0.0):
    """Translate array a by integer vector sv (content moves +s along each axis); vacated voxels = fill."""
    out = np.full_like(a, fill)
    src, dst = [], []
    for s, n in zip(sv, a.shape):
        s = int(s)
        src.append(slice(0, n - s) if s >= 0 else slice(-s, n))
        dst.append(slice(s, n) if s >= 0 else slice(0, n + s))
    out[tuple(dst)] = a[tuple(src)]
    return out


def valid_region(sv, n=N, margin=0):
    """Boolean mask (original frame) of voxels that stay inside the patch after translating by sv (and, for non-integer
    shifts, are at least `margin` voxels from the border)."""
    v = np.zeros((n, n, n), bool)
    sl = []
    for s in sv:
        lo = int(np.ceil(max(0, -s))) + margin
        hi = n - int(np.ceil(max(0, s))) - margin
        sl.append(slice(lo, hi))
    v[tuple(sl)] = True
    return v


def pooled(a, block=16):
    n = a.shape[0] // block
    return a.reshape(n, block, n, block, n, block).mean(axis=(1, 3, 5))


def compare(ref, back, roi, valid, top=0.10):
    """ref, back: attribution maps in the original frame; roi: ROI; valid: voxels valid for both."""
    m = roi & valid
    a, b = np.clip(ref, 0, None) * m, np.clip(back, 0, None) * m
    rho = spearmanr(a[m], b[m])[0] if a[m].std() > 0 and b[m].std() > 0 else np.nan
    pa, pb = pooled(a), pooled(b)
    frac = pooled(m.astype(float))
    keep = frac > 0.5
    rho_p = spearmanr(pa[keep], pb[keep])[0] if keep.sum() > 3 and pa[keep].std() > 0 and pb[keep].std() > 0 else np.nan
    k = max(1, int(round(top * m.sum())))
    ia = set(np.argpartition(-a[m], k - 1)[:k])
    ib = set(np.argpartition(-b[m], k - 1)[:k])
    dice = 2 * len(ia & ib) / (2 * k)
    return dict(spearman=float(rho), spearman_pooled16=float(rho_p), dice_top10=float(dice))


def radial_shifted(n, sv, device, dtype):
    import torch
    ax = [torch.linspace(-1, 1, n, device=device) - s * (2.0 / (n - 1)) for s in sv]
    zz, yy, xx = torch.meshgrid(*ax, indexing="ij")
    return torch.sqrt(zz ** 2 + yy ** 2 + xx ** 2).to(dtype)[None, None]


def explain(model, x, shift, S6, steps, batch, device):
    """IG (air baseline) of x with the model's centre prior translated by `shift` voxels. Returns (map, logit)."""
    import torch
    xt = torch.from_numpy(x)[None, None].to(device)
    if getattr(model, "centre_prior", False):
        key = (tuple(xt.shape[2:]), xt.device, xt.dtype)
        model._radial = {key: radial_shifted(xt.shape[2], shift, xt.device, xt.dtype)}
    try:
        attr, fx, _ = S6.integrated_gradients(model, xt, steps, batch)
    finally:
        model._radial = {}
    return attr, fx


def load_by_path(name, fname):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(fname))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def self_test():
    rng = np.random.default_rng(0)
    a = rng.random((N, N, N)).astype(np.float32)
    for sv in [(1, 1, 1), (16, 16, 16), (0, 0, 16), (3, 3, 3)]:
        sh = translate_int(a, sv)
        back = translate_int(sh, tuple(-s for s in sv))
        v = valid_region(sv)
        assert np.allclose(back[v], a[v]), sv
        assert v.sum() == np.prod([N - s for s in sv]), (sv, v.sum())
    roi = np.zeros((N, N, N), bool)
    roi[20:100, 20:100, 20:100] = True
    r = compare(a, a, roi, valid_region((16, 16, 16)))
    assert abs(r["spearman"] - 1) < 1e-9 and abs(r["dice_top10"] - 1) < 1e-9, r
    noisy = a + rng.normal(0, 0.5, a.shape).astype(np.float32)
    r2 = compare(a, noisy, roi, valid_region((1, 1, 1)))
    assert r2["spearman"] < 0.9, r2
    # smooth map: sub-voxel translation and back barely changes it
    sm = nd_shift(rng.random((N, N, N)).astype(np.float32), 0, order=1)
    from scipy.ndimage import gaussian_filter
    sm = gaussian_filter(sm, 3)
    sb = nd_shift(nd_shift(sm, (0.5,) * 3, order=1, mode="constant"), (-0.5,) * 3, order=1, mode="constant")
    r3 = compare(sm, sb, roi, valid_region((0.5,) * 3, margin=2))
    assert r3["spearman"] > 0.99, r3
    print("self-test ok", r, r2, r3)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--saliency", type=Path)
    ap.add_argument("--runs", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--n-runs", type=int, default=5)
    ap.add_argument("--n-cells", type=int, default=30)
    ap.add_argument("--ig-steps", type=int, default=32)
    ap.add_argument("--ig-batch", type=int, default=8)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        self_test()
        return
    import torch
    from gauge_lib import PRIMARY_EXCLUDE, from_uint8
    from gauge_model import load_checkpoint

    S6 = load_by_path("rider_saliency", "06_rider_saliency.py")
    roi = S6.ROI
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpts = sorted(args.runs.glob("seed*/best.pt"))[: args.n_runs]
    models = {int(c.parent.name.replace("seed", "")): load_checkpoint(c, device)[0] for c in ckpts}
    cells = sorted(p for p in (args.saliency / "patches").glob("*_scan1_1.25L.npz")
                   if p.stem.rsplit("_", 2)[0] not in PRIMARY_EXCLUDE)[: args.n_cells]
    args.out.mkdir(parents=True, exist_ok=True)
    rows, t0 = [], time.time()
    for ci, pp in enumerate(cells, 1):
        x = from_uint8(np.load(pp)["patch"]).astype(np.float32)
        pid = pp.stem.rsplit("_", 2)[0]
        for seed, model in models.items():
            ref, ref_logit = explain(model, x, (0, 0, 0), S6, args.ig_steps, args.ig_batch, device)
            for v in VARIANTS:
                sv = shift_vec(v)
                if v == "mapinterp":
                    back = nd_shift(nd_shift(ref, sv, order=1, mode="constant"), tuple(-s for s in sv), order=1, mode="constant")
                    valid, dlogit = valid_region(sv, margin=2), 0.0
                elif v == "sub0.5":
                    xs = nd_shift(x, sv, order=1, mode="constant", cval=0.0)
                    m_s, lg = explain(model, xs, sv, S6, args.ig_steps, args.ig_batch, device)
                    back = nd_shift(m_s, tuple(-s for s in sv), order=1, mode="constant")
                    valid, dlogit = valid_region(sv, margin=2), lg - ref_logit
                else:
                    xs = translate_int(x, sv)
                    m_s, lg = explain(model, xs, sv, S6, args.ig_steps, args.ig_batch, device)
                    back = translate_int(m_s, tuple(-s for s in sv))
                    valid, dlogit = valid_region(sv), lg - ref_logit
                rows.append(dict(cell=pp.stem, PatientID=pid, seed=seed, variant=v, dlogit=dlogit, **compare(ref, back, roi, valid)))
        print(f"{ci}/{len(cells)} cells ({(time.time() - t0) / 60:.1f} min)", flush=True)
        pd.DataFrame(rows).to_csv(args.out / "grid_phase_rows.csv", index=False)
    df = pd.DataFrame(rows)
    summ = df.groupby("variant")[["spearman", "spearman_pooled16", "dice_top10", "dlogit"]].median().reindex(VARIANTS)
    summ["abs_dlogit_median"] = df.assign(a=df.dlogit.abs()).groupby("variant").a.median().reindex(VARIANTS)
    summ.round(4).to_csv(args.out / "grid_phase_summary.csv")
    print(summ.round(3).to_string())
    small = np.median([summ.loc[f"diag{s}", "spearman"] for s in (1, 2, 3, 4, 8)])
    s16 = summ.loc["diag16", "spearman"]
    ints = [summ.loc[f"diag{s}", "spearman"] for s in INT_SHIFTS]
    if small < 0.6 and s16 > 0.9:
        verdict = "grid-locked"
    elif min(ints) > 0.9 and summ.loc["sub0.5", "spearman"] < 0.6:
        verdict = "interpolation"
    else:
        verdict = "mixed or acquisition-driven (keep 'consistent with, not isolated')"
    (args.out / "verdict.txt").write_text(f"median Spearman s in 1,2,3,4,8: {small:.3f}; s=16: {s16:.3f}; sub0.5: {summ.loc['sub0.5','spearman']:.3f}\nverdict: {verdict}\n")
    print("verdict:", verdict)


if __name__ == "__main__":
    main()
