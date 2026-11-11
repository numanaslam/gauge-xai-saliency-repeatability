#!/usr/bin/env python3
"""
GAUGE-XAI step 14 (journal): faithfulness of the explanations by deletion, and its scan-rescan repeatability.

For each cell, training run and explainer, the top-k% of ROI voxels ranked by positive attribution are replaced
by a blurred copy of the patch (or air, --fill air) and the malignancy logit is recomputed; the same number of
randomly chosen ROI voxels is deleted as a reference (--n-random draws). The deletion score is
    score = mean_k [ (logit(x) - logit(x without top-k)) - (logit(x) - logit(x without random k)) ]
in logit units: > 0 means the explanation ranks voxels that matter to the model above random voxels.
A faithful explanation should also have a REPEATABLE score (scan-rescan ICC of the score).

Model-randomisation sanity check (Adebayo et al., NeurIPS 2018), --sanity-cells N (default 40; 0 = off):
on N primary cells (scan 1, 1.25L), the weights of the classifier head, then head + layer4, then the whole
network are re-initialised in cascade (the radial centre-prior channel is an input, so it is kept) and IG and
Grad-CAM are recomputed. Reported: Spearman of the positive attribution (ROI) and LAF with the trained model's
map. An explanation that is insensitive to the weights (rho near 1) cannot be explaining the model.

Needs 06 run with --save-patches (CT patches) plus its maps/ and masks/, and the models (--runs).
Writes <out>/faithfulness.csv (one row per cell x run x explainer), faithfulness_summary.csv,
sanity_randomisation.csv and sanity_summary.csv.

Usage:
  python 14_faithfulness.py --saliency ..\\results\\saliency --runs ..\\results\\runs --out ..\\results\\analysis\\faithfulness
"""
import argparse
import importlib.util
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from gauge_lib import PATCH_SPACING_MM, PRIMARY_EXCLUDE, RESULTS_DIR, SECONDARY_SETTINGS, SETTINGS, from_uint8
from gauge_model import load_checkpoint

P = [s for s in SETTINGS if s not in SECONDARY_SETTINGS]


def load_by_path(name, fname):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(fname))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


S7 = load_by_path("gauge_analysis", "07_gauge_analysis.py")  # load_map, _roi, map_similarity, pairs
S6 = load_by_path("rider_saliency", "06_rider_saliency.py")   # integrated_gradients, grad_cam

CASCADE = (("head", ("fc",)), ("head+layer4", ("fc", "layer4")), ("all", None))


def randomised(model, modules, seed=0):
    """Copy of model with the named top-level modules (None = all) re-initialised."""
    import copy

    m = copy.deepcopy(model)
    torch.manual_seed(seed)
    targets = [m] if modules is None else [getattr(m, k) for k in modules]
    for t in targets:
        for sub in t.modules():
            if hasattr(sub, "reset_parameters"):
                sub.reset_parameters()
            if hasattr(sub, "reset_running_stats"):
                sub.reset_running_stats()
    return m.eval()


def sanity(models, sal, n_cells, device, ig_steps=32):
    from scipy.stats import spearmanr

    cells = sorted(p.stem for p in (sal / "patches").glob("*_scan1_1.25L.npz")
                   if p.stem.rsplit("_", 2)[0] not in PRIMARY_EXCLUDE)[:n_cells]
    rows = []
    for name in cells:
        x = from_uint8(np.load(sal / "patches" / f"{name}.npz")["patch"])
        m = np.load(sal / "masks" / f"{name}.npz")["mask"].astype(bool)
        n = x.shape[0]
        roi = S7._roi(n)
        xt = torch.from_numpy(x)[None, None].to(device)

        def explain(model):
            ig = S6.integrated_gradients(model, xt, ig_steps, 8)[0]
            cam = S6.grad_cam(model, xt)[1]
            return {"ig": ig, "gradcam": cam}

        for seed, model in models.items():
            ref = explain(model)
            for level, mods in CASCADE:
                new = explain(randomised(model, mods, seed=1000 + seed))
                for e in ref:
                    pa, pb = np.clip(ref[e][roi], 0, None), np.clip(new[e][roi], 0, None)
                    rho = spearmanr(pa, pb).correlation if pa.max() > 0 and pb.max() > 0 else np.nan
                    laf = lambda a: float(np.clip(a, 0, None)[roi & m].sum() / max(np.clip(a, 0, None)[roi].sum(), 1e-12))  # noqa: E731
                    rows.append(dict(cell=name, seed=seed, explainer=e, randomised=level, spearman=rho,
                                     laf_trained=laf(ref[e]), laf_randomised=laf(new[e])))
    return pd.DataFrame(rows)


@torch.no_grad()
def logits(model, xs, device, batch=8):
    out = []
    for i in range(0, len(xs), batch):
        xb = torch.from_numpy(np.stack(xs[i:i + batch]))[:, None].to(device)
        out.append(model(xb).float().cpu().numpy())
    return np.concatenate(out)


def inserted(x, fill, flat_idx):
    y = fill.copy().reshape(-1)
    y[flat_idx] = x.reshape(-1)[flat_idx]
    return y.reshape(x.shape)


def deleted(x, fill, flat_idx):
    y = x.copy().reshape(-1)
    y[flat_idx] = fill.reshape(-1)[flat_idx]
    return y.reshape(x.shape)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--saliency", default=RESULTS_DIR / "saliency", type=Path)
    ap.add_argument("--runs", default=RESULTS_DIR / "runs", type=Path)
    ap.add_argument("--out", default=RESULTS_DIR / "analysis" / "faithfulness", type=Path)
    ap.add_argument("--fractions", default="0.01,0.02,0.05,0.10,0.20")
    ap.add_argument("--fill", default="blur", choices=["blur", "air", "mean"],
                    help="replacement for deleted voxels: 5 mm-blurred patch, air (0), or the patch mean")
    ap.add_argument("--unit", default="voxel", choices=["voxel", "cube"],
                    help="delete single voxels, or whole 8 mm cubes ranked by their summed positive attribution "
                         "(region-level deletion; avoids salt-and-pepper patterns)")
    ap.add_argument("--within", default="roi", choices=["roi", "lesion"],
                    help="candidate voxels: the ROI, or only the lesion (SEG within the ROI; lesion-restricted test)")
    ap.add_argument("--random", default="uniform", choices=["uniform", "matched"],
                    help="random reference: uniform over the candidates, or matched to the intensity histogram (16 "
                         "quantile bins) of the explainer's top-k voxels (tissue-matched control)")
    ap.add_argument("--mode", default="deletion", choices=["deletion", "insertion"],
                    help="deletion: replace top-k voxels by the fill; insertion: start from the fill image and restore "
                         "the top-k voxels (score = logit gain over the same number of random voxels)")
    ap.add_argument("--explainers", default=None, help="comma list (default: every explainer with maps)")
    ap.add_argument("--n-random", type=int, default=3)
    ap.add_argument("--settings", default=",".join(P), help="reconstructions to evaluate (default: primary)")
    ap.add_argument("--patients", nargs="*")
    ap.add_argument("--sanity-cells", type=int, default=40, help="cells for the model-randomisation check (0 = off)")
    ap.add_argument("--only-sanity", action="store_true")
    args = ap.parse_args()
    if args.unit == "cube" and (args.random == "matched" or args.within == "lesion"):
        raise SystemExit("--unit cube is only implemented with --random uniform --within roi")
    args.out.mkdir(parents=True, exist_ok=True)
    fr = [float(v) for v in args.fractions.split(",")]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = {int(c.parent.name.replace("seed", "")): load_checkpoint(c, device)[0] for c in sorted(args.runs.glob("seed*/best.pt"))}
    explainers = sorted(d.name for d in (args.saliency / "maps").iterdir() if d.is_dir())
    if args.explainers:
        explainers = [e for e in explainers if e in args.explainers.split(",")]
    patches = sorted((args.saliency / "patches").glob("*.npz"))
    if not patches:
        raise SystemExit("no patches/ - run 06_rider_saliency.py with --save-patches first")
    settings = set(args.settings.split(","))
    if args.sanity_cells:
        sn = sanity(models, args.saliency, args.sanity_cells, device)
        sn.round(5).to_csv(args.out / "sanity_randomisation.csv", index=False)
        ss = sn.groupby(["explainer", "randomised"]).agg(n=("spearman", "size"), spearman_median=("spearman", "median"),
                                                          spearman_q1=("spearman", lambda v: v.quantile(.25)),
                                                          spearman_q3=("spearman", lambda v: v.quantile(.75)),
                                                          laf_trained=("laf_trained", "median"),
                                                          laf_randomised=("laf_randomised", "median")).reset_index()
        ss.round(4).to_csv(args.out / "sanity_summary.csv", index=False)
        print("model-randomisation sanity check (Spearman with the trained model's map):")
        print(ss.round(3).to_string(index=False), "\n", flush=True)
        if args.only_sanity:
            return
    rng = np.random.default_rng(0)
    rows, t0 = [], time.time()
    for n_cell, pp in enumerate(patches, 1):
        pid, acq, setting = pp.stem.rsplit("_", 2)
        if setting not in settings or (args.patients and pid not in args.patients):
            continue
        x = from_uint8(np.load(pp)["patch"])
        n = x.shape[0]
        roi = S7._roi(n)
        roi_idx = np.flatnonzero(roi.reshape(-1))
        if args.within == "lesion":
            les = np.load(args.saliency / "masks" / f"{pp.stem}.npz")["mask"].astype(bool) & roi
            roi_idx = np.flatnonzero(les.reshape(-1))
            if len(roi_idx) < 20:
                continue
        cand = np.zeros(n ** 3, bool)
        cand[roi_idx] = True
        xflat = x.reshape(-1)
        edges = np.quantile(xflat[roi_idx], np.linspace(0, 1, 17)[1:-1])
        bin_of = np.digitize(xflat, edges)
        bins = {b: roi_idx[bin_of[roi_idx] == b] for b in range(16)}

        def matched(top_idx):
            """Random candidate voxels with the same intensity-bin counts as top_idx."""
            out = []
            for b, c in zip(*np.unique(bin_of[top_idx], return_counts=True)):
                pool = bins.get(b, roi_idx)
                out.append(rng.choice(pool, c, replace=c > len(pool)))
            return np.concatenate(out)

        change = (lambda idx: inserted(x, fill, idx)) if args.mode == "insertion" else (lambda idx: deleted(x, fill, idx))
        if args.fill == "blur":
            from scipy.ndimage import gaussian_filter
            fill = gaussian_filter(x, sigma=5.0 / PATCH_SPACING_MM).astype(np.float32)
        elif args.fill == "mean":
            fill = np.full_like(x, float(x[roi].mean()))
        else:
            fill = np.zeros_like(x)
        ks = [max(1, int(f * len(roi_idx))) for f in fr]
        rand_sets = [[rng.choice(roi_idx, k, replace=False) for k in ks] for _ in range(args.n_random)]
        if args.unit == "cube":  # same voxel budget, but deleted as whole 8 mm cubes
            c = 8
            cid = (np.arange(n)[:, None, None] // c * (n // c) ** 2 + np.arange(n)[None, :, None] // c * (n // c)
                   + np.arange(n)[None, None, :] // c).reshape(-1)
            cubes_in_roi = np.unique(cid[roi_idx])
            members = {q: np.flatnonzero(cid == q) for q in cubes_in_roi}

            def take(order_cubes, k):
                out, tot = [], 0
                for q in order_cubes:
                    if tot >= k:
                        break
                    out.append(members[q])
                    tot += len(members[q])
                return np.concatenate(out)

            rand_sets = [[take(rng.permutation(cubes_in_roi), k) for k in ks] for _ in range(args.n_random)]
        for seed, model in models.items():
            f_x = logits(model, [x], device)[0]
            f_ref = logits(model, [fill], device)[0] if args.mode == "insertion" else f_x
            sign = -1.0 if args.mode == "insertion" else 1.0  # insertion: gain = f(changed) - f(fill)
            if args.random == "uniform":
                rand_logit = logits(model, [change(idx) for rs in rand_sets for idx in rs], device)
                rand_drop = sign * (f_ref - rand_logit).reshape(args.n_random, len(ks)).mean(0)
            for expl in explainers:
                a = S7.load_map(args.saliency, expl, seed, pp.stem, (n,) * 3)
                if a is None:
                    continue
                score_vox = np.where(cand.reshape(a.shape), np.clip(a, 0, None), -np.inf).reshape(-1)
                if args.unit == "cube":
                    pos = np.clip(a, 0, None).reshape(-1) * roi.reshape(-1)
                    csum = {q: pos[members[q]].sum() for q in cubes_in_roi}
                    order_c = sorted(cubes_in_roi, key=lambda q: -csum[q])
                    top = [take(order_c, k) for k in ks]
                else:
                    order = np.argsort(-score_vox, kind="stable")
                    top = [order[:k] for k in ks]
                top_logit = logits(model, [change(idx) for idx in top], device)
                top_drop = sign * (f_ref - top_logit)
                if args.random == "matched":
                    rl = logits(model, [change(matched(idx)) for _ in range(args.n_random) for idx in top], device)
                    rand_drop = sign * (f_ref - rl).reshape(args.n_random, len(ks)).mean(0)
                row = dict(explainer=expl, seed=seed, PatientID=pid, acq=acq, setting=setting, logit=float(f_x),
                           deletion_score=float(np.mean(top_drop - rand_drop)))
                for f_, td, rd in zip(fr, top_drop, rand_drop):
                    row[f"drop_top_{f_:g}"], row[f"drop_rand_{f_:g}"] = float(td), float(rd)
                rows.append(row)
        if n_cell % 25 == 0:
            print(f"  {n_cell}/{len(patches)} cells ({(time.time() - t0) / 60:.1f} min)", flush=True)
    df = pd.DataFrame(rows)
    df.round(5).to_csv(args.out / "faithfulness.csv", index=False)

    summ = []
    prim = df[~df.PatientID.isin(PRIMARY_EXCLUDE)]
    for (e, seed), g in prim.groupby(["explainer", "seed"]):
        r = dict(explainer=e, seed=seed, n_maps=len(g), score_median=g.deletion_score.median(),
                 score_q1=g.deletion_score.quantile(0.25), score_q3=g.deletion_score.quantile(0.75),
                 share_positive=float((g.deletion_score > 0).mean()))
        for f_ in fr:
            r[f"drop_top_{f_:g}_median"] = g[f"drop_top_{f_:g}"].median()
            r[f"drop_rand_{f_:g}_median"] = g[f"drop_rand_{f_:g}"].median()
        iccs = []
        for setting, gs in g.groupby("setting"):
            ids, y = S7.pairs(gs, "deletion_score", "scan1", "scan2", "acq")
            if y is not None and len(y) >= 5:
                iccs.append(S7.PC.icc_1_1(y)[0])
        r["score_scan_rescan_icc_median"] = float(np.median(iccs)) if iccs else np.nan
        summ.append(r)
    sm = pd.DataFrame(summ)
    sm.round(4).to_csv(args.out / "faithfulness_summary.csv", index=False)
    with pd.option_context("display.width", 220, "display.max_columns", 30):
        print(sm.groupby("explainer")[["score_median", "share_positive", "score_scan_rescan_icc_median"]].median().round(3))
    print(f"\nWrote {args.out}/faithfulness.csv and faithfulness_summary.csv ({len(df)} rows)")


if __name__ == "__main__":
    main()
