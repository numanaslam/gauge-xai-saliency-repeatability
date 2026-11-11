#!/usr/bin/env python3
"""
GAUGE-XAI step 18 (journal, review round 2): two sensitivity analyses.

(a) resolution  Explainer comparison at matched spatial resolution. IG maps are pooled to 16 mm cells (8^3, the
                layer4 Grad-CAM and Occlusion grid) and 8 mm cells (16^3, the layer3 Grad-CAM grid) and upsampled
                trilinearly as Grad-CAM is; LAF, shifted-SEG control (25 mm, six directions), scan-rescan ICC/wSD and
                between-run wSD are computed for these and for the native maps (IG 1 mm, Grad-CAM layer4 and, if
                present, layer3, Occlusion).
(b) centring    Lesion-centring sensitivity. Compares IG and Grad-CAM LAF from the reference run (patch centred on the
                SEG centroid) with runs of 06 --centre-jitter-mm J (centre displaced by J mm in a random direction,
                independently per cell): median LAF, scan-rescan ICC and wSD, between-run wSD, and the variance
                components (patient x run vs patient x scan).

Usage:
  python 18_sensitivity.py resolution --saliency D:\\paper8\\results\\journal\\saliency_resnet --out D:\\paper8\\results\\journal\\analysis_resnet\\sensitivity
  python 18_sensitivity.py centring --ref D:\\paper8\\results\\journal\\saliency_resnet \\
         --jitter 5=D:\\paper8\\results\\journal\\saliency_resnet_jit5 10=D:\\paper8\\results\\journal\\saliency_resnet_jit10 \\
         --out D:\\paper8\\results\\journal\\analysis_resnet\\sensitivity
"""
import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

from gauge_lib import PATCH_SPACING_MM, PRIMARY_EXCLUDE, RESULTS_DIR, SECONDARY_SETTINGS, SETTINGS

P = [s for s in SETTINGS if s not in SECONDARY_SETTINGS]


def load_by_path(name, fname):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(fname))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


S7 = load_by_path("gauge_analysis", "07_gauge_analysis.py")
R10 = load_by_path("robustness", "10_robustness.py")


def pool_up(a, grid):
    """Positive attribution averaged in grid^3 blocks, then trilinearly upsampled to the input size."""
    import torch
    import torch.nn.functional as F

    n = a.shape[0]
    p = n // grid
    g = np.clip(a, 0, None).reshape(grid, p, grid, p, grid, p).mean((1, 3, 5)).astype(np.float32)
    return F.interpolate(torch.from_numpy(g)[None, None], size=(n,) * 3, mode="trilinear", align_corners=False)[0, 0].numpy()


def laf(a, mask, roi):
    pos = np.clip(a, 0, None) * roi
    t = pos.sum()
    return float(pos[mask].sum() / t) if t > 0 else np.nan


def reliability(d, value="laf"):
    """Scan-rescan ICC/wSD (median / mean over runs and primary reconstructions) and between-run wSD."""
    d = d[~d.PatientID.isin(PRIMARY_EXCLUDE) & d.setting.isin(P)]
    iccs, wsds, runs = [], [], []
    for (seed, st), g in d.groupby(["seed", "setting"]):
        _, y = S7.pairs(g, value, "scan1", "scan2", "acq")
        if y is not None and len(y) >= 5:
            icc, wsd = S7.PC.icc_1_1(y)
            iccs.append(icc)
            wsds.append(wsd)
    for (acq, st), g in d.groupby(["acq", "setting"]):
        w = g.pivot_table(index="PatientID", columns="seed", values=value).dropna()
        if w.shape[1] >= 2 and len(w) >= 5:
            runs.append(S7.PC.icc_1_1(w.to_numpy(float))[1])
    # nan-aware: an all-zero (empty) Grad-CAM stratum has an undefined ICC; it is skipped and counted
    return dict(laf_median=d[value].median(), icc_scan_median=np.nanmedian(iccs) if iccs else np.nan,
                icc_scan_min=np.nanmin(iccs) if iccs else np.nan, n_icc_undefined=int(np.isnan(iccs).sum()) if iccs else 0,
                n_icc=len(iccs), wsd_scan=np.nanmean(wsds) if wsds else np.nan, wsd_run=np.nanmean(runs) if runs else np.nan,
                share_zero_laf=float((d[value] == 0).mean()))


def resolution(args):
    sal = args.saliency
    if not (sal / "maps" / "ig").exists():
        raise SystemExit(f"{sal / 'maps' / 'ig'} not found - check --saliency")
    seeds = sorted(int(p.name[4:]) for p in (sal / "maps" / "ig").glob("seed*"))
    expl = [e for e in ("ig", "gradcam", "gradcam_l3", "occlusion") if (sal / "maps" / e).exists()]
    names = sorted(p.stem for p in (sal / "masks").glob("*.npz"))
    rows = []
    for k, name in enumerate(names, 1):
        pid, acq, setting = name.rsplit("_", 2)
        mask = np.load(sal / "masks" / f"{name}.npz")["mask"].astype(bool)
        n = mask.shape[0]
        roi = S7._roi(n)
        sh = int(round(25.0 * n / 128 / PATCH_SPACING_MM))
        shifts = [R10.shifted(mask, v) for v in ((sh, 0, 0), (-sh, 0, 0), (0, sh, 0), (0, -sh, 0), (0, 0, sh), (0, 0, -sh))]
        for seed in seeds:
            for e in expl:
                a = S7.load_map(sal, e, seed, name, (n,) * 3)
                if a is None:
                    continue
                variants = {e: a}
                if e == "ig":
                    variants = {"ig_1mm": a, "ig_pooled_8mm": pool_up(a, n // 8), "ig_pooled_16mm": pool_up(a, 8)}
                for lab, m in variants.items():
                    rows.append(dict(map=lab, seed=seed, PatientID=pid, acq=acq, setting=setting, laf=laf(m, mask, roi),
                                     laf_shift=np.nanmean([laf(m, s_, roi) for s_ in shifts]) if (m > 0).any() else np.nan))
        if k % 50 == 0:
            print(f"  {k}/{len(names)} cells", flush=True)
    df = pd.DataFrame(rows)
    summarise_resolution(df, args)


def summarise_resolution(df, args):
    sal = getattr(args, "saliency", None)
    if df.empty:
        raise SystemExit(f"no maps found under {sal} - check --saliency")
    args.out.mkdir(parents=True, exist_ok=True)
    df.round(5).to_csv(args.out / "resolution_cells.csv", index=False)
    grid = {"ig_1mm": "1 mm", "ig_pooled_8mm": "8 mm", "ig_pooled_16mm": "16 mm", "gradcam_l3": "8 mm",
            "gradcam": "16 mm", "occlusion": "16 mm"}
    summ = []
    for lab, g in df.groupby("map"):
        r = reliability(g)
        prim = g[~g.PatientID.isin(PRIMARY_EXCLUDE) & g.setting.isin(P)]
        summ.append(dict(map=lab, cell_size=grid.get(lab, "?"), laf_shift_median=prim.laf_shift.median(),
                         specificity_ratio=prim.laf.median() / prim.laf_shift.median(), **r))
    sm = pd.DataFrame(summ).sort_values(["cell_size", "map"])
    sm.round(4).to_csv(args.out / "resolution_summary.csv", index=False)
    print(sm.round(3).to_string(index=False))


def centring(args):
    runs = {"0": args.ref}
    for item in args.jitter:
        j, path = item.split("=", 1)
        runs[j] = Path(path)
    rows, vcs = [], []
    for j, path in runs.items():
        m = pd.read_csv(path / "measurands.csv")
        for e in ("ig", "gradcam"):
            g = m[m.explainer == e]
            rows.append(dict(jitter_mm=float(j), explainer=e, n_rows=len(g), **reliability(g)))
        d = m[(m.explainer == "ig") & ~m.PatientID.isin(PRIMARY_EXCLUDE) & m.setting.isin(P)].copy()
        d["kernel"], d["thick"] = d.setting.str[-1], d.setting.str[:-1]
        est, conv = R10.fit_vc(d, "laf")
        vcs.append(dict(jitter_mm=float(j), explainer="ig", converged=conv, **{f"sd_{k.replace(' ', '_')}": np.sqrt(max(v, 0)) for k, v in est.items()}))
    out = pd.DataFrame(rows).merge(pd.DataFrame(vcs), on=["jitter_mm", "explainer"], how="left")  # VC for IG only
    out["sd_ratio_run_scan"] = out.sd_patient_x_seed / out.sd_patient_x_scan
    out["variance_ratio_run_scan"] = out.sd_ratio_run_scan ** 2
    args.out.mkdir(parents=True, exist_ok=True)
    out.round(4).to_csv(args.out / "centring_summary.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(out.round(3).to_string(index=False))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("resolution")
    a.add_argument("--saliency", default=RESULTS_DIR / "journal" / "saliency_resnet", type=Path)
    a.add_argument("--out", default=RESULTS_DIR / "journal" / "analysis_resnet" / "sensitivity", type=Path)
    a.add_argument("--from-cells", action="store_true", help="re-summarise <out>/resolution_cells.csv without the maps")
    b = sub.add_parser("centring")
    b.add_argument("--ref", default=RESULTS_DIR / "journal" / "saliency_resnet", type=Path)
    b.add_argument("--jitter", nargs="+", required=True, help="J=path pairs, e.g. 5=D:\\...\\saliency_resnet_jit5")
    b.add_argument("--out", default=RESULTS_DIR / "journal" / "analysis_resnet" / "sensitivity", type=Path)
    args = ap.parse_args()
    if args.cmd == "resolution" and args.from_cells:
        summarise_resolution(pd.read_csv(args.out / "resolution_cells.csv"), args)
    elif args.cmd == "resolution":
        resolution(args)
    else:
        centring(args)


if __name__ == "__main__":
    main()
