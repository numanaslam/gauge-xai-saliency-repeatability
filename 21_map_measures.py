#!/usr/bin/env python3
"""
GAUGE-XAI step 21 (journal, review round 3): map-level measures beyond the positive LAF, from the stored maps.

Per cell, run and explainer (IG air and Grad-CAM layer4 by default):
  laf_r30, laf_r45, laf_r60   LAF with an ROI of radius 30, 45 and 60 mm (60 mm = the primary definition)
  core_share, rim_share       share of the lesion's positive attribution in its core (SEG eroded by 2 mm) and in the
                              2 mm rim; core_density_ratio = attribution per voxel in the core / in the rim
                              (> 1: internal texture, < 1: boundary)
  shell_share                 positive attribution in a 0-5 mm shell outside the SEG, as a share of the ROI total
  neg_laf                     negative attribution on the lesion / negative attribution in the ROI
Writes <out>/map_measures_cells.csv and map_measures_summary.csv (median, scan-rescan ICC, between-run wSD).

Usage:
  python 21_map_measures.py --saliency D:\\paper8\\results\\journal\\saliency_resnet --out D:\\paper8\\results\\journal\\analysis_resnet\\map_measures
"""
import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import binary_dilation, binary_erosion

from gauge_lib import PATCH_SPACING_MM, RESULTS_DIR


def load_by_path(name, fname):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(fname))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


S7 = load_by_path("gauge_analysis", "07_gauge_analysis.py")
S18 = load_by_path("sensitivity", "18_sensitivity.py")
MEASURES = ["laf_r30", "laf_r45", "laf_r60", "core_share", "rim_share", "core_density_ratio", "shell_share", "neg_laf"]


def sphere(n, r_mm):
    g = (np.arange(n) - (n - 1) / 2.0) * PATCH_SPACING_MM
    zz, yy, xx = np.meshgrid(g, g, g, indexing="ij", sparse=True)
    return np.sqrt(zz ** 2 + yy ** 2 + xx ** 2) <= r_mm


def measures(a, mask, rois, core, shell):
    pos, neg = np.clip(a, 0, None), np.clip(a, None, 0)
    out = {}
    for r, roi in rois.items():
        t = pos[roi].sum()
        out[f"laf_r{r}"] = float(pos[roi & mask].sum() / t) if t > 0 else np.nan
    roi = rois[60]
    les = pos[mask].sum()
    rim = mask & ~core
    if les > 0 and core.any() and rim.any():
        out["core_share"] = float(pos[core].sum() / les)
        out["rim_share"] = float(pos[rim].sum() / les)
        dr = pos[rim].mean()
        out["core_density_ratio"] = float(pos[core].mean() / dr) if dr > 0 else np.nan
    else:
        out.update(core_share=np.nan, rim_share=np.nan, core_density_ratio=np.nan)
    t = pos[roi].sum()
    out["shell_share"] = float(pos[roi & shell].sum() / t) if t > 0 else np.nan
    tn = neg[roi].sum()
    out["neg_laf"] = float(neg[roi & mask].sum() / tn) if tn < 0 else np.nan
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--saliency", default=RESULTS_DIR / "journal" / "saliency_resnet", type=Path)
    ap.add_argument("--out", default=RESULTS_DIR / "journal" / "analysis_resnet" / "map_measures", type=Path)
    ap.add_argument("--explainers", default="ig,gradcam")
    args = ap.parse_args()
    sal = args.saliency
    expl = [e for e in args.explainers.split(",") if (sal / "maps" / e).exists()]
    if not expl:
        raise SystemExit(f"no maps for {args.explainers} under {sal / 'maps'}")
    seeds = sorted(int(p.name[4:]) for p in (sal / "maps" / expl[0]).glob("seed*"))
    names = sorted(p.stem for p in (sal / "masks").glob("*.npz"))
    rows, rois = [], None
    k_erode = int(round(2.0 / PATCH_SPACING_MM))
    k_shell = int(round(5.0 / PATCH_SPACING_MM))
    for k, name in enumerate(names, 1):
        pid, acq, setting = name.rsplit("_", 2)
        mask = np.load(sal / "masks" / f"{name}.npz")["mask"].astype(bool)
        n = mask.shape[0]
        if rois is None:
            rois = {r: sphere(n, r * n / 128) for r in (30, 45, 60)}
        core = binary_erosion(mask, iterations=k_erode)
        shell = binary_dilation(mask, iterations=k_shell) & ~mask
        for seed in seeds:
            for e in expl:
                a = S7.load_map(sal, e, seed, name, (n,) * 3)
                if a is None:
                    continue
                rows.append(dict(explainer=e, seed=seed, PatientID=pid, acq=acq, setting=setting, lesion_voxels=int(mask.sum()),
                                 core_voxels=int(core.sum()), **measures(a, mask, rois, core, shell)))
        if k % 50 == 0:
            print(f"  {k}/{len(names)} cells", flush=True)
    df = pd.DataFrame(rows)
    args.out.mkdir(parents=True, exist_ok=True)
    df.round(5).to_csv(args.out / "map_measures_cells.csv", index=False)
    summ = []
    for (e, meas), g in df.melt(id_vars=["explainer", "seed", "PatientID", "acq", "setting"], value_vars=MEASURES,
                                var_name="measure", value_name="value").groupby(["explainer", "measure"]):
        r = S18.reliability(g.rename(columns={"value": "laf"}))
        summ.append(dict(explainer=e, measure=meas, n_valid=int(g.value.notna().sum()), **r))
    sm = pd.DataFrame(summ).rename(columns={"laf_median": "median"})
    sm.round(4).to_csv(args.out / "map_measures_summary.csv", index=False)
    print(sm.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
