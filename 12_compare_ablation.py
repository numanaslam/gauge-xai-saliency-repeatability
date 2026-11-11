#!/usr/bin/env python3
"""
GAUGE-XAI step 12: centre-prior ablation - side-by-side comparison of two analysed arms (same 64 mm field of
view, with vs. without the radial-distance channel), from the outputs of 07 + 10 for each arm.

Reads, per arm:  <arm>/summary.txt, repeatability.csv, conditions.csv, seeds.csv, map_conditions.csv,
                 <arm>/robustness/{variance_components,controls,seeds_cluster_bootstrap,rider_predictions,
                 laf_adj_repeatability}.csv
Writes:          <out>/ablation_comparison.csv and .md (one row per metric, one column per arm)

Usage:
  python 12_compare_ablation.py --arms D:\\paper8\\results\\ablation\\analysis_prior64 D:\\paper8\\results\\ablation\\analysis_noprior64 \\
                                --names prior no-prior --out D:\\paper8\\results\\ablation
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from gauge_lib import PRIMARY_EXCLUDE, PRIMARY_SETTING, SECONDARY_SETTINGS, SETTINGS

P = [s for s in SETTINGS if s not in SECONDARY_SETTINGS]


def metrics(a):
    """Dictionary of headline metrics for one analysed arm."""
    rob = a / "robustness"
    rep = pd.read_csv(a / "repeatability.csv")
    rep = rep[rep.population == "primary"]
    con = pd.read_csv(a / "conditions.csv")
    con = con[~con.contrast.str.contains("2.5S")]
    sd = pd.read_csv(a / "seeds.csv")
    mc = pd.read_csv(a / "map_conditions.csv")
    mc = mc[~mc.contrast.str.contains("2.5S")]
    vc = pd.read_csv(rob / "variance_components.csv").set_index(["explainer", "measure", "component"])
    ctl = pd.read_csv(rob / "controls.csv").groupby("explainer")[["laf_check", "laf_shift", "laf_mismatch", "laf_abs"]].median()
    sc = pd.read_csv(rob / "seeds_cluster_bootstrap.csv")
    sc = sc[sc.setting.isin(P)].groupby(["explainer", "measure"])[["icc", "icc_ci_lo", "icc_ci_hi", "wSD"]].median()
    pr = pd.read_csv(rob / "rider_predictions.csv")
    summ = (a / "summary.txt").read_text()
    out = {}
    for e in ("ig", "gradcam"):
        E = "IG" if e == "ig" else "Grad-CAM"
        m = re.search(rf"{e}: median LAF ([0-9.]+) vs chance ([0-9.]+) \(enrichment ([0-9.]+)x\); empty maps ([0-9.]+)%", summ)
        if m:
            out[f"{E} median LAF"] = float(m.group(1))
            out[f"{E} chance level"] = float(m.group(2))
            out[f"{E} enrichment (x chance)"] = float(m.group(3))
            out[f"{E} empty maps (%)"] = float(m.group(4))
        pe = rep[(rep.explainer == e) & (rep.setting == PRIMARY_SETTING) & (rep.measure == "laf")]
        out[f"{E} primary ICC (min over runs)"] = pe.icc.min()
        out[f"{E} primary ICC (max over runs)"] = pe.icc.max()
        out[f"{E} primary RC/median (max over runs)"] = pe.RC_over_median.max()
        r = rep[(rep.explainer == e) & (rep.measure == "laf") & rep.setting.isin(P)]
        out[f"{E} wSD scan-rescan"] = r.wSD.mean()
        out[f"{E} ICC scan-rescan (median)"] = r.icc.median()
        for kind in ("thickness", "kernel"):
            c = con[(con.explainer == e) & (con.measure == "laf") & (con.kind == kind)]
            out[f"{E} wSD {kind}"] = c.wSD.mean()
            out[f"{E} ICC {kind} (median)"] = c.icc.median()
            out[f"{E} ICC {kind} (min)"] = c.icc.min()
        s = sd[(sd.explainer == e) & (sd.measure == "laf") & sd.setting.isin(P)]
        out[f"{E} wSD between runs"] = s.wSD.mean()
        out[f"{E} between-run ICC (cluster bootstrap)"] = sc.loc[(e, "laf"), "icc"]
        out[f"{E} between-run ICC CI lo"] = sc.loc[(e, "laf"), "icc_ci_lo"]
        out[f"{E} between-run ICC CI hi"] = sc.loc[(e, "laf"), "icc_ci_hi"]
        for comp, lab in (("patient", "patient"), ("patient x scan", "x scan"), ("patient x kernel", "x kernel"),
                          ("patient x thickness", "x thickness"), ("patient x seed", "x run"), ("residual", "residual")):
            if (e, "laf", comp) in vc.index:
                out[f"{E} VC SD {lab}"] = vc.loc[(e, "laf", comp), "sd"]
                out[f"{E} VC % {lab}"] = vc.loc[(e, "laf", comp), "pct_of_random"]
        if e in ctl.index:
            out[f"{E} LAF own SEG (controls)"] = ctl.loc[e, "laf_check"]
            out[f"{E} LAF shifted SEG"] = ctl.loc[e, "laf_shift"]
            out[f"{E} LAF foreign size-matched SEG"] = ctl.loc[e, "laf_mismatch"]
            out[f"{E} foreign/own LAF ratio"] = ctl.loc[e, "laf_mismatch"] / ctl.loc[e, "laf_check"]
            out[f"{E} shape-specific LAF (own - foreign)"] = ctl.loc[e, "laf_check"] - ctl.loc[e, "laf_mismatch"]
            out[f"{E} |a| LAF"] = ctl.loc[e, "laf_abs"]
        cc = pd.read_csv(rob / "controls.csv")
        cc = cc[(cc.explainer == e) & ~cc.PatientID.isin(PRIMARY_EXCLUDE)]
        if "laf_mismatch_outside" in cc:
            cc = cc.dropna(subset=["laf_mismatch_outside"])
            out[f"{E} outside-own share (per lesion, median)"] = (cc.laf_mismatch_outside / cc.laf_check).median()
            out[f"{E} inside-own share (per lesion, median)"] = (cc.laf_mismatch_inside_own / cc.laf_check).median()
            out[f"{E} mask overlap (median)"] = cc.mismatch_overlap.median()
            out[f"{E} outside/lesion density ratio"] = (cc.laf_mismatch_outside / (1 - cc.mismatch_overlap) / cc.laf_check).median()
        mm = mc[mc.explainer == e].groupby("contrast")[["spearman", "spearman_8"]].median()
        out[f"{E} map Spearman 1 mm (min)"], out[f"{E} map Spearman 1 mm (max)"] = mm.spearman.min(), mm.spearman.max()
        out[f"{E} map Spearman 8^3 (min)"], out[f"{E} map Spearman 8^3 (max)"] = mm.spearman_8.min(), mm.spearman_8.max()
    s = pr[pr["size"] == "all"]
    out["RIDER sensitivity (median)"] = s.sensitivity_logit_gt0.median()
    out["logit between-run ICC"] = sc.loc[("ig", "logit"), "icc"]
    lr = rep[(rep.explainer == "ig") & (rep.measure == "logit") & rep.setting.isin(P)]
    out["logit wSD scan-rescan"] = lr.wSD.mean()
    if ("ig", "logit", "patient x seed") in vc.index:
        out["logit VC SD x run"] = vc.loc[("ig", "logit", "patient x seed"), "sd"]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arms", nargs=2, type=Path, required=True)
    ap.add_argument("--names", nargs=2, default=["prior", "no-prior"])
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    cols = {n: metrics(a) for n, a in zip(args.names, args.arms)}
    df = pd.DataFrame(cols)
    df["difference (2nd - 1st)"] = df[args.names[1]] - df[args.names[0]]
    args.out.mkdir(parents=True, exist_ok=True)
    df.round(4).to_csv(args.out / "ablation_comparison.csv")
    md = ["| metric | " + " | ".join(df.columns) + " |", "|---" * (len(df.columns) + 1) + "|"]
    md += [f"| {i} | " + " | ".join(f"{v:.4f}" if isinstance(v, float) and np.isfinite(v) else str(v) for v in r) + " |"
           for i, r in df.iterrows()]
    (args.out / "ablation_comparison.md").write_text("\n".join(md) + "\n")
    with pd.option_context("display.width", 200, "display.max_rows", 200):
        print(df.round(4).to_string())
    print(f"\nWrote {args.out / 'ablation_comparison.csv'} and .md")


if __name__ == "__main__":
    main()
