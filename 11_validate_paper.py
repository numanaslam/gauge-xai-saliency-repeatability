#!/usr/bin/env python3
"""
GAUGE-XAI step 11: results-validation sheet. Every number in paper/main.tex is recomputed from the result
files (or, for values only ever printed on the server, taken from paper/logged_values.json) and compared
with the text. Writes paper/validation_sheet.md and .csv.

Status:  MATCH     recomputed value, formatted as in the paper, equals the paper's value
         CONFIG    design constant, equal to the default in the code that produced the results
         MISMATCH  it does not - fix the paper or find out why
         LOGGED    value transcribed from server output (not recomputable locally); shown with its source
         MISSING   the paper no longer contains the text this check looks for (text changed - update the check)

Usage (from code/):  python 11_validate_paper.py
"""
import importlib.util
import json
from decimal import ROUND_HALF_UP, Decimal
import re
from pathlib import Path

import numpy as np
import pandas as pd

from gauge_lib import PRIMARY_EXCLUDE, RESULTS_DIR

ROOT = Path(__file__).resolve().parent.parent
A = RESULTS_DIR / "analysis"
R = A / "robustness"
AU = RESULTS_DIR / "audit"
P = ["1.25L", "1.25S", "2.5L", "5L", "5S"]  # primary reconstructions


def load_by_path(name, fname):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(fname))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def f(x, d=2):
    """Round half up (the paper's convention), not Python's binary/half-even rounding."""
    q = Decimal(1).scaleb(-d)
    return str(Decimal(repr(float(x))).quantize(q, rounding=ROUND_HALF_UP))


def rng(a, b, d=2, sep="--"):
    return f"{f(a, d)}{sep}{f(b, d)}"


def pct(x):
    return f(100 * x, 0)


def main():
    tex = (ROOT / "paper" / "main.tex").read_text()
    flat = re.sub(r"\s+", " ", tex)  # line breaks in the source do not matter
    logged = json.loads((ROOT / "paper" / "logged_values.json").read_text())
    rep = pd.read_csv(A / "repeatability.csv")
    con = pd.read_csv(A / "conditions.csv")
    tm = pd.read_csv(A / "table_main.csv").set_index(["explainer", "measure", "source"])
    mc = pd.read_csv(A / "map_conditions.csv")
    stab = pd.read_csv(A / "stability.csv")
    summ = (A / "summary.txt").read_text()
    pr = pd.read_csv(R / "rider_predictions.csv")
    lr = pd.read_csv(R / "laf_adj_repeatability.csv")
    ctl = pd.read_csv(R / "controls.csv")
    sc = pd.read_csv(R / "seeds_cluster_bootstrap.csv")
    vc = pd.read_csv(R / "variance_components.csv").set_index(["explainer", "measure", "component"])
    ls = pd.read_csv(AU / "lesion_stats.csv")
    cm = pd.read_csv(AU / "condition_manifest.csv")
    acq = pd.read_csv(AU / "acquisitions.csv")

    rows = []

    def check(section, claim, paper, recomputed, source, status=None):
        present = re.sub(r"\s+", " ", paper) in flat
        st = status or ("MATCH" if paper == recomputed else "MISMATCH")
        if not present:
            st = "MISSING"
        rows.append(dict(section=section, claim=claim, paper=paper, recomputed=recomputed, source=source, status=st))

    def logged_check(section, claim, paper, recomputed, key):
        check(section, claim, paper, recomputed, f"LOGGED: {logged[key]['source']}", status="LOGGED")

    # ---- shared quantities -------------------------------------------------------------------
    prim = rep[(rep.population == "primary")]
    ig125 = prim[(prim.explainer == "ig") & (prim.setting == "1.25L") & (prim.measure == "laf")].sort_values("seed")
    allpop = rep[(rep.population == "all") & (rep.explainer == "ig") & (rep.setting == "1.25L") & (rep.measure == "laf")]
    auc = logged["val_auc"]["value"]
    le10, gt10 = logged["val_auc_le10mm"]["value"], logged["val_auc_gt10mm"]["value"]
    sens = pr[pr["size"] == "all"]
    vcs = lambda e, ms, c, col="sd": vc.loc[(e, ms, c), col]  # noqa: E731
    tmw = lambda e, ms, s: tm.loc[(e, ms, s), "wSD"]  # noqa: E731
    ctl_med = ctl.groupby("explainer")[["laf_check", "laf_shift", "laf_mismatch", "laf_abs"]].median()
    scs = sc[sc.setting.isin(P)].groupby(["explainer", "measure"])[["icc", "icc_ci_lo", "icc_ci_hi", "wSD"]].median()
    mcp = mc[~mc.contrast.str.contains("2.5S")].groupby(["explainer", "contrast"])[["spearman", "spearman_8"]].median()
    g = lambda pat: float(re.search(pat, summ).group(1))  # noqa: E731
    ig_gap = g(r"IG completeness \|gap\|: median ([0-9.]+)")
    ig_chance = g(r"ig: median LAF [0-9.]+ vs chance ([0-9.]+)")
    ig_enr = g(r"ig: median LAF [0-9.]+ vs chance [0-9.]+ \(enrichment ([0-9.]+)x")
    gc_laf = g(r"gradcam: median LAF ([0-9.]+)")
    gc_chance = g(r"gradcam: median LAF [0-9.]+ vs chance ([0-9.]+)")
    gc_enr = g(r"gradcam: median LAF [0-9.]+ vs chance [0-9.]+ \(enrichment ([0-9.]+)x")
    gc_empty = g(r"gradcam: .*?empty maps ([0-9.]+)%")

    # ---- abstract (147 words since the move to the official template; other numbers checked in Results) ----
    check("Abstract", "validation AUC range", "validation AUC 0.947--0.969", f"validation AUC {rng(min(auc), max(auc), 3)}",
          "logged_values.json val_auc (5 runs)")
    check("Abstract", "primary ICC across runs", "ICC 0.974--0.994", f"ICC {rng(ig125.icc.min(), ig125.icc.max(), 3)}",
          "repeatability.csv: primary, ig, 1.25L, laf; min-max over seeds")
    check("Abstract", "RC % of median", "12--18\\%", f"{pct(ig125.RC_over_median.min())}--{pct(ig125.RC_over_median.max())}\\%",
          "repeatability.csv RC_over_median, primary endpoint rows")
    ratio = tmw("ig", "laf", "training seed") / tmw("ig", "laf", "scan-rescan")
    check("Abstract", "run/scan wSD ratio", "3.4 times", f"{f(ratio, 1)} times", "table_main.csv wSD seed / wSD scan-rescan")
    r6 = vcs("ig", "laf", "patient x seed") / vcs("ig", "laf", "patient x scan")
    check("Abstract", "VC run/scan SD ratio ('six times')", "component was six times that of the lesion-by-scan component", "component was six times that of the lesion-by-scan component" if 5.5 <= r6 < 6.5 else f"{r6:.1f} times",
          f"variance_components.csv: {r6:.2f}")
    igm = mcp.loc["ig"]
    check("Abstract", "IG 1 mm map Spearman", "Spearman 0.36--0.50", f"Spearman {rng(igm.spearman.min(), igm.spearman.max())}",
          "map_conditions.csv ig, primary contrasts, median per contrast, min-max")

    # ---- data --------------------------------------------------------------------------------
    for k, txt in (("lidc_patients", "1{,}010 patients"), ("lidc_ct_series", "1{,}018 CT scans"),
                   ("lidc_nodules", "2{,}684 nodules"), ("lidc_nodules_4_readers", "906 marked by all four"),
                   ("lidc_benign", "(449)"), ("lidc_malignant", "malignant (431)"), ("train_nodules", "train 697 nodules"),
                   ("train_patients", "428 patients"), ("val_nodules", "validation 183/107")):
        logged_check("Data", k, txt, str(logged[k]["value"]), k)
    gap = acq[acq.role == "scan2"].rescan_interval_min.median()
    check("Data", "median rescan interval", "median interval 7.7~min", f"median interval {f(gap, 1)}~min", "audit/acquisitions.csv")
    n_seg = int((cm.usable & cm.seg_file.notna()).sum())
    check("Data", "cells with CT + tumour SEG", "359 had a CT", f"{n_seg} had a CT", "audit/condition_manifest.csv usable")
    one = ls[ls.n_frames < 2]
    check("Data", "rejected single-frame SEGs", "6 SEGs outlining", f"{len(one)} SEGs outlining", "audit/lesion_stats.csv n_frames < 2")
    xyz = ["centroid_x", "centroid_y", "centroid_z"]
    med = ls[ls.n_frames >= 2].groupby(["PatientID", "acq"])[xyz].median()
    d = np.sqrt(((one.set_index(["PatientID", "acq"])[xyz] - med.reindex(one.set_index(["PatientID", "acq"]).index)) ** 2)
                .sum(1))
    check("Data", "rejected SEG distance from tumour", "77--213~mm", f"{d.min():.0f}--{d.max():.0f}~mm",
          "lesion_stats centroids vs acquisition median of valid SEGs")
    check("Data", "usable cells", "leaving 353", f"leaving {n_seg - len(one)}", "359 - 6")
    s0 = prim[(prim.explainer == "ig") & (prim.measure == "laf") & (prim.seed == 0)].set_index("setting").n
    check("Data", "paired patients at 1.25L", "28 patients at 1.25L", f"{int(s0['1.25L'])} patients at 1.25L", "repeatability.csv n")
    others = s0[[s for s in P if s != "1.25L"]]
    check("Data", "paired patients, other primary", "29--30 at the other", f"{int(others.min())}--{int(others.max())} at the other",
          "repeatability.csv n")
    good = ls[ls.n_frames >= 2]
    a125 = good[(good.setting == "1.25L") & (good.acq == "scan1")].longest_axial_mm.median()
    check("Data", "median longest axial diameter", "diameter 30~mm at 1.25L", f"diameter {a125:.0f}~mm at 1.25L",
          "lesion_stats 1.25L scan1, valid SEGs")
    n30 = int((good.groupby("PatientID").longest_axial_mm.max() > 30).sum())
    check("Data", "tumours > 30 mm", "18 of 32 exceed", f"{n30} of {good.PatientID.nunique()} exceed", "lesion_stats per-patient max")
    PC = load_by_path("positive_control", "03_positive_control.py")
    pcd = good[~good.PatientID.isin(PRIMARY_EXCLUDE)]

    def pc(measure):
        w = pcd[pcd.setting == "1.25L"].pivot_table(index="PatientID", columns="acq", values=measure)[["scan1", "scan2"]].dropna()
        return PC.summarise(w.to_numpy(float), "log")

    v, dm = pc("volume_ml"), pc("longest_axial_mm")
    check("Data", "SEG volume ICC", "ICC 0.995 and", f"ICC {f(v['icc'], 3)} and", "03 on lesion_stats (valid SEGs, primary)")
    check("Data", "SEG volume LoA", "$-29\\%$ to $+37\\%$", f"$-{abs(v['BA_loa_lo']):.0f}\\%$ to $+{v['BA_loa_hi']:.0f}\\%$",
          "03 Bland-Altman % limits, 1.25L")
    check("Data", "diameter LoA", "within $\\pm13\\%$", f"within $\\pm{max(abs(dm['BA_loa_lo']), dm['BA_loa_hi']):.0f}\\%$",
          "03 Bland-Altman % limits, longest axial")

    # ---- model ------------------------------------------------------------------------------
    logged_check("Model", "no-prior AUC", "validation AUC 0.75", f(logged["val_auc_no_prior_128"]["value"]),
                 "val_auc_no_prior_128")
    check("Model", "AUC range", "Validation AUC was 0.947--0.969", f"Validation AUC was {rng(min(auc), max(auc), 3)}",
          "logged_values.json val_auc")
    check("Model", "diameter-only AUC", "vs.\\ 0.938 for diameter alone",
          f"vs.\\ {f(logged['diameter_only_val_auc']['value'], 3)} for diameter alone", "logged_values.json")
    check("Model", "AUC > 10 mm", "(0.92--0.96 vs.\\ 0.86)", f"({rng(min(gt10), max(gt10))} vs.\\ "
          f"{f(logged['diameter_only_val_auc_gt10mm']['value'])})", "logged_values.json")
    n_above = sum(x > logged["diameter_only_val_auc_le10mm"]["value"] + 0.005 for x in le10)
    check("Model", "runs above diameter <= 10 mm", "for four of five runs", f"for {['zero','one','two','three','four','five'][n_above]} of five runs",
          "logged AUC <=10 mm vs 0.7825 (+0.005 margin)")
    check("Model", "AUC <= 10 mm", "(0.78--0.90 vs.\\ 0.78)", f"({rng(min(le10), max(le10))} vs.\\ "
          f"{f(logged['diameter_only_val_auc_le10mm']['value'])})", "logged_values.json")
    check("Model", "RIDER median sensitivity", "a median of 65\\%", f"a median of {pct(sens.sensitivity_logit_gt0.median())}\\%",
          "rider_predictions.csv size=all")
    prim_s = sens[sens.setting.isin(P)]
    check("Model", "series per run x reconstruction", "59--61 series from 31 patients",
          f"{prim_s.n_cells.min()}--{prim_s.n_cells.max()} series from {sens.n_patients.max()} patients", "rider_predictions.csv")
    check("Model", "series at 2.5S", "40 at 2.5S", f"{int(sens[sens.setting == '2.5S'].n_cells.max())} at 2.5S", "rider_predictions.csv")
    check("Model", "sensitivity range", "(48--95\\%", f"({pct(sens.sensitivity_logit_gt0.min())}--{pct(sens.sensitivity_logit_gt0.max())}\\%",
          "rider_predictions.csv size=all")
    bys = pr[pr["size"] != "all"].groupby("size")[["sensitivity_logit_gt0", "logit_median"]].median()
    check("Model", "sensitivity <= 30 mm", "81\\% for tumours", f"{pct(bys.loc['<=30 mm', 'sensitivity_logit_gt0'])}\\% for tumours",
          "rider_predictions.csv size strata (median)")
    check("Model", "sensitivity > 30 mm", "but 41\\% above", f"but {pct(bys.loc['>30 mm', 'sensitivity_logit_gt0'])}\\% above", "same")
    check("Model", "median logit > 30 mm", "$-0.30$), i.e.", f"${bys.loc['>30 mm', 'logit_median']:.2f}$), i.e.", "same")

    # ---- measurands -------------------------------------------------------------------------
    check("Measurands", "IG completeness gap", "gap 0.07 logit", f"gap {ig_gap:.2f} logit",
          "summary.txt")
    check("Measurands", "chance level", "(median 0.009)", f"(median {ig_chance:.3f})",
          "summary.txt explainer sanity")

    # ---- results: variance components ----------------------------------------------------
    check("Results/VC", "IG between-lesion share", "IG LAF, 80\\%", f"IG LAF, {vcs('ig', 'laf', 'patient', 'pct_of_random'):.0f}\\%",
          "variance_components.csv")
    pxs = vc.loc[("ig", "laf", "patient x seed")]
    check("Results/VC", "IG patient x run SD [CI] %", "(SD 0.064 [0.054, 0.079], 17\\%)",
          f"(SD {f(pxs.sd, 3)} [{f(pxs.sd_ci_lo, 3)}, {f(pxs.sd_ci_hi, 3)}], {pxs.pct_of_random:.0f}\\%)", "variance_components.csv")
    acqs = [vcs("ig", "laf", c) for c in ("patient x scan", "patient x thickness", "patient x kernel")]
    check("Results/VC", "acquisition components", "(each 0.011, 0.5\\%)",
          f"(each {f(np.mean(acqs), 3)}, {np.mean([vcs('ig', 'laf', c, 'pct_of_random') for c in ('patient x scan', 'patient x thickness', 'patient x kernel')]):.1f}\\%)"
          if max(acqs) - min(acqs) < 0.0015 else f"(not equal: {[round(x, 4) for x in acqs]})", "variance_components.csv")
    check("Results/VC", "leave-one-run-out range", "(0.056--0.079)",
          f"({rng(pxs.sd_leave_one_seed_out_min, pxs.sd_leave_one_seed_out_max, 3)})", "variance_components.csv")
    for comp, txt in (("patient x seed", "run (34\\%)"), ("patient x kernel", "kernel (26\\%)"), ("patient", "component (20\\%)")):
        val = vcs("gradcam", "laf", comp, "pct_of_random")
        check("Results/VC", f"Grad-CAM {comp} share", txt, re.sub(r"\(\d+", f"({val:.0f}", txt), "variance_components.csv")
    check("Results/VC", "Grad-CAM scan component", "with no scan component",
          "with no scan component" if vcs("gradcam", "laf", "patient x scan") < 0.001 else "scan component present",
          "variance_components.csv")
    check("Results/VC", "logit run share", "largest (36\\%)", f"largest ({vcs('ig', 'logit', 'patient x seed', 'pct_of_random'):.0f}\\%)",
          "variance_components.csv")
    lx = vc.loc[("ig", "logit", "patient x seed")]
    check("Results/VC", "logit leave-one-run-out vs full", "SD 0.89--1.19 vs.\\ 1.51",
          f"SD {rng(lx.sd_leave_one_seed_out_min, lx.sd_leave_one_seed_out_max)} vs.\\ {f(lx.sd)}", "variance_components.csv")
    for src, txt in (("scan-rescan", "0.021 between scans"), ("thickness", "0.022 between thicknesses"),
                     ("kernel", "0.026 between kernels"), ("training seed", "0.070 between runs")):
        check("Results/VC", f"pairwise wSD {src}", txt, f"{f(tmw('ig', 'laf', src), 3)} {txt.split(' ', 1)[1]}", "table_main.csv")
    for meas, txt in (("laf", "ICC 0.81 [0.69, 0.87]"), ("logit", "0.84 [0.78, 0.88]")):
        r = scs.loc[("ig", meas)]
        rec = f"{f(r.icc)} [{f(r.icc_ci_lo)}, {f(r.icc_ci_hi)}]"
        check("Results/VC", f"between-run ICC {meas}", txt, ("ICC " if txt.startswith("ICC") else "") + rec,
              "seeds_cluster_bootstrap.csv, median over primary settings")
    check("Results/VC", "logit wSD runs vs scans", "SD 1.19 logit vs.\\ 0.28",
          f"SD {f(scs.loc[('ig', 'logit'), 'wSD'])} logit vs.\\ {f(tmw('ig', 'logit', 'scan-rescan'))}",
          "seeds_cluster (runs) and table_main (scans)")
    # Table 1
    for e, col in (("ig", "IG"), ("gradcam", "Grad-CAM"), ("ig_logit", "Logit")):
        ee, ms = ("ig", "logit") if e == "ig_logit" else (e, "laf")
        for comp, lab in (("patient", "patient"), ("patient x scan", "pat.$\\times$scan"),
                          ("patient x thickness", "pat.$\\times$thickness"), ("patient x kernel", "pat.$\\times$kernel"),
                          ("patient x seed", "pat.$\\times$run"), ("residual", "residual")):
            r = vc.loc[(ee, ms, comp)]
            fmt = (lambda x: f(x, 2)) if ms == "logit" else (lambda x: f(x, 3)[1:] if x < 1 else f(x, 3))
            pv = r.pct_of_random
            pv_s = f(pv, 1) if (pv < 2 and pv >= 0.05) else f(pv, 0)
            cell = f"{fmt(r.sd)} [{fmt(r.sd_ci_lo)}, {fmt(r.sd_ci_hi)}] {pv_s}\\%"
            m_ = re.search(re.escape(lab) + r"\s*&(.*?)\\\\", tex)
            cells = [c.strip() for c in m_.group(1).split("&")] if m_ else ["", "", ""]
            paper = cells[{"IG": 0, "Grad-CAM": 1, "Logit": 2}[col]] if len(cells) == 3 else ""
            check("Table 1", f"{col} {comp}", paper or f"(row {lab} not found)", cell, "variance_components.csv")

    # ---- results: primary endpoint ---------------------------------------------------------
    low = ig125.loc[ig125.icc.idxmin()]
    check("Results/primary", "lowest-run ICC [CI]", "(lowest: 0.974 [0.942, 0.990])",
          f"(lowest: {f(low.icc, 3)} [{f(low.icc_ci_lo, 3)}, {f(low.icc_ci_hi, 3)}])", "repeatability.csv primary endpoint")
    check("Results/primary", "all runs conclusive excellent", "each conclusively ``excellent''",
          "each conclusively ``excellent''" if (ig125.koo.eq("excellent") & ig125.koo_conclusive).all() else "not all",
          "repeatability.csv koo, koo_conclusive")
    check("Results/primary", "RC range", "RC 0.042--0.065", f"RC {rng(ig125.RC.min(), ig125.RC.max(), 3)}", "repeatability.csv RC")
    check("Results/primary", "max |bias|", "bias $\\leq$0.012", f"bias $\\leq${f(ig125.BA_mean.abs().max(), 3)}", "repeatability.csv BA_mean")
    check("Results/primary", "LOO minimum", "below 0.969", f"below {f(ig125.icc_loo_min.min(), 3)}", "repeatability.csv icc_loo_min")
    check("Results/primary", "%GRR range", "(\\%GRR 8--16\\%)", f"(\\%GRR {ig125.pct_grr.min():.0f}--{ig125.pct_grr.max():.0f}\\%)",
          "repeatability.csv pct_grr")
    dd = (allpop.set_index("seed").icc - ig125.set_index("seed").icc).abs().max()
    check("Results/primary", "effect of the excluded patient", "ICC by $\\leq$0.002", f"ICC by $\\leq${f(dd, 3)}",
          "repeatability.csv population all vs primary")

    # ---- results: strata (size, predicted class) --------------------------------------------
    sp = R / "strata_summary.csv"
    labels = (("<=30 mm", "0.020 for tumours"), (">30 mm", "vs.\\ 0.021 above"), ("logit>0", "0.019 for a positive"),
              ("logit<=0", "vs.\\ 0.022 for a negative"))
    if sp.exists():
        ss = pd.read_csv(sp).set_index(["explainer", "stratum", "level"])
        for level, txt in labels:
            kind = "size" if "mm" in level else "pred"
            val = f(ss.loc[("ig", kind, level), "wSD_median"], 3)
            check("Results/strata", f"IG wSD {level}", txt, re.sub(r"0\.\d{3}", val, txt), "strata_summary.csv ig wSD_median")
    else:
        for level, txt in labels:
            logged_check("Results/strata", f"IG wSD {level}", txt, f(logged["strata_ig_wsd"]["value"][level], 3), "strata_ig_wsd")

    # ---- results: controls ------------------------------------------------------------------
    check("Results/controls", "IG enrichment", "IG placed 36.5$\\times$", f"IG placed {ig_enr:.1f}$\\times$",
          "summary.txt")
    check("Results/controls", "IG median LAF", "(median LAF 0.38;", f"(median LAF {f(ctl_med.loc['ig', 'laf_check'])};", "controls.csv laf_check")
    lri = lr[(lr.explainer == "ig") & lr.setting.isin(P)]
    check("Results/controls", "LAF_adj ICC", "ICC 0.98 [0.94--0.99", f"ICC {f(lri.icc.median())} [{rng(lri.icc.min(), lri.icc.max())}",
          "laf_adj_repeatability.csv ig, primary settings")
    check("Results/controls", "shifted SEG LAF", "LAF fell to 0.025", f"LAF fell to {f(ctl_med.loc['ig', 'laf_shift'], 3)}", "controls.csv")
    check("Results/controls", "foreign SEG LAF", "centre captured 0.30,", f"centre captured {f(ctl_med.loc['ig', 'laf_mismatch'])},",
          "controls.csv")
    check("Results/controls", "|a| LAF", "LAF was 0.32", f"LAF was {f(ctl_med.loc['ig', 'laf_abs'])}", "controls.csv laf_abs")
    check("Results/controls", "Grad-CAM enrichment", "Grad-CAM placed 2.1$\\times$",
          f"Grad-CAM placed {gc_enr:.1f}$\\times$", "summary.txt")
    check("Results/controls", "Grad-CAM LAF vs chance", "(0.014 vs.\\ 0.009;",
          f"({gc_laf:.3f} vs.\\ {gc_chance:.3f};", "summary.txt")
    check("Results/controls", "Grad-CAM empty maps", "empty maps 2.2\\%)", f"empty maps {gc_empty:.1f}\\%)",
          "summary.txt")
    check("Results/controls", "Grad-CAM shifted / foreign", "shifted (0.012) or foreign SEG (0.014)",
          f"shifted ({f(ctl_med.loc['gradcam', 'laf_shift'], 3)}) or foreign SEG ({f(ctl_med.loc['gradcam', 'laf_mismatch'], 3)})",
          "controls.csv")
    check("Results/controls", "Grad-CAM scan-rescan ICC median", "(median 0.98)",
          f"(median {f(tm.loc[('gradcam', 'laf', 'scan-rescan'), 'icc_median'])})", "table_main.csv")
    gcr = prim[(prim.explainer == "gradcam") & (prim.measure == "laf") & prim.setting.isin(P)]
    check("Results/controls", "Grad-CAM RC > median share", "LAF in 64\\%", f"LAF in {pct((gcr.RC_over_median > 1).mean())}\\%",
          "repeatability.csv gradcam, primary settings")
    check("Results/controls", "layer3 LAF vs shifted", "(LAF 0.007 vs.\\ 0.012",
          f"(LAF {f(ctl_med.loc['gradcam_l3', 'laf_check'], 3)} vs.\\ {f(ctl_med.loc['gradcam_l3', 'laf_shift'], 3)}", "controls.csv")
    check("Results/controls", "layer3 between-run ICC", "between-run ICC 0.31", f"between-run ICC {f(scs.loc[('gradcam_l3', 'laf'), 'icc'])}",
          "seeds_cluster_bootstrap.csv")

    cmain = pd.read_csv(R / "controls.csv")
    if "laf_mismatch_outside" in cmain:
        ci = cmain[(cmain.explainer == "ig") & ~cmain.PatientID.isin(PRIMARY_EXCLUDE)].dropna(subset=["laf_mismatch_outside"])
        osh, ish = ci.laf_mismatch_outside / ci.laf_check, ci.laf_mismatch_inside_own / ci.laf_check
        dens = (ci.laf_mismatch_outside / (1 - ci.mismatch_overlap) / ci.laf_check).median()
        check("Results/controls", "mask overlap", "overlap (median 64\\%", f"overlap (median {pct(ci.mismatch_overlap.median())}\\%",
              "controls.csv ig, primary population, per lesion")
        check("Results/controls", "outside-own share", "received only 7\\% of the own-SEG share",
              f"received only {pct(osh.median())}\\% of the own-SEG share", "controls.csv laf_mismatch_outside / laf_check")
        check("Results/controls", "outside-own share IQR", "(IQR 5--10\\%;", f"(IQR {pct(osh.quantile(0.25))}--{pct(osh.quantile(0.75))}\\%;",
              "controls.csv")
        check("Results/controls", "density ratio ('one fifth')", "density one fifth of the lesion's",
              "density one fifth of the lesion's" if 0.17 <= dens <= 0.23 else f"density {dens:.2f}", f"controls.csv: {dens:.3f}")
        check("Results/controls", "overlapping part of lesion", "overlapping 64\\% of the lesion received 76\\%",
              f"overlapping {pct(ci.mismatch_overlap.median())}\\% of the lesion received {pct(ish.median())}\\%", "controls.csv")
        check("Discussion", "outside share 128 mm", "(7\\% of its lesion share", f"({pct(osh.median())}\\% of its lesion share",
              "controls.csv")

    # ---- results: reproducibility -------------------------------------------------------------
    igk = pd.concat([tm.loc[("ig", "laf", "thickness")], tm.loc[("ig", "laf", "kernel")]], axis=1).T
    check("Results/reproducibility", "IG thickness+kernel ICC (min)", "(ICC $\\geq$0.92 for every", f"(ICC $\\geq${f(igk.icc_min.min())} for every",
          "table_main.csv thickness/kernel icc_min-max")
    k125 = con[(con.contrast == "1.25L vs 1.25S") & (con.measure == "laf") & (con.explainer == "gradcam")]
    check("Results/reproducibility", "Grad-CAM kernel ICC at 1.25", "(ICC 0.07--0.69", f"(ICC {rng(k125.icc.min(), k125.icc.max())}",
          "conditions.csv gradcam 1.25L vs 1.25S")
    check("Results/reproducibility", "Grad-CAM exceed RC", "10--63\\% of lesions",
          f"{pct(k125.exceed_rc.min())}--{pct(k125.exceed_rc.max())}\\% of lesions", "conditions.csv exceed_rc")
    l125 = con[(con.contrast == "1.25L vs 1.25S") & (con.measure == "logit") & (con.explainer == "ig")]
    check("Results/reproducibility", "logit kernel bias", "by 0.57--1.32", f"by {rng(l125.BA_mean.abs().min(), l125.BA_mean.abs().max())}",
          "conditions.csv logit BA_mean (S - L sign)")
    check("Results/reproducibility", "logit exceed RC", "and 45--83\\%", f"and {pct(l125.exceed_rc.min())}--{pct(l125.exceed_rc.max())}\\%",
          "conditions.csv logit exceed_rc")

    # ---- results: map level, model vs explainer ---------------------------------------------
    check("Results/map", "IG 1 mm Spearman", "(Spearman 0.36--0.50)", f"(Spearman {rng(igm.spearman.min(), igm.spearman.max())})",
          "map_conditions.csv")
    check("Results/map", "IG 16 mm Spearman", "closely (0.95--0.98)", f"closely ({rng(igm.spearman_8.min(), igm.spearman_8.max())})",
          "map_conditions.csv spearman_8")
    check("Results/map", "Grad-CAM 16 mm kernel 1.25", "less (0.73 between", f"less ({f(mcp.loc[('gradcam', '1.25L vs 1.25S'), 'spearman_8'])} between",
          "map_conditions.csv")
    check("Results/instability", "Spearman range", "(Spearman $-0.12$ to 0.14",
          f"(Spearman ${stab.spearman_absdlogit_absdlaf.min():.2f}$ to {stab.spearman_absdlogit_absdlaf.max():.2f}", "stability.csv")
    check("Results/instability", "min p", "all $p>0.08$", f"all $p>{np.floor(stab.p.min() * 100) / 100:.2f}$", "stability.csv p (floored)")
    check("Discussion", "run/scan ratio", "LAF varied 3.4 times more", f"LAF varied {f(ratio, 1)} times more", "table_main.csv")

    # ---- centre-prior ablation (64 mm), from results/ablation ---------------------------------
    ab = RESULTS_DIR / "ablation"
    if (ab / "ablation_comparison.csv").exists():
        cmp_ = pd.read_csv(ab / "ablation_comparison.csv", index_col=0)
        v = lambda metric, arm: float(cmp_.loc[metric, arm])  # noqa: E731
        aucs = {arm: [json.loads((ab / f"summary_{arm}64_seed{k}.json").read_text())["best_val_auc"] for k in range(3)]
                for arm in ("prior", "noprior")}
        check("Results/ablation", "AUC with / without prior", "(0.962--0.971 with, 0.966--0.975 without the prior)",
              f"({rng(min(aucs['prior']), max(aucs['prior']), 3)} with, {rng(min(aucs['noprior']), max(aucs['noprior']), 3)} "
              f"without the prior)", "results/ablation/summary_*64_seed*.json")
        pa = pd.read_csv(ab / "measurands_prior64.csv")
        pb = pd.read_csv(ab / "measurands_noprior64.csv")
        k = ["PatientID", "acq", "setting"]
        same = set(map(tuple, pa[k].drop_duplicates().values)) == set(map(tuple, pb[k].drop_duplicates().values))
        ncell = pa[k].drop_duplicates().shape[0]
        nprim = pa[~pa.PatientID.isin(PRIMARY_EXCLUDE)].PatientID.nunique()
        check("Results/ablation", "same cells in both arms", "explain the same 353 cells (31 primary patients)",
              f"explain the same {ncell} cells ({nprim} primary patients)" if same else "cell sets differ",
              "measurands_prior64/noprior64.csv cell sets")
        ci = pd.read_csv(ab / "ablation_ci.csv").set_index("metric")

        def sgn(x, d=3):
            t = f(x, d)
            return f"$-{t[1:]}$" if t.startswith("-") else t

        def dci(metric, d=3):
            r = ci.loc[metric]
            return f"{sgn(r.difference, d)} [{sgn(r.diff_ci_lo, d)}, {sgn(r.diff_ci_hi, d)}]"

        for metric, lab in (("wSD_scan", "scan--rescan wSD"), ("wSD_kernel", "kernel wSD"), ("wSD_run", "between-run wSD"),
                            ("icc_primary", "primary ICC")):
            paper_txt = re.search(re.escape(lab) + r" (\$?-?[0-9.]+\$? \[[^\]]*\])", flat)
            check("Results/ablation", f"paired difference {metric}", f"{lab} {paper_txt.group(1)}" if paper_txt else f"{lab} ?",
                  f"{lab} {dci(metric)}", "ablation_ci.csv (paired patient-cluster bootstrap)")
        check("Results/ablation", "outside share with / without", "(7\\% vs.\\ 9\\%;",
              f"({pct(ci.loc['outside_share', 'prior'])}\\% vs.\\ {pct(ci.loc['outside_share', 'no-prior'])}\\%;", "ablation_ci.csv")
        check("Results/ablation", "outside share difference", "9\\%; 0.021 [0.012, 0.029])",
              f"{pct(ci.loc['outside_share', 'no-prior'])}\\%; {dci('outside_share')})", "ablation_ci.csv")
        check("Results/ablation", "logit SD with / without", "(scan--rescan SD 0.35 vs.\\ 0.54;",
              f"(scan--rescan SD {f(ci.loc['logit_wSD_scan', 'prior'])} vs.\\ {f(ci.loc['logit_wSD_scan', 'no-prior'])};",
              "ablation_ci.csv")
        check("Results/ablation", "logit SD difference", "0.54; 0.20 [0.12, 0.27])",
              f"{f(ci.loc['logit_wSD_scan', 'no-prior'])}; {dci('logit_wSD_scan', 2)})", "ablation_ci.csv")
        check("Results/ablation", "Grad-CAM between-run ICC", "between runs (ICC 0.75 vs.\\ 0.45)",
              f"between runs (ICC {f(v('Grad-CAM between-run ICC (cluster bootstrap)', 'prior'))} vs.\\ "
              f"{f(v('Grad-CAM between-run ICC (cluster bootstrap)', 'no-prior'))})", "ablation_comparison.csv")
        check("Results/ablation", "run SDs", "(SD 0.027 and 0.036)",
              f"(SD {f(v('IG VC SD x run', 'prior'), 3)} and {f(v('IG VC SD x run', 'no-prior'), 3)})", "ablation_comparison.csv")
        check("Results/ablation", "scan SDs", "component (0.018 and 0.023)",
              f"component ({f(v('IG VC SD x scan', 'prior'), 3)} and {f(v('IG VC SD x scan', 'no-prior'), 3)})",
              "ablation_comparison.csv")
        r1 = v("IG VC SD x run", "prior") / v("IG VC SD x scan", "prior")
        r2 = v("IG VC SD x run", "no-prior") / v("IG VC SD x scan", "no-prior")
        for sec, txt in (("Results/ablation", "had about 1.5 times the SD"), ("Abstract", "only 1.5 times at 64~mm")):
            check(sec, "run/scan ratio at 64 mm", txt, txt if 1.3 <= r1 <= 1.7 and 1.3 <= r2 <= 1.7 else f"{r1:.2f} / {r2:.2f}",
                  f"ablation_comparison.csv: {r1:.2f} (prior), {r2:.2f} (no prior)")
        if "IG outside-own share (per lesion, median)" in cmp_.index:
            check("Discussion", "outside share without prior", "9\\% without the prior",
                  f"{pct(v('IG outside-own share (per lesion, median)', 'no-prior'))}\\% without the prior", "ablation_comparison.csv")

    # ---- subgroup sizes and the RIDER > 30 mm negative share ---------------------------------
    if sp.exists():
        ssn = pd.read_csv(sp).set_index(["explainer", "stratum", "level"])
        nn = lambda kind, lev: int(ssn.loc[("ig", kind, lev), "n_patients"])  # noqa: E731
        for kind, lev, txt in (("size", "<=30 mm", "0.020 for tumours $\\leq$30~mm ({n})"), ("size", ">30 mm", "0.021 above ({n})"),
                               ("pred", "logit>0", "positive mean logit ({n})"), ("pred", "logit<=0", "negative one ({n})")):
            check("Results/strata", f"subgroup size {lev}", txt.format(n={"<=30 mm": 16, ">30 mm": 13, "logit>0": 18, "logit<=0": 11}[lev]),
                  txt.format(n=nn(kind, lev)), "strata_summary.csv n_patients (median over runs x reconstructions)")
    check("Limitations", "negative share above 30 mm", "(59\\% of tumour series above 30~mm had a negative logit",
          f"({pct(1 - bys.loc['>30 mm', 'sensitivity_logit_gt0'])}\\% of tumour series above 30~mm had a negative logit",
          "rider_predictions.csv size >30 mm (median)")

    # ---- design constants: checked against the code that ran ---------------------------------
    import gauge_lib as GL

    src = {n: (Path(__file__).with_name(n)).read_text() for n in ("05_train.py", "06_rider_saliency.py", "07_gauge_analysis.py",
                                                                  "10_robustness.py", "03_positive_control.py")}

    def default(fname, flag):
        m_ = re.search(r'"--' + flag + r'".*?default=([^,)]+)', src[fname])
        return m_.group(1).strip() if m_ else "?"

    def cfg(claim, paper, value, source):
        check("Design", claim, paper, value, source, status="CONFIG" if paper == value else "MISMATCH")

    cfg("patch grid", "isotropic $128^3$ grid", f"isotropic ${GL.PATCH_SIZE}^3$ grid", "gauge_lib.PATCH_SIZE")
    cfg("field of view", "(128~mm field of view", f"({GL.PATCH_SIZE * GL.PATCH_SPACING_MM:g}~mm field of view", "PATCH_SIZE x spacing")
    cfg("HU window", "clipped to $[-1000,400]$", f"clipped to $[{GL.HU_WINDOW[0]:g},{GL.HU_WINDOW[1]:g}]$", "gauge_lib.HU_WINDOW")
    cfg("ROI radius", "fixed ROI (sphere of radius 60~mm", f"fixed ROI (sphere of radius {GL.roi_radius_mm(GL.PATCH_SIZE):g}~mm", "gauge_lib.roi_radius_mm(128)")
    cfg("training epochs", "100 epochs", f"{default('05_train.py', 'epochs')} epochs", "05_train.py --epochs default")
    cfg("learning rate", "learning rate $10^{-3}$", "learning rate $10^{-3}$" if default("05_train.py", "lr") == "1e-3" else "?",
        "05_train.py --lr default")
    cfg("IG steps", "in 32 steps", f"in {default('06_rider_saliency.py', 'ig-steps')} steps", "06 --ig-steps default")
    cfg("bootstrap resamples", "from 2{,}000 patient-level bootstrap resamples",
        "from 2{,}000 patient-level bootstrap resamples" if default("07_gauge_analysis.py", "n-boot") == "2000" else "?",
        "07 --n-boot default")
    cfg("VC bootstrap refits", "bootstrap (200 refits)", f"bootstrap ({default('10_robustness.py', 'n-boot')} refits)",
        "10 --n-boot default")
    cfg("RC constant", "$=2.77\\,$wSD", "$=2.77\\,$wSD" if "2.77 * wsd" in src["07_gauge_analysis.py"] else "?",
        "07 agreement(): RC = 2.77 wSD")
    cfg("SEG shift", "SEG shifted 25~mm off-centre", "SEG shifted 25~mm off-centre" if "25.0 * n / PATCH_SIZE" in src["10_robustness.py"] else "?",
        "10 controls(): 25 mm at 128^3")
    cfg("design cells", "Of 384 possible cells", f"Of {32 * 2 * 6} possible cells", "32 patients x 2 scans x 6 reconstructions")
    try:
        import torch  # noqa: F401
        from gauge_model import build_model
        n_par = sum(p.numel() for p in build_model(centre_prior=True, n_aux=1).parameters()) / 1e6
        cfg("parameters", "(8.3\\,M parameters", f"({f(n_par, 1)}\\,M parameters", "gauge_model.build_model (centre prior, aux head)")
    except ImportError:
        check("Design", "parameters", "(8.3\\,M parameters", "torch not installed here", "gauge_model", status="LOGGED")

    df = pd.DataFrame(rows)
    out = ROOT / "paper"
    df.to_csv(out / "validation_sheet.csv", index=False)
    counts = df.status.value_counts().to_dict()
    md = ["# Results validation sheet", "", f"Generated by `code/11_validate_paper.py` from `results/` and "
          f"`paper/logged_values.json`. Status counts: {counts}.", "",
          "| # | Section | Claim | Paper | Recomputed | Source | Status |", "|---|---|---|---|---|---|---|"]
    for i, r in df.iterrows():
        md.append(f"| {i + 1} | {r.section} | {r.claim} | `{r.paper}` | `{r.recomputed}` | {r.source} | **{r.status}** |")
    (out / "validation_sheet.md").write_text("\n".join(md) + "\n")
    print(f"{len(df)} checks: {counts}")
    bad = df[df.status.isin(["MISMATCH", "MISSING"])]
    if len(bad):
        with pd.option_context("display.width", 250, "display.max_colwidth", 70):
            print(bad[["section", "claim", "paper", "recomputed", "status"]].to_string(index=False))
    print(f"Wrote {out / 'validation_sheet.md'} and validation_sheet.csv")


if __name__ == "__main__":
    main()
