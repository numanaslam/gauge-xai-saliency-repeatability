#!/usr/bin/env python3
"""
GAUGE-XAI step 17: results-validation sheet for the journal manuscript (paper_cmc/main.tex).

Each check rebuilds a phrase of the manuscript from the result tables (results/journal/..., results/ablation/...)
and looks for it in the text, so a number that changes in the results, or a phrase that is edited in the text,
is caught. Rounding is half up, as in the paper.

Status:  MATCH     the rebuilt phrase is in the manuscript
         MISMATCH  the anchor text is there but followed by a different value (shown in 'paper')
         MISSING   not even the anchor text is found (text was reworded - update the check)
         CONFIG    design constant, equal to the default in the code that produced the results
         LOGGED    value only printed on the server (paper/logged_values.json)

Writes paper_cmc/validation_sheet.md and .csv.   Usage (from code/):  python 17_validate_cmc.py
"""
import importlib.util
import json
import os
import re
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import numpy as np
import pandas as pd

from gauge_lib import PRIMARY_EXCLUDE, RESULTS_DIR

ROOT = Path(__file__).resolve().parent.parent
J = RESULTS_DIR / "journal"
AB = RESULTS_DIR / "ablation"
P = ["1.25L", "1.25S", "2.5L", "5L", "5S"]
ARCH = ("resnet", "densenet")


def f(x, d=2):
    return str(Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP))


def f0(x, d=2):
    """As f, without the leading zero (.141), as in the variance-component table."""
    s = f(x, d)
    return s[1:] if s.startswith("0.") else s


def pct(x, d=0):
    return f(100 * x, d)


def rng(a, b, d=2):
    return f"{f(a, d)}--{f(b, d)}"


def ci(lo, hi, d=3):
    g = lambda v: f"$-{f(-v, d)}$" if float(f(v, d)) < 0 else f(v, d)  # noqa: E731
    return f"[{g(lo)}, {g(hi)}]"


def load(a):
    d = J / f"analysis_{a}"
    rb = d / "robustness"
    return dict(
        rep=pd.read_csv(d / "repeatability.csv"), con=pd.read_csv(d / "conditions.csv"), sd=pd.read_csv(d / "seeds.csv"),
        mc=pd.read_csv(d / "map_conditions.csv"), stab=pd.read_csv(d / "stability.csv"), summ=(d / "summary.txt").read_text(),
        vc=pd.read_csv(rb / "variance_components.csv").set_index(["explainer", "measure", "component"]),
        sc=pd.read_csv(rb / "seeds_cluster_bootstrap.csv"), ctl=pd.read_csv(rb / "controls.csv"),
        pr=pd.read_csv(rb / "rider_predictions.csv"), strd=pd.read_csv(rb / "strata_differences.csv"),
        fs=pd.read_csv(d / "faithfulness" / "faithfulness_summary.csv"), san=pd.read_csv(d / "faithfulness" / "sanity_summary.csv"),
        reg=pd.read_csv(d / "registration" / "registration_summary.csv"), sp=pd.read_csv(d / "registration" / "scan_pairs.csv"),
        rfm=pd.read_csv(d / "registration" / "recon_fixed_mask_summary.csv"), meas=pd.read_csv(J / f"saliency_{a}" / "measurands.csv"),
        runs=[json.loads((J / f"runs_{a}" / f"seed{s}" / "summary.json").read_text()) for s in range(10)])


def main():
    paper_dir = os.environ.get("PAPER_DIR", "paper_cmc")   # e.g. PAPER_DIR=paper_diagnostics
    tex = (ROOT / paper_dir / "main.tex").read_text()
    flat = re.sub(r"\s+", " ", re.sub(r"(?m)^%.*$", "", tex))
    rows = []

    def check(section, claim, anchor, value, source):
        """MATCH if 'anchor + value' is in the text; otherwise report what follows the anchor."""
        phrase = re.sub(r"\s+", " ", anchor + value)
        if phrase in flat:
            st, paper = "MATCH", value
        elif re.sub(r"\s+", " ", anchor) in flat:
            i = flat.index(re.sub(r"\s+", " ", anchor)) + len(re.sub(r"\s+", " ", anchor))
            st, paper = "MISMATCH", flat[i:i + max(12, len(value) + 6)]
        else:
            st, paper = "MISSING", ""
        rows.append(dict(section=section, claim=claim, paper=paper, recomputed=value, source=source, status=st))

    D = {a: load(a) for a in ARCH}
    spec = importlib.util.spec_from_file_location("pc", Path(__file__).with_name("03_positive_control.py"))
    S7PC = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(S7PC)
    R, N = D["resnet"], D["densenet"]

    # ---- model performance ----------------------------------------------------------------------
    for a, lab in (("resnet", "ResNet"), ("densenet", "DenseNet")):
        auc = [r["best_val_auc"] for r in D[a]["runs"]]
        le, gt = [r["val_auc_le10mm"] for r in D[a]["runs"]], [r["val_auc_gt10mm"] for r in D[a]["runs"]]
        dle = D[a]["runs"][0]["diameter_only_val_auc_le10mm"]
        if a == "resnet":
            check("Performance", "ResNet AUC range", "ResNet validation AUC was ", rng(min(auc), max(auc), 3), "runs_resnet/seed*/summary.json")
            check("Performance", "diameter-only AUC", "over the ten runs vs.\\ ", f(D[a]["runs"][0]["diameter_only_val_auc"], 3), "summary.json")
            check("Performance", ">10 mm range", "in every run (", rng(min(gt), max(gt)), "summary.json")
            check("Performance", "<=10 mm above diameter (margin 0.005, as in the ISBI sheet)", "and within $\\leq$10~mm in ", f"{['zero','one','two','three','four','five','six','seven','eight','nine','ten'][sum(v > dle + 0.005 for v in le)]} of ten ({rng(min(le), max(le))}", "summary.json")
        else:
            check("Performance", "DenseNet AUC range", "DenseNet AUC was ", rng(min(auc), max(auc), 3), "runs_densenet/seed*/summary.json")
            check("Performance", "DenseNet >10 / <=10 mm", "(", f"{rng(min(gt), max(gt))} above 10~mm; {rng(min(le), max(le))}, "
                  f"{['zero','one','two','three','four','five','six','seven','eight','nine','ten'][sum(v > dle + 0.005 for v in le)]} of ten above diameter", "summary.json")
        s = D[a]["pr"]
        al = s[s["size"] == "all"].sensitivity_logit_gt0
        sm = s[s["size"] == "<=30 mm"].sensitivity_logit_gt0.median() if "<=30 mm" in set(s["size"]) else np.nan
        lg = s[s["size"] == ">30 mm"].sensitivity_logit_gt0.median() if ">30 mm" in set(s["size"]) else np.nan
        if a == "resnet":
            check("Performance", "RIDER sensitivity ResNet", "a median of ", f"{pct(al.median())}\\% of tumour series had a positive ResNet logit ({pct(al.min())}--{pct(al.max())}\\%", "rider_predictions.csv")
            check("Performance", "sensitivity by size ResNet", "over runs and reconstructions): ", f"{pct(sm)}\\% for tumours $\\leq$30~mm but {pct(lg)}\\% above 30~mm", "rider_predictions.csv")
        else:
            check("Performance", "RIDER sensitivity DenseNet", "For the DenseNet: ", f"{pct(al.median())}\\% ({pct(al.min())}--{pct(al.max())}\\%); {pct(sm)}\\% vs.\\ {pct(lg)}\\%", "rider_predictions.csv")
    for a, anchor in (("resnet", "The median IG completeness gap was "), ("densenet", "0.33) and ")):
        g = D[a]["meas"]
        g = g[g.explainer == "ig"].ig_gap.abs()
        val = f"{f(g.median())} logit for the ResNet (95th percentile\n{f(g.quantile(.95))}" if a == "resnet" else f"{f(g.median())} for the DenseNet ({f(g.quantile(.95))})"
        check("Performance", f"IG completeness gap {a}", anchor, val, f"saliency_{a}/measurands.csv")
    for a in ARCH:
        lab = "ResNet" if a == "resnet" else "DenseNet"
        s = D[a]["pr"]
        lg = s[s["size"] == ">30 mm"].sensitivity_logit_gt0.median()
        rows.append(dict(section="Limitations", claim=f"negative logit >30 mm {lab}", paper="", recomputed=pct(1 - lg), source="rider_predictions.csv", status="INFO"))
    check("Limitations", "negative logit >30 mm range", "RIDER tumours exceed the LIDC size range (",
          f"{pct(1 - R['pr'][R['pr']['size'] == '>30 mm'].sensitivity_logit_gt0.median())}--{pct(1 - N['pr'][N['pr']['size'] == '>30 mm'].sensitivity_logit_gt0.median())}\\% of",
          "rider_predictions.csv")

    # ---- variance components (table and text) ----------------------------------------------------
    comps = [("patient", "patient"), ("patient x scan", "pat.$\\times$scan"), ("patient x thickness", "pat.$\\times$thickness"),
             ("patient x kernel", "pat.$\\times$kernel"), ("patient x seed", "pat.$\\times$run"), ("residual", "residual")]
    for comp, lab in comps:
        cells = []
        for a, e, ms in (("resnet", "ig", "laf"), ("resnet", "gradcam", "laf"), ("resnet", "ig", "logit"), ("densenet", "ig", "laf")):
            v = D[a]["vc"].loc[(e, ms, comp)]
            fmt = (lambda x: f(x, 2)) if ms == "logit" else (lambda x: f0(x, 3))
            p = v.pct_of_random
            ptxt = f(p, 1) if p < 2.95 and f(p, 1) not in ("0.0",) else f(p, 0)
            cells.append(f"{fmt(v.sd)} [{fmt(v.sd_ci_lo)}, {fmt(v.sd_ci_hi)}] {ptxt}\\%")
        line = f"{lab}" + "".join(f" & {c}" for c in cells)
        check("Table VC", f"row {comp}", "", re.sub(r"\s+", " ", line), "variance_components.csv (both architectures)")
    vr, vn = R["vc"], N["vc"]
    rs, rr = vr.loc[("ig", "laf", "patient x scan")], vr.loc[("ig", "laf", "patient x seed")]
    ns, nr = vn.loc[("ig", "laf", "patient x scan")], vn.loc[("ig", "laf", "patient x seed")]
    check("VC text", "ResNet IG patient share", "For ResNet IG LAF (Table~\\ref{tab:vc}, Fig.~\\ref{fig:sources}), ", f"{pct(vr.loc[('ig','laf','patient')].pct_of_random/100)}\\%", "variance_components.csv")
    check("VC text", "ResNet run SD and share", "the patient$\\times$run component (SD ", f"{f(rr.sd,3)} [{f(rr.sd_ci_lo,3)}, {f(rr.sd_ci_hi,3)}], {pct(rr.pct_of_random/100)}\\%) had an SD {f(rr.sd/rs.sd,1)}", "variance_components.csv")
    check("VC text", "ResNet scan SD", "times that of the patient$\\times$scan component (", f"{f(rs.sd,3)} [{f(rs.sd_ci_lo,3)}, {f(rs.sd_ci_hi,3)}])", "variance_components.csv")
    check("VC text", "ResNet thickness/kernel SD", "larger than the $\\times$thickness (", f"{f(vr.loc[('ig','laf','patient x thickness')].sd,3)}) and $\\times$kernel ({f(vr.loc[('ig','laf','patient x kernel')].sd,3)})", "variance_components.csv")
    check("VC text", "ResNet run LOO", "leaving out any one run gave ", rng(rr.sd_leave_one_seed_out_min, rr.sd_leave_one_seed_out_max, 3), "variance_components.csv")
    check("VC text", "DenseNet patient share and run", "pattern: patient ", f"{pct(vn.loc[('ig','laf','patient')].pct_of_random/100)}\\%, patient$\\times$run {f(nr.sd,3)} [{f(nr.sd_ci_lo,3)}, {f(nr.sd_ci_hi,3)}] ({pct(nr.pct_of_random/100)}\\%), an SD {f(nr.sd/ns.sd,1)} times", "variance_components.csv")
    check("VC text", "DenseNet scan", "times that of the patient$\\times$scan component\n(", f"{f(ns.sd,3)} [{f(ns.sd_ci_lo,3)}, {f(ns.sd_ci_hi,3)}])", "variance_components.csv")
    check("Abstract", "run/scan SD and variance ratios", "the standard deviation (SD) of the lesion-by-run component was ",
          f"{f(rr.sd/rs.sd,1)} (ResNet) and 4.4 (DenseNet) times that of the lesion-by-scan component (variance ratios about {f((rr.sd/rs.sd)**2,-1).split('.')[0] if False else int(round((rr.sd/rs.sd)**2/10))*10} and {int(round((nr.sd/ns.sd)**2/10))*10})"
          if f(nr.sd/ns.sd,1) == "4.4" else "DENSENET RATIO CHANGED", "variance_components.csv (SD ratio squared)")
    check("VC text", "ResNet variance ratio", "(variance ratio about ", f"{int(round((rr.sd/rs.sd)**2/10))*10}) and was similarly larger", "variance_components.csv")
    lv = vr.loc[("ig", "logit", "patient x seed")]
    check("VC text", "logit run component", "the run component (ResNet SD ", f"{f(lv.sd)} [{f(lv.sd_ci_lo)}, {f(lv.sd_ci_hi)}], {pct(lv.pct_of_random/100)}\\%; leave-one-run-out {rng(lv.sd_leave_one_seed_out_min, lv.sd_leave_one_seed_out_max)}) had an SD", "variance_components.csv")
    check("VC text", "Grad-CAM / Occlusion run share", "For Grad-CAM (", f"{pct(vr.loc[('gradcam','laf','patient x seed')].pct_of_random/100)}\\%) and Occlusion ({pct(vr.loc[('occlusion','laf','patient x seed')].pct_of_random/100)}\\%)", "variance_components.csv")

    # ---- pairwise wSD ----------------------------------------------------------------------------
    def pw(a, e="ig"):
        rep, con, sd = D[a]["rep"], D[a]["con"], D[a]["sd"]
        rep = rep[(rep.population == "primary") & rep.setting.isin(P) & (rep.measure == "laf") & (rep.explainer == e)]
        con = con[~con.contrast.str.contains("2.5S") & (con.measure == "laf") & (con.explainer == e)]
        sd = sd[sd.setting.isin(P) & (sd.measure == "laf") & (sd.explainer == e)]
        return rep.wSD.mean(), con[con.kind == "thickness"].wSD.mean(), con[con.kind == "kernel"].wSD.mean(), sd.wSD.mean()
    a_, b_, c_, d_ = pw("resnet")
    check("Pairwise", "ResNet pairwise wSD", "the within-lesion SD of ResNet IG LAF was ", f"{f(a_,3)} between scans, {f(b_,3)} between\nthicknesses, {f(c_,3)} between kernels and {f(d_,3)} between runs ({f(d_/a_,1)} times", "repeatability/conditions/seeds.csv")
    a2, b2, c2, d2 = pw("densenet")
    check("Pairwise", "DenseNet pairwise wSD", "DenseNet ", f"{f(a2,3)}, {f(b2,3)}, {f(c2,3)}\nand {f(d2,3)}, {f(d2/a2,1)} times", "repeatability/conditions/seeds.csv")
    for a, anchor in (("resnet", "Between-run agreement on the same cell was ICC "), ("densenet", "(DenseNet ")):
        s = D[a]["sc"]
        s = s[s.setting.isin(P)].groupby(["explainer", "measure"])[["icc", "icc_ci_lo", "icc_ci_hi"]].median().loc[("ig", "laf")]
        check("Pairwise", f"between-run ICC {a}", anchor, f"{f(s.icc)} [{f(s.icc_ci_lo)}, {f(s.icc_ci_hi)}]", "seeds_cluster_bootstrap.csv")

    # ---- primary endpoint ------------------------------------------------------------------------
    for a in ARCH:
        rep = D[a]["rep"]
        r = rep[(rep.population == "primary") & (rep.setting == "1.25L") & (rep.measure == "laf") & (rep.explainer == "ig")]
        lo = r.loc[r.icc.idxmin()]
        if a == "resnet":
            check("Primary", "ResNet ICC range", "had scan--rescan ICC ", f"{rng(r.icc.min(), r.icc.max(), 3)} across the ten runs", "repeatability.csv")
            check("Primary", "ResNet lowest ICC CI", "(lowest ", f"{f(lo.icc,3)} [{f(lo.icc_ci_lo,3)}, {f(lo.icc_ci_hi,3)}]); RC {pct(r.RC_over_median.min())}--{pct(r.RC_over_median.max())}\\%", "repeatability.csv")
            check("Abstract", "ICC ranges", "ICC ", f"{rng(r.icc.min(), r.icc.max(), 3)} over ten ResNet runs", "repeatability.csv")
        else:
            check("Primary", "DenseNet ICC range", "DenseNet IG LAF: ICC ", f"{rng(r.icc.min(), r.icc.max(), 3)} (lowest {f(lo.icc,3)}\n[{f(lo.icc_ci_lo,3)}, {f(lo.icc_ci_hi,3)}]), RC {pct(r.RC_over_median.min())}--{pct(r.RC_over_median.max())}\\%", "repeatability.csv")
            check("Abstract", "DenseNet ICC range", "and ", f"{rng(r.icc.min(), r.icc.max(), 3)} over ten DenseNet runs", "repeatability.csv")
    for a in ARCH:
        m = re.findall(r"leave-one-patient-out ICC ([0-9.]+)-", D[a]["summ"])
        if a == "resnet":
            check("Primary", "LOO minimum", "no single patient moved the ICC below ", min(m, key=float), "summary.txt")
        g = re.findall(r"AIAG %GRR (\d+)%", D[a]["summ"])
        if a == "resnet":
            check("Primary", "AIAG %GRR range", "(\\%GRR ", f"{min(map(int, g))}--{max(map(int, g))}\\%)", "summary.txt")
    ac = pd.read_csv(J / "compare_arch" / "arch_ci.csv").set_index("metric")
    w = ac.loc["wSD_scan"]
    check("Primary", "arch scan wSD diff", "slightly larger (", f"{f(w.densenet,3)} vs.\\ {f(w.resnet,3)}; difference {f(w.difference,3)} {ci(w.diff_ci_lo, w.diff_ci_hi)})", "compare_arch/arch_ci.csv")
    w = ac.loc["wSD_kernel"]
    check("Primary", "arch kernel diff", "its kernel wSD\nalso (", f"{f(w.difference,3)} {ci(w.diff_ci_lo, w.diff_ci_hi)})", "arch_ci.csv")
    w = ac.loc["wSD_run"]
    check("Primary", "arch run diff", "between-run wSD slightly smaller (", f"$-{f(-w.difference,3)}$ {ci(w.diff_ci_lo, w.diff_ci_hi)}", "arch_ci.csv")
    w = ac.loc["icc_primary"]
    check("Primary", "arch ICC diff", "did not differ (", f"$-{f(-w.difference,3)}$ {ci(w.diff_ci_lo, w.diff_ci_hi)})", "arch_ci.csv")

    # ---- subgroups -------------------------------------------------------------------------------
    s = R["strd"].set_index(["explainer", "stratum", "metric"])
    z, p_ = s.loc[("ig", "size", "wSD")], s.loc[("ig", "pred", "wSD")]
    check("Subgroups", "ResNet size", "(wSD\n", f"{f(z.value_a,3)} vs.\\ {f(z.value_b,3)}; difference {f(z.diff_b_minus_a,3)} {ci(z.diff_ci_lo, z.diff_ci_hi)}; {z.n_patients_a} and {z.n_patients_b} patients)", "strata_differences.csv")
    check("Subgroups", "ResNet logit sign", "negative (", f"{f(p_.value_b,3)} vs.\\ {f(p_.value_a,3)}; {f(p_.diff_b_minus_a,3)} {ci(p_.diff_ci_lo, p_.diff_ci_hi)})", "strata_differences.csv")
    s = N["strd"].set_index(["explainer", "stratum", "metric"])
    z, p_ = s.loc[("ig", "size", "wSD")], s.loc[("ig", "pred", "wSD")]
    check("Subgroups", "DenseNet size", "\\emph{smaller} wSD\n(", f"{f(z.value_b,3)} vs.\\ {f(z.value_a,3)}; $-{f(-z.diff_b_minus_a,3)}$ {ci(z.diff_ci_lo, z.diff_ci_hi)})", "strata_differences.csv")
    check("Subgroups", "DenseNet logit sign", "made no clear difference (", f"{f(p_.diff_b_minus_a,3)} {ci(p_.diff_ci_lo, p_.diff_ci_hi)})", "strata_differences.csv")

    # ---- specificity / explainers -----------------------------------------------------------------
    def enr(a, e):
        m = re.search(rf"{e}: median LAF ([0-9.]+) vs chance ([0-9.]+) \(enrichment ([0-9.]+)x\)", D[a]["summ"])
        return float(m.group(1)), float(m.group(2)), float(m.group(3))
    lafR, chR, enR = enr("resnet", "ig")
    check("Specificity", "IG enrichment and LAF", "ResNet IG placed ", f"{f(enR,0)}$\\times$ its chance share on the lesion (median LAF {f(lafR)})", "summary.txt")
    c = R["ctl"]
    cm = c.groupby("explainer")[["laf_shift", "laf_mismatch"]].median()  # all maps, as in 10_robustness / the ISBI sheet
    c = c[~c.PatientID.isin(PRIMARY_EXCLUDE) & c.setting.isin(P)]
    check("Specificity", "IG shifted SEG", "With the SEG\nshifted 25~mm, LAF fell to ", f(cm.loc["ig", "laf_shift"], 3), "controls.csv")
    check("Specificity", "IG foreign SEG", "a centred, size-matched foreign SEG captured ", f(cm.loc["ig", "laf_mismatch"]), "controls.csv")
    for a in ARCH:
        g = D[a]["ctl"]
        g = g[~g.PatientID.isin(PRIMARY_EXCLUDE) & g.setting.isin(P) & (g.explainer == "ig")].dropna(subset=["laf_mismatch_outside"])
        r_ = g.laf_mismatch_outside / g.laf_check
        dens = (g.laf_mismatch_outside / (1 - g.mismatch_overlap) / g.laf_check).median()
        if a == "resnet":
            check("Specificity", "overlap", "(median ", f"{pct(g.mismatch_overlap.median())}\\% of the own lesion)", "controls.csv")
            check("Specificity", "outside share IQR density", "received ", f"{pct(r_.median())}\\% of the own-SEG share\n(IQR {pct(r_.quantile(.25))}--{pct(r_.quantile(.75))}\\%; attribution density {f(dens)} of the lesion's)", "controls.csv")
            check("Specificity", "inside share", "whereas the overlapping part received ", f"{pct((g.laf_mismatch_inside_own / g.laf_check).median())}\\%", "controls.csv")
        else:
            check("Specificity", "DenseNet outside share", "(outside share ", f"{pct(r_.median())}\\%, density {f(dens)})", "controls.csv")
    lg, chg, eng = enr("resnet", "gradcam")
    check("Specificity", "Grad-CAM enrichment", "Grad-CAM placed ", f"{f(eng,1)}$\\times$ chance on the lesion ({f(lg,3)}\nvs.\\ {f(chg,3)})", "summary.txt")
    check("Specificity", "Grad-CAM controls", "the same with a\nshifted (", f"{f(cm.loc['gradcam','laf_shift'],3)}) or foreign SEG ({f(cm.loc['gradcam','laf_mismatch'],3)})", "controls.csv")
    g = c[(c.explainer == "gradcam")].dropna(subset=["laf_mismatch_outside"])
    check("Specificity", "Grad-CAM density", "its outside density was ", f((g.laf_mismatch_outside / (1 - g.mismatch_overlap) / g.laf_check).median()), "controls.csv")
    lo_, _, _ = enr("resnet", "occlusion")
    check("Specificity", "Occlusion LAF and shift", "Occlusion behaved similarly (LAF ", f"{f(lo_,3)}, shifted {f(cm.loc['occlusion','laf_shift'],3)})", "summary.txt, controls.csv")
    laf_b, _, _ = enr("resnet", "ig_blur")
    laf_m, _, _ = enr("resnet", "ig_mean")
    check("Baselines", "LAF by baseline", "(median LAF ", f"{f(lafR)}\nwith the air, {f(laf_b)} with the blurred and {f(laf_m,3)} with the mean baseline; shifted-SEG controls "
          f"{f(cm.loc[['ig','ig_blur','ig_mean'],'laf_shift'].min(),3)}--{f(cm.loc[['ig','ig_blur','ig_mean'],'laf_shift'].max(),3)})", "summary.txt, controls.csv")
    rp = R["rep"]
    rp = rp[(rp.population == "primary") & rp.setting.isin(P) & (rp.measure == "laf")]
    icc = rp.groupby("explainer").icc.median()
    wsd = rp.groupby("explainer").wSD.mean()
    check("Baselines", "ICC by baseline", "(median scan--rescan ICC ", f"{f(icc['ig'],3)}, {f(icc['ig_blur'],3)} and {f(icc['ig_mean'],3)}; scan--rescan wSD {f(wsd['ig'],3)}, {f(wsd['ig_blur'],3)}, {f(wsd['ig_mean'],3)})", "repeatability.csv")
    check("Explainers", "ICC range all explainers", "All explainers had high median scan--rescan ICC (", f"{f(icc.min(),3)}--{f(icc.max(),3)}", "repeatability.csv")
    rcm = rp.groupby("explainer").apply(lambda q: (q.RC_over_median > 1).mean(), include_groups=False)
    check("Explainers", "RC>median Grad-CAM", "The RC exceeded the median LAF in ", f"{pct(rcm['gradcam'])}\\% of\nrun--reconstruction combinations for Grad-CAM", "repeatability.csv")
    sc_ = R["sc"]
    sc_ = sc_[sc_.setting.isin(P) & (sc_.measure == "laf")].groupby("explainer").icc.median()
    check("Explainers", "between-run ICC Grad-CAM/Occlusion/IG", "Grad-CAM (between-run ICC ", f"{f(sc_['gradcam'])}) and Occlusion ({f(sc_['occlusion'])}) were far more run-specific than IG ({f(sc_['ig'])})", "seeds_cluster_bootstrap.csv")
    fsr = R["fs"].groupby("explainer")[["score_median", "share_positive"]].median()
    sdm = R["sd"]
    sdm = sdm[sdm.setting.isin(P) & (sdm.measure == "laf")].groupby("explainer").wSD.mean()
    names = {"ig": "IG (air)", "ig_blur": "IG (blur)", "ig_mean": "IG (mean)", "gradcam": "Grad-CAM layer4", "occlusion": "Occlusion"}
    for e, nm in names.items():
        lafe, _, ene = enr("resnet", e)
        dec = 3 if lafe < 0.1 else 2
        star = "$^\\ast$" if e == "ig_blur" else ""
        row = (f"{nm} & {f(lafe, dec)} & {f(ene, 1)} & {f(icc[e], 3)} & {f(wsd[e], 3)} & {f(sdm[e], 3)} & {pct(rcm[e])}\\% "
               f"& {f(fsr.loc[e, 'score_median'])} ({pct(fsr.loc[e, 'share_positive'])}\\%){star} \\\\")
        tex_row = re.sub(r" +", " ", row).replace(" \\\\", " &")  # a mean-fill column follows (checked below)
        present = tex_row in re.sub(r" +", " ", flat)
        rows.append(dict(section="Table explainers", claim=f"row {e}", paper="(as recomputed)" if present else "see manuscript",
                         recomputed=tex_row, source="summary.txt, repeatability, seeds, faithfulness_summary", status="MATCH" if present else "MISMATCH"))

    # ---- faithfulness ----------------------------------------------------------------------------
    fr = R["fs"][R["fs"].explainer == "ig"]
    check("Faithfulness", "IG deletion score", "For IG with the air baseline the median deletion\nscore was ",
          f"{f(fr.score_median.median())} logit (positive in {pct(fr.share_positive.median())}\\% of maps; run medians {rng(fr.score_median.min(), fr.score_median.max())}; DenseNet "
          f"{f(N['fs'][N['fs'].explainer=='ig'].score_median.median())}, {pct(N['fs'][N['fs'].explainer=='ig'].share_positive.median())}\\%)", "faithfulness_summary.csv")
    check("Faithfulness", "Occlusion has a run with negative median", "Occlusion (", f"{f(fsr.loc['occlusion','score_median'])}; {['no','one','two','three','four','five'][int((R['fs'][R['fs'].explainer=='occlusion'].score_median < 0).sum())]} of ten run medians below zero)", "faithfulness_summary.csv")
    allicc = pd.concat([D[a]["fs"].groupby("explainer").score_scan_rescan_icc_median.median() for a in ARCH])
    check("Faithfulness", "score ICC range", "(median ICC ", f"{rng(allicc.min(), allicc.max())} for all explainers and both architectures)", "faithfulness_summary.csv")
    sr = R["san"].set_index(["explainer", "randomised"])
    sn = N["san"].set_index(["explainer", "randomised"])
    check("Randomisation", "IG Spearman cascade", "(median Spearman ", f"{f(sr.loc[('ig','head'),'spearman_median'])}; head+layer4 {f(sr.loc[('ig','head+layer4'),'spearman_median'])}; whole network {f(sr.loc[('ig','all'),'spearman_median'])}; DenseNet "
          f"{f(sn.loc[('ig','head'),'spearman_median'])}, {f(sn.loc[('ig','head+layer4'),'spearman_median'])},\n{f(sn.loc[('ig','all'),'spearman_median'])})", "sanity_summary.csv")
    check("Randomisation", "IG LAF cascade", "LAF\nfell from ", f"{f(sr.loc[('ig','head'),'laf_trained'])} to {f(sr.loc[('ig','head'),'laf_randomised'])} (head) and {f(sr.loc[('ig','head+layer4'),'laf_randomised'])} (head+layer4), and to {f(sr.loc[('ig','all'),'laf_randomised'])}", "sanity_summary.csv")
    check("Randomisation", "DenseNet LAF cascade", "whole network was randomised (DenseNet\n", f"{f(sn.loc[('ig','head'),'laf_trained'])}, {f(sn.loc[('ig','head'),'laf_randomised'])}, {f(sn.loc[('ig','head+layer4'),'laf_randomised'])}, {f(sn.loc[('ig','all'),'laf_randomised'])})", "sanity_summary.csv")
    surv = np.mean([sr.loc[("ig", "head"), "laf_randomised"] / sr.loc[("ig", "head"), "laf_trained"], sr.loc[("ig", "head+layer4"), "laf_randomised"] / sr.loc[("ig", "head"), "laf_trained"]])
    check("Randomisation", "share surviving (about)", "About ", f"{int(round(surv * 10)) * 10}\\% of the lesion-level summary therefore survives", "sanity_summary.csv")
    gc = pd.concat([sr.loc["gradcam"].spearman_median, sn.loc["gradcam"].spearman_median])
    check("Randomisation", "Grad-CAM max Spearman", "Grad-CAM also decorrelated (median\nSpearman $\\leq$", f(gc.max()), "sanity_summary.csv")
    nrand = R["san"].n.iloc[0]
    check("Methods", "randomisation map count", "(", f"{nrand} maps\nper architecture and explainer)", "sanity_summary.csv")

    # ---- reproducibility, fixed reference mask -----------------------------------------------------
    for a, val in (("resnet", None), ("densenet", None)):
        con = D[a]["con"]
        con = con[~con.contrast.str.contains("2.5S") & (con.measure == "laf")]
        mn = con[con.explainer == "ig"].icc.min()
        if a == "resnet":
            check("Reproducibility", "IG min ICC ResNet", "(ICC $\\geq$", f"{f(mn)} for every ResNet run and contrast; DenseNet $\\geq${f(D['densenet']['con'][~D['densenet']['con'].contrast.str.contains('2.5S') & (D['densenet']['con'].measure=='laf') & (D['densenet']['con'].explainer=='ig')].icc.min())})", "conditions.csv")
            k = con[(con.contrast == "1.25L vs 1.25S") & (con.explainer == "gradcam")]
            check("Reproducibility", "Grad-CAM kernel ICC and exceed", "Grad-CAM did not reproduce across kernels (ICC ", f"{rng(k.icc.min(), k.icc.max())} at 1.25~mm over runs; {pct(k.exceed_rc.min())}--{pct(k.exceed_rc.max())}\\%", "conditions.csv")
    rf = R["rfm"].groupby("explainer").median(numeric_only=True)
    nf = N["rfm"].groupby("explainer").median(numeric_only=True)
    check("Reproducibility", "mask Dice", "(mask Dice with the\nseries' own SEG ", f"{f(rf.loc['ig','mask_dice_own_vs_ref'])})", "recon_fixed_mask_summary.csv")
    check("Reproducibility", "fixed reference mask wSD", "the between-reconstruction wSD of IG LAF fell from ", f"{f(rf.loc['ig','wsd_vs_1.25L_own'],3)} to {f(rf.loc['ig','wsd_vs_1.25L_refmask'],3)} (DenseNet {f(nf.loc['ig','wsd_vs_1.25L_own'],3)} to {f(nf.loc['ig','wsd_vs_1.25L_refmask'],3)})", "recon_fixed_mask_summary.csv")

    # ---- map level and registration ---------------------------------------------------------------
    mR = R["mc"][~R["mc"].contrast.str.contains("2.5S")].groupby(["explainer", "contrast"])[["spearman", "spearman_8"]].median().loc["ig"]
    mN = N["mc"][~N["mc"].contrast.str.contains("2.5S")].groupby(["explainer", "contrast"])[["spearman", "spearman_8"]].median().loc["ig"]
    check("Map level", "between-recon 1 mm", "IG maps correlated moderately at 1~mm (Spearman ", f"{rng(mR.spearman.min(), mR.spearman.max())} for the\nResNet, {rng(mN.spearman.min(), mN.spearman.max())} for the DenseNet)", "map_conditions.csv")
    check("Map level", "between-recon 8^3", "pooled to 16~mm cells they agreed closely\n(", f"{rng(mR.spearman_8.min(), mR.spearman_8.max())} and {rng(mN.spearman_8.min(), mN.spearman_8.max())})", "map_conditions.csv")
    gR = R["reg"].groupby("explainer").median(numeric_only=True)
    gN = N["reg"].groupby("explainer").median(numeric_only=True)
    check("Map level", "scan-rescan unregistered", "with centroid alignment only, IG maps correlated at ", f"{f(gR.loc['ig','spearman_unregistered'])} (ResNet) and {f(gN.loc['ig','spearman_unregistered'])}\n(DenseNet) at 1~mm; Grad-CAM at {f(gR.loc['gradcam','spearman_unregistered'])} and Occlusion at {f(gR.loc['occlusion','spearman_unregistered'])}", "registration_summary.csv")
    check("Abstract", "scan-rescan 1 mm range", "agreed only moderately between scans (Spearman ", f"{f(gR.loc['ig','spearman_unregistered'])}--{f(gN.loc['ig','spearman_unregistered'])}; {f(gR.loc['ig','spearman_8_unregistered'])} when pooled", "registration_summary.csv")
    check("Registration", "whole-patch Dice", "(SEG Dice ", f"{f(gR.loc['ig','seg_dice_before'])} before, {f(gR.loc['ig','seg_dice_global'])} after)", "registration_summary.csv")
    sp = R["sp"]
    sp = sp[(sp.explainer == "ig") & (sp.seed == 0) & ~sp.PatientID.isin(PRIMARY_EXCLUDE) & sp.setting.isin(P)]
    check("Registration", "local shift and zero share", "registration (median shift ", f"{f(gR.loc['ig','shift_mm'],0)}~mm; zero in {pct(gR.loc['ig','share_zero_shift'])}\\% of pairs) raised the CT NCC in the tumour neighbourhood from {f(gR.loc['ig','ncc_lesion_before'])} to\n{f(gR.loc['ig','ncc_lesion_after'])} and left the SEG Dice unchanged ({f(gR.loc['ig','seg_dice_after'])})", "registration_summary.csv")
    check("Registration", "registered 1 mm agreement", "lowered the 1~mm IG agreement (Spearman ", f"{f(gR.loc['ig','spearman_unregistered'])} to {f(gR.loc['ig','spearman'])} for the\nResNet, {f(gN.loc['ig','spearman_unregistered'])} to {f(gN.loc['ig','spearman'])} for the DenseNet; top-10\\% Dice {f(gR.loc['ig','dice_top10_unregistered'])} to {f(gR.loc['ig','dice_top10'])})", "registration_summary.csv")
    check("Registration", "8^3 unchanged", "agreement pooled to $8^3$ cells was unchanged\n(", f"{f(gR.loc['ig','spearman_8_unregistered'])} before and after)" if f(gR.loc['ig','spearman_8_unregistered']) == f(gR.loc['ig','spearman_8']) else "DIFFERS", "registration_summary.csv")
    rr_ = R["reg"][R["reg"].explainer == "ig"]
    nn_ = N["reg"][N["reg"].explainer == "ig"]
    check("Registration", "fixed-mask wSD and ICC", "the scan--rescan wSD of IG LAF fell from ", f"{f(rr_.laf_wsd_own.median(),3)} to {f(rr_.laf_wsd_fixed.median(),3)} (ICC {f(rr_.laf_icc_own.median(),3)} to\n{f(rr_.laf_icc_fixed.median(),3)}; DenseNet {f(nn_.laf_wsd_own.median(),3)} to {f(nn_.laf_wsd_fixed.median(),3)}) in every run"
          if (rr_.laf_wsd_fixed < rr_.laf_wsd_own).all() and (nn_.laf_wsd_fixed < nn_.laf_wsd_own).all() else "NOT EVERY RUN", "registration_summary.csv")
    share = 1 - (rr_.laf_wsd_fixed.median() / rr_.laf_wsd_own.median()) ** 2
    check("Registration", "contour share", "about a third (", f"{pct(share)}\\%)", "registration_summary.csv")

    # ---- model vs explainer ----------------------------------------------------------------------
    st = R["stab"][R["stab"].explainer == "ig"]
    stn = N["stab"][N["stab"].explainer == "ig"]
    gcn = N["stab"][N["stab"].explainer == "gradcam"]
    check("Model vs explainer", "IG Spearman ranges and significant runs", "(Spearman ", f"$-{f(-st.spearman_absdlogit_absdlaf.min())}$ to {f(st.spearman_absdlogit_absdlaf.max())} over\nthe ResNet runs, {['none','one','two','three','four','five','six','seven','eight','nine','ten'][int((st.p<0.05).sum())]} with $p<0.05$; DenseNet $-{f(-stn.spearman_absdlogit_absdlaf.min())}$ to {f(stn.spearman_absdlogit_absdlaf.max())}, "
          f"{['no','one','two','three','four','five','six','seven','eight','nine','ten'][int((stn.p<0.05).sum())]} of ten runs with $p<0.05$)", "stability.csv")
    check("Model vs explainer", "Grad-CAM DenseNet", "(DenseNet up to ", f"{f(gcn.spearman_absdlogit_absdlaf.max())}, {['no','one','two','three','four','five','six','seven','eight','nine','ten'][int((gcn.p<0.05).sum())]} of ten runs with $p<0.05$)", "stability.csv")

    # ---- centre-prior ablation (64 mm) -----------------------------------------------------------
    ab = pd.read_csv(AB / "ablation_ci.csv").set_index("metric")
    nm = [c for c in ab.columns if c not in ("n_patients", "difference", "diff_ci_lo", "diff_ci_hi") and not c.endswith(("_lo", "_hi"))]
    for metric, anchor in (("wSD_scan", "difference without $-$ with [95\\% CI]):\nscan--rescan wSD "), ("wSD_kernel", "0.010], kernel wSD "),
                           ("wSD_run", "0.002], between-run wSD "), ("icc_primary", "0.021], primary ICC ")):
        w = ab.loc[metric]
        d = w.difference
        val = (f"$-{f(-d,3)}$" if d < 0 else f(d, 3)) + " " + ci(w.diff_ci_lo, w.diff_ci_hi)
        check("Ablation", metric, anchor, val, "ablation/ablation_ci.csv")
    w = ab.loc["logit_wSD_scan"]
    check("Ablation", "logit scan SD", "(scan--rescan SD ", f"{f(w[nm[0]])} vs.\\ {f(w[nm[1]])}; {f(w.difference)} [{f(w.diff_ci_lo)}, {f(w.diff_ci_hi)}])", "ablation_ci.csv")
    w = ab.loc["outside_share"]
    check("Ablation", "outside share", "outside the lesion (", f"{pct(w[nm[0]])}\\% vs.\\ {pct(w[nm[1]])}\\%; {f(w.difference,3)} [{f(w.diff_ci_lo,3)}, {f(w.diff_ci_hi,3)}])", "ablation_ci.csv")

    # ---- review round 2: decisions by logit sign, secondary table, 64 mm variance ratios --------------
    rng_ = np.random.default_rng(0)
    for a, lab in (("resnet", "ResNet"), ("densenet", "DenseNet")):
        fa = pd.read_csv(J / f"analysis_{a}" / "faithfulness" / "faithfulness.csv")
        fa = fa[~fa.PatientID.isin(PRIMARY_EXCLUDE)]
        for e in ("ig", "gradcam"):
            g = fa[fa.explainer == e]
            pos, neg = g[g.logit > 0].deletion_score.median(), g[g.logit <= 0].deletion_score.median()
            if a == "resnet" and e == "ig":
                pats = g.PatientID.unique()
                byp = {q: g[g.PatientID == q] for q in pats}
                diffs = []
                for _ in range(1000):
                    gg = pd.concat([byp[q] for q in rng_.choice(pats, len(pats))])
                    diffs.append(gg[gg.logit <= 0].deletion_score.median() - gg[gg.logit > 0].deletion_score.median())
                check("Decisions", "IG deletion by logit sign", "for IG, a median of ",
                      f"{f(neg)}\nlogit when the logit was negative vs.\\ {f(pos)} when it was positive (difference $-{f(-(neg-pos))}$ [$-{f(-np.percentile(diffs, 2.5))}$, $-{f(-np.percentile(diffs, 97.5))}$],", "faithfulness.csv (1000 patient resamples)")
            elif a == "densenet" and e == "ig":
                check("Decisions", "DenseNet IG by sign", "patient-cluster bootstrap; DenseNet ", f"{f(neg)} vs.\\ {f(pos)})", "faithfulness.csv")
            elif a == "resnet" and e == "gradcam":
                check("Decisions", "Grad-CAM by sign", "similarly for Grad-CAM (", f"{f(neg)} vs.\\ {f(pos)})", "faithfulness.csv")
    for a in ARCH:
        rep_ = D[a]["rep"]
        r = rep_[(rep_.setting == "1.25L") & (rep_.measure == "laf") & (rep_.explainer == "ig")].pivot_table(index="seed", columns="population", values="icc")
        D[a]["dicc"] = (r["all"] - r["primary"]).abs().max()
        s25 = rep_[(rep_.setting == "2.5S") & (rep_.population == "primary") & (rep_.measure == "laf") & (rep_.explainer == "ig")]
        D[a]["s25"] = (int(s25.n.median()), s25.icc.min(), s25.icc.max(), s25.wSD.mean())
        c = D[a]["ctl"]
        c = c[~c.PatientID.isin(PRIMARY_EXCLUDE)]
        g = c[(c.explainer == "ig") & c.setting.isin(P)]
        iccs = []
        for _, gg in g.groupby(["seed", "setting"]):
            w = gg.pivot_table(index="PatientID", columns="acq", values="laf_abs").dropna()
            if len(w) >= 5:
                iccs.append(S7PC.icc_1_1(w[["scan1", "scan2"]].to_numpy(float))[0])
        D[a]["abs"] = (g.laf_abs.median(), np.median(iccs))
        m = re.search(r"gradcam: median LAF.*?empty maps ([0-9.]+)% \((\d+)/(\d+)\)", D[a]["summ"])
        D[a]["empty"] = (int(m.group(2)), float(m.group(1)), int(m.group(3)))
    check("Secondary", "excluded patient text", "changed the primary-endpoint ICC by at most ", f"{f(D['resnet']['dicc'],3)} (ResNet) and\n{f(D['densenet']['dicc'],3)} (DenseNet)", "repeatability.csv all vs primary")
    t = lambda a: D[a]["s25"]  # noqa: E731
    check("Secondary", "table: excluded patient", "Excluded patient added: max change of primary ICC & ", f"{f(D['resnet']['dicc'],3)} & {f(D['densenet']['dicc'],3)}", "repeatability.csv")
    check("Secondary", "table: 2.5S n and ICC", "IG ICC range over runs & ", f"{t('resnet')[0]}; {rng(t('resnet')[1], t('resnet')[2])} & {t('densenet')[0]}; {rng(t('densenet')[1], t('densenet')[2])}", "repeatability.csv 2.5S")
    check("Secondary", "table: 2.5S wSD", "IG scan--rescan wSD & ", f"{f(t('resnet')[3],3)} & {f(t('densenet')[3],3)}", "repeatability.csv 2.5S")
    check("Secondary", "table: |a| LAF", "scan--rescan ICC (median) & ", f"{f(D['resnet']['abs'][0])}; {f(D['resnet']['abs'][1])} & {f(D['densenet']['abs'][0])}; {f(D['densenet']['abs'][1])}", "controls.csv laf_abs")
    check("Secondary", "table: empty Grad-CAM", "Maps without positive attribution: Grad-CAM & ", f"{D['resnet']['empty'][0]} ({f(D['resnet']['empty'][1],1)}\\%) & {D['densenet']['empty'][0]} ({f(D['densenet']['empty'][1],1)}\\%)", "summary.txt")
    check("Secondary", "table: maps per explainer", "(primary population unless stated; ", f"{D['resnet']['empty'][2]:,}".replace(",", "{,}") + " maps per explainer", "summary.txt")
    cmp_ = pd.read_csv(AB / "ablation_comparison.csv", index_col=0)
    v1 = (cmp_.loc["IG VC SD x run", "prior"] / cmp_.loc["IG VC SD x scan", "prior"]) ** 2
    v2 = (cmp_.loc["IG VC SD x run", "no-prior"] / cmp_.loc["IG VC SD x scan", "no-prior"]) ** 2
    check("Ablation", "64 mm variance ratios", "variance ratios about ", f"{f(v1,1)} and {f(v2,1)}), against", "ablation_comparison.csv")

    # ---- review round 2: centring, resolution, neutral deletion (run_review2.ps1) ------------------
    cen = pd.read_csv(J / "analysis_resnet" / "sensitivity" / "centring_summary.csv")
    ci_ = cen[cen.explainer == "ig"].set_index("jitter_mm").sort_index()
    cg = cen[cen.explainer == "gradcam"].set_index("jitter_mm").sort_index()
    three = lambda col, d=2: ", ".join(f(v, d) for v in ci_[col].iloc[:2]) + f" and {f(ci_[col].iloc[2], d)}"  # noqa: E731
    check("Centring", "LAF medians", "the median IG LAF changed little (", f"{three('laf_median')})", "centring_summary.csv")
    check("Centring", "ICC", "the scan--rescan ICC fell from ", f"{f(ci_.icc_scan_median.iloc[0],3)} to {f(ci_.icc_scan_median.iloc[1],3)} and {f(ci_.icc_scan_median.iloc[2],3)} (lowest {three('icc_scan_min',3)})", "centring_summary.csv")
    check("Centring", "scan wSD", "scan--rescan wSD rose from ", f"{f(ci_.wsd_scan.iloc[0],3)} to {f(ci_.wsd_scan.iloc[1],3)} and {f(ci_.wsd_scan.iloc[2],3)}, while the between-run wSD was unchanged ({three('wsd_run',3)})", "centring_summary.csv")
    check("Centring", "residual SD", "the added error appeared in the residual (SD ", f"{three('sd_residual',3)})", "centring_summary.csv")
    check("Centring", "run SD and ratios", "the patient$\\times$run SD stayed at ", f"{f(ci_.sd_patient_x_seed.min(),3)}--{f(ci_.sd_patient_x_seed.max(),3)}, {three('sd_ratio_run_scan',1)} times", "centring_summary.csv")
    check("Centring", "Grad-CAM ICC", "Grad-CAM was more sensitive (ICC ", f"{', '.join(f(v,3) for v in cg.icc_scan_median.iloc[:2])} and {f(cg.icc_scan_median.iloc[2],3)})", "centring_summary.csv")
    check("Abstract", "centring ICC range", "lowered the scan--rescan ICC of IG LAF to ", f"{f(ci_.icc_scan_median.iloc[1],3)}--{f(ci_.icc_scan_median.iloc[2],3)}", "centring_summary.csv")
    rs_ = pd.read_csv(J / "analysis_resnet" / "sensitivity" / "resolution_summary.csv").set_index("map")
    rd_ = pd.read_csv(J / "analysis_densenet" / "sensitivity" / "resolution_summary.csv").set_index("map")
    check("Resolution", "pooled IG 16 mm", "pooled IG still placed ", f"{f(rs_.loc['ig_pooled_16mm','specificity_ratio'],1)} times\nits shifted-SEG value on the lesion (LAF {f(rs_.loc['ig_pooled_16mm','laf_median'])} vs.\\ {f(rs_.loc['ig_pooled_16mm','laf_shift_median'],3)}; DenseNet {f(rd_.loc['ig_pooled_16mm','specificity_ratio'],1)} times)", "resolution_summary.csv")
    check("Resolution", "Grad-CAM / Occlusion 16 mm", "whereas Grad-CAM layer4 (", f"{f(rs_.loc['gradcam','specificity_ratio'],1)}) and\nOcclusion ({f(rs_.loc['occlusion','specificity_ratio'],1)}) did not", "resolution_summary.csv")
    check("Resolution", "8 mm", "at 8~mm, pooled IG reached ", f"{f(rs_.loc['ig_pooled_8mm','specificity_ratio'],1)} times (LAF {f(rs_.loc['ig_pooled_8mm','laf_median'])}) against {f(rs_.loc['gradcam_l3','specificity_ratio'],1)} for Grad-CAM layer3 (LAF {f(rs_.loc['gradcam_l3','laf_median'],3)};\nDenseNet {f(rd_.loc['ig_pooled_8mm','specificity_ratio'],1)} vs.\\ {f(rd_.loc['gradcam_l3','specificity_ratio'],1)})", "resolution_summary.csv")
    check("Resolution", "pooled ICC", "Pooling raised IG's scan--rescan ICC (", f"{f(rs_.loc['ig_pooled_16mm','icc_scan_median'],3)} at 16~mm)", "resolution_summary.csv")
    check("Resolution", "layer3 zero share and run wSD", "Layer3 Grad-CAM placed no\nattribution on the lesion in ", f"{pct(rs_.loc['gradcam_l3','share_zero_laf'])}\\% of maps and had a between-run wSD of {f(rs_.loc['gradcam_l3','wsd_run'],3)}", "resolution_summary.csv")
    fm = {u: pd.read_csv(J / "analysis_resnet" / f"faithfulness_mean_{u}" / "faithfulness_summary.csv").groupby("explainer")[["score_median", "share_positive", "score_scan_rescan_icc_median"]].median() for u in ("voxel", "cube")}
    sg = lambda v: f"$-{f(-v)}$" if float(f(v)) < 0 else f(v)  # noqa: E731
    v_, c_ = fm["voxel"], fm["cube"]
    check("Neutral deletion", "IG (air)", "IG (air) no longer beat random deletion\n(median ", f"{sg(v_.loc['ig','score_median'])} logit, positive in {pct(v_.loc['ig','share_positive'])}\\% of maps; 8~mm cubes {sg(c_.loc['ig','score_median'])}, {pct(c_.loc['ig','share_positive'])}\\%)", "faithfulness_mean_voxel/cube")
    check("Neutral deletion", "others", "whereas Grad-CAM (", f"{f(v_.loc['gradcam','score_median'])}; cubes {f(c_.loc['gradcam','score_median'])}),\nOcclusion ({f(v_.loc['occlusion','score_median'])}; {f(c_.loc['occlusion','score_median'])}), IG (blur) ({f(v_.loc['ig_blur','score_median'])}; {f(c_.loc['ig_blur','score_median'])}) and IG (mean) ({f(v_.loc['ig_mean','score_median'])}; {f(c_.loc['ig_mean','score_median'])}) remained positive", "faithfulness_mean_voxel/cube")
    allr = pd.concat([v_.score_scan_rescan_icc_median, c_.score_scan_rescan_icc_median])
    check("Neutral deletion", "score ICC", "The score stayed repeatable between scans (median ICC\n", f"{rng(allr.min(), allr.max())})", "faithfulness_mean_voxel/cube")
    star = {"ig_mean": "$^\\ast$"}
    for e, nm in names.items():
        check("Table explainers", f"mean-fill cell {e}", "", f"{sg(v_.loc[e,'score_median'])} ({pct(v_.loc[e,'share_positive'])}\\%){star.get(e, '')} \\\\", "faithfulness_mean_voxel")

    # ---- review round 3 (local): size dependence, decisions, attribution mass, sample accounting -------
    SM = {a: {k: pd.read_csv(J / f"analysis_{a}" / "size_mass" / f"{k}.csv") for k in ("size", "size_adj", "decision", "mass")} for a in ARCH}
    def sp(a, x, y, d=2):
        r = SM[a]["size"].set_index(["x", "y"]).loc[(x, y)]
        g = lambda v: f"$-{f(-v, d)}$" if float(f(v, d)) < 0 else f(v, d)  # noqa: E731
        return g(r.spearman), f"[{g(r.ci_lo)}, {g(r.ci_hi)}]"
    v, c = sp("resnet", "diam", "laf"); v2, c2 = sp("densenet", "diam", "laf")
    check("Size", "LAF vs diameter", "IG LAF rose with the equivalent lesion diameter (Spearman ", f"{v} {c},\npatient bootstrap; DenseNet {v2} {c2})", "size_mass/size.csv")
    v, c = sp("resnet", "diam", "laf_adj")
    check("Size", "LAF_adj vs diameter", "remained so after chance adjustment (", f"{v} {c}); the logit", "size_mass/size.csv")
    v, c = sp("resnet", "diam", "logit"); v3, c3 = sp("resnet", "logit", "laf")
    check("Size", "logit vs diameter, LAF vs logit", "fell with diameter (", f"{v} {c}), and LAF was weakly and inconclusively related to the logit ({v3}\n{c3})", "size_mass/size.csv")
    sa = SM["resnet"]["size_adj"]; sa2 = SM["densenet"]["size_adj"]
    check("Size", "size-adjusted ICC", "the between-lesion SD of LAF halved (", f"{f(sa.sd_between_raw.median(),3)} to {f(sa.sd_between_size_adj.median(),3)}) and the scan--rescan ICC was\n{f(sa.icc_size_adj.median(),3)} (median over runs and reconstructions; range {f(sa.icc_size_adj.min(),3)}--{f(sa.icc_size_adj.max(),3)}; DenseNet {f(sa2.icc_size_adj.median(),3)})", "size_mass/size_adj.csv")
    check("Abstract", "size", "LAF tracked lesion size (Spearman ", f"{f(SM['resnet']['size'].set_index(['x','y']).loc[('diam','laf')].spearman)}), and the size-adjusted ICC was {f(sa.icc_size_adj.median())}", "size_mass")
    ms = SM["resnet"]["mass"].set_index("measure")
    v, c = sp("resnet", "diam", "sphere10")
    check("Size", "10 mm sphere", "centroid, was as repeatable (ICC ", f"{f(ms.loc['laf_sphere10','icc_scan_median'],3)}) and fell with size ({v} {c})", "size_mass")
    check("Mass", "negative share", "Negative attribution made up ", f"{pct(ms.loc['neg_share','median'])}\\% of the total absolute IG attribution and its share\nwas repeatable (ICC {f(ms.loc['neg_share','icc_scan_median'],3)})", "size_mass/mass.csv")
    check("Mass", "positive mass", "The absolute positive mass (log) was repeatable between scans (ICC ", f"{f(ms.loc['log_pos_mass','icc_scan_median'],3)}) but not between runs\n(between-run ICC {f(ms.loc['log_pos_mass','icc_run_median'],3)}, vs.\\ {f(ms.loc['laf','icc_run_median'],3)} for LAF)", "size_mass/mass.csv")
    dc = SM["resnet"]["decision"].set_index("group")
    cb, mb, dd = dc.loc["correct in both scans"], dc.loc["missed in both scans"], dc.loc["discordant between scans"]
    check("Decisions", "correct vs missed", "Restricted to pairs classified correctly in both scans (", f"{pct(cb.share_of_pairs)}\\% of pairs, {int(cb.n_patients)} patients), the scan--rescan\nICC of IG LAF was {f(cb.icc_median,3)} (wSD {f(cb.wsd_mean,3)}), as for pairs missed in both scans ({pct(mb.share_of_pairs)}\\%, {int(mb.n_patients)} patients; {f(mb.icc_median,3)}, wSD {f(mb.wsd_mean,3)}); {pct(dd.share_of_pairs)}\\% of\npairs changed class", "size_mass/decision.csv")
    meas_r = R["meas"][(R["meas"].explainer == "ig") & (R["meas"].seed == 0)]
    cnt = meas_r.groupby("setting").size()
    prim_ = meas_r[~meas_r.PatientID.isin(PRIMARY_EXCLUDE)]
    cntp = prim_.groupby("setting").size()
    pairs_ = prim_.groupby(["setting", "PatientID"]).acq.nunique().eq(2).groupby("setting").sum()
    order = ["1.25L", "1.25S", "2.5L", "2.5S", "5L", "5S"]
    check("Table flow", "cells", f"(all 32 patients; {len(meas_r)} in total) & ", " & ".join(str(cnt[o]) for o in order), "measurands.csv (seed 0)")
    check("Table flow", "primary cells", "Cells in the primary population (31 patients) & ", " & ".join(str(cntp[o]) for o in order), "measurands.csv")
    check("Table flow", "pairs", "Patients with both scans (scan--rescan pairs) & ", " & ".join(str(int(pairs_[o])) for o in order), "measurands.csv")

    # ---- review round 3 (server): variance-ratio CIs and offsets, map measures, deletion controls ------
    vi = {a: pd.read_csv(J / f"analysis_{a}" / "robustness" / "vc_inference.csv").query("measure == 'laf'").set_index("quantity") for a in ARCH}
    q = lambda a, k, d: (f(vi[a].loc[k, "estimate"], d), f(vi[a].loc[k, "ci_lo"], d), f(vi[a].loc[k, "ci_hi"], d))  # noqa: E731
    r1, r2, r3, r4 = q("resnet", "sd_ratio", 1), q("resnet", "var_ratio", 0), q("densenet", "sd_ratio", 1), q("densenet", "var_ratio", 0)
    check("VC inference", "ratio CIs", "refits), the SD ratio was ", f"{r1[0]} [{r1[1]}, {r1[2]}] (variance ratio {r2[0]} [{r2[1]}, {r2[2]}]; DenseNet {r3[0]} [{r3[1]}, {r3[2]}] and {r4[0]} [{r4[1]}, {r4[2]}])", "vc_inference.csv (1000 refits)")
    check("VC inference", "offsets and total", "had an SD of ", f"{f(vi['resnet'].loc['sd_run_offsets','estimate'],3)}\nacross runs (range {f(vi['resnet'].loc['offset_range','estimate'],3)}; DenseNet {f(vi['densenet'].loc['sd_run_offsets','estimate'],3)}); including them, the total run-to-run SD of a lesion's LAF was {f(vi['resnet'].loc['sd_run_total','estimate'],3)}\n(DenseNet {f(vi['densenet'].loc['sd_run_total','estimate'],3)}), of which {pct(vi['resnet'].loc['lesion_specific_share','estimate'])}\\% (DenseNet {pct(vi['densenet'].loc['lesion_specific_share','estimate'])}\\%)", "vc_inference.csv")
    mm = {a: pd.read_csv(J / f"analysis_{a}" / "map_measures" / "map_measures_summary.csv").set_index(["explainer", "measure"]) for a in ARCH}
    g_ = lambda a, e, m_, c="median": mm[a].loc[(e, m_), c]  # noqa: E731
    check("Map measures", "core share and density", "Within the lesion, ", f"{pct(g_('resnet','ig','core_share'))}\\% of IG's positive\nattribution lay in the core (SEG eroded by 2~mm) and the core held {f(g_('resnet','ig','core_density_ratio'),1)} times more attribution per voxel than the 2~mm\nrim (DenseNet {pct(g_('densenet','ig','core_share'))}\\%, {f(g_('densenet','ig','core_density_ratio'),1)} times)", "map_measures_summary.csv")
    check("Map measures", "shell", "the SEG received ", f"{f(100*g_('resnet','ig','shell_share'),1)}\\% of the ROI's positive attribution (ICC {f(g_('resnet','ig','shell_share','icc_scan_median'),3)})", "map_measures_summary.csv")
    check("Map measures", "Grad-CAM density", "Grad-CAM showed no such structure (core-to-rim\ndensity ratio ", f"{f(g_('resnet','gradcam','core_density_ratio'),1)})", "map_measures_summary.csv")
    check("Map measures", "negative LAF", "Of IG's negative attribution, ", f"{pct(g_('resnet','ig','neg_laf'))}\\% also lay on the lesion, with a\nscan--rescan ICC of {f(g_('resnet','ig','neg_laf','icc_scan_median'),3)}", "map_measures_summary.csv")
    rr3 = [(f(g_('resnet','ig',f'laf_r{r}')), f(g_('resnet','ig',f'laf_r{r}','icc_scan_median'),3), f(g_('resnet','ig',f'laf_r{r}','wsd_run'),3)) for r in (30, 45, 60)]
    check("Map measures", "ROI radius", "With an ROI radius of 30, 45 and 60~mm, IG LAF was ", f"{rr3[0][0]}, {rr3[1][0]} and {rr3[2][0]}\nand its scan--rescan ICC {rr3[0][1]}, {rr3[1][1]} and {rr3[2][1]}; the between-run wSD was {rr3[0][2]}, {rr3[1][2]} and {rr3[2][2]}", "map_measures_summary.csv")
    fc = {d: pd.read_csv(J / "analysis_resnet" / d / "faithfulness_summary.csv").groupby("explainer") for d in
          ("faithfulness_mean_matched", "faithfulness_blur_matched", "faithfulness_mean_lesion", "insertion_mean")}
    med = lambda d, e: fc[d].score_median.median()[e]  # noqa: E731
    shp = lambda d, e: fc[d].share_positive.median()[e]  # noqa: E731
    check("Controls", "matched random", "(IG (air): mean\nreplacement ", f"{sg(med('faithfulness_mean_matched','ig'))}, positive in {pct(shp('faithfulness_mean_matched','ig'))}\\% of maps; blurred replacement {f(med('faithfulness_blur_matched','ig'))}, {pct(shp('faithfulness_blur_matched','ig'))}\\%)", "faithfulness_*_matched")
    neg_runs = int((pd.read_csv(J / "analysis_resnet" / "faithfulness_mean_lesion" / "faithfulness_summary.csv").query("explainer == 'ig'").score_median < 0).sum())
    check("Controls", "lesion-restricted", "IG (air)\nscored ", f"{f(med('faithfulness_mean_lesion','ig'))} (positive in {pct(shp('faithfulness_mean_lesion','ig'))}\\% of maps, in every run), IG (blur) {f(med('faithfulness_mean_lesion','ig_blur'))} and IG (mean) {f(med('faithfulness_mean_lesion','ig_mean'))}, whereas Grad-CAM ({sg(med('faithfulness_mean_lesion','gradcam'))},\n{pct(shp('faithfulness_mean_lesion','gradcam'))}\\%) and Occlusion ({f(med('faithfulness_mean_lesion','occlusion'))})" if neg_runs == 0 else "IG NEGATIVE IN SOME RUN", "faithfulness_mean_lesion")
    check("Controls", "insertion", "for all explainers (IG (air) ", f"{f(med('insertion_mean','ig'))}, positive in {pct(shp('insertion_mean','ig'))}\\% of\nmaps; Grad-CAM {f(med('insertion_mean','gradcam'))}; Occlusion {f(med('insertion_mean','occlusion'))})", "insertion_mean")
    ins_min = min(med("insertion_mean", e) for e in ("ig", "ig_blur", "ig_mean", "gradcam", "occlusion"))
    rows.append(dict(section="Controls", claim="insertion positive for all explainers", paper="", recomputed=f"min median {ins_min:.2f}", source="insertion_mean", status="MATCH" if ins_min > 0 else "MISMATCH"))

    # ---- Diagnostics review round: specification sensitivity and LAF beyond size (21_review_sensitivity.py) ------
    rs = J / "review_sensitivity"
    if (rs / "vc_spec_sensitivity.csv").exists() and "was re-estimated for IG LAF under" in tex:
        sp_ = pd.read_csv(rs / "vc_spec_sensitivity.csv").set_index(["arch", "spec"]).sd_ratio
        li_ = pd.read_csv(rs / "laf_incremental.csv").set_index("arch")
        r_ = [f(sp_[("resnet", k)], 1) for k in ("primary", "runRE", "reduced", "logitLAF")]
        d_ = [f(sp_[("densenet", k)], 1) for k in ("primary", "runRE", "reduced", "logitLAF")]
        check("Review", "spec sensitivity", "The SD ratio of the lesion-by-run to the lesion-by-scan component was ",
              f"{r_[0]} for the primary ResNet model, {r_[1]} with run as a random effect, {r_[2]} in the reduced model and {r_[3]} on logit(LAF) (DenseNet {d_[0]}, {d_[1]}, {d_[2]} and {d_[3]}).",
              "review_sensitivity/vc_spec_sensitivity.csv")
        check("Review", "size share", "Lesion size explained a median of ",
              f"{pct(li_.loc['resnet','laf_var_explained_by_size_median'])}\\% (ResNet) and {pct(li_.loc['densenet','laf_var_explained_by_size_median'])}\\% (DenseNet)",
              "review_sensitivity/laf_incremental.csv")
        check("Review", "partial correlation", "the residuals were positively correlated (median Spearman ",
              f"{f(li_.loc['resnet','partial_spearman_laf_logit_given_size'])}, 95\\% CI {f(li_.loc['resnet','ci_lo'])}--{f(li_.loc['resnet','ci_hi'])}, ResNet; {f(li_.loc['densenet','partial_spearman_laf_logit_given_size'])}, {f(li_.loc['densenet','ci_lo'])}--{f(li_.loc['densenet','ci_hi'])}, DenseNet)",
              "review_sensitivity/laf_incremental.csv")

    # ---- Diagnostics review round: faithfulness CIs, AUC vs diameter, undefined LAF (22/23) --------------------------
    fci, aucp = J / "review_faithfulness_ci" / "faithfulness_ci.csv", J / "review_empty_laf_auc" / "auc_vs_diameter.csv"
    if fci.exists() and aucp.exists() and "Grad-CAM produced no positive attribution in the ROI" in tex:
        fc_ = pd.read_csv(fci).query("arch == 'resnet' and explainer == 'ig'").set_index("variant")
        sg2 = lambda v: f"$-{abs(v):.2f}$" if v < 0 else f"{v:.2f}"  # noqa: E731
        check("Review", "faithfulness CIs", "of the IG (air) median score were ",
              f"{sg2(fc_.loc['faithfulness_mean_voxel','ci_lo'])} to {sg2(fc_.loc['faithfulness_mean_voxel','ci_hi'])} for ROI-wide deletion with the patch mean, {f(fc_.loc['faithfulness_mean_lesion','ci_lo'])}--{f(fc_.loc['faithfulness_mean_lesion','ci_hi'])} within the lesion and {f(fc_.loc['insertion_mean','ci_lo'])}--{f(fc_.loc['insertion_mean','ci_hi'])} for insertion.",
              "review_faithfulness_ci/faithfulness_ci.csv")
        au = pd.read_csv(aucp).set_index(["arch", "stratum"])
        fm = lambda v: ("+" + f"{v:.3f}") if v >= 0 else f"$-{abs(v):.3f}$"  # noqa: E731
        ci3 = lambda a, st: (lambda lo, hi: f"{(f'$-{abs(lo):.3f}$') if lo < 0 else f'{lo:.3f}'} to {(f'$-{abs(hi):.3f}$') if hi < 0 else f'{hi:.3f}'}")(au.loc[(a, st), "ci_lo"], au.loc[(a, st), "ci_hi"])  # noqa: E731
        check("Review", "AUC vs diameter (all)", "was not distinguishable from zero in all nodules (ResNet ",
              f"{fm(au.loc[('resnet','all'),'diff'])[:6]}, 95\\% CI {ci3('resnet','all')}; DenseNet", "review_empty_laf_auc/auc_vs_diameter.csv")
    # undefined-LAF paragraph numbers are checked by hand against empty_share.csv / empty_laf_icc.csv (see notes)

    # ---- Diagnostics review round: grid-phase control (24_grid_phase_control.py, run on the server) -----------------
    gp = {a: J / "review_grid_phase" / a / "grid_phase_summary.csv" for a in ARCH}
    if all(p_.exists() for p_ in gp.values()) and "Translating the identical CT patch" in tex:
        g_ = {a: pd.read_csv(p_).set_index("variant") for a, p_ in gp.items()}
        v2 = lambda var, col="spearman": (f(g_["resnet"].loc[var, col]), f(g_["densenet"].loc[var, col]))  # noqa: E731
        d1, x1, d3, d2, d16, x16, sb, mi = (v2(k) for k in ("diag1", "x1", "diag3", "diag2", "diag16", "x16", "sub0.5", "mapinterp"))
        check("Grid phase", "1 and 3 voxel shifts", "Translating the identical CT patch by one voxel gave a median Spearman correlation of ",
              f"{d1[0]} (ResNet) and {d1[1]} (DenseNet) with the unshifted IG map (along $x$ only: {x1[0]} and {x1[1]}), and by three voxels {d3[0]} and {d3[1]}, as low as",
              "review_grid_phase/*/grid_phase_summary.csv")
        check("Grid phase", "even shifts", "Even shifts agreed better (2 voxels ",
              f"{d2[0]} and {d2[1]}; 16 voxels {d16[0]} and {d16[1]}; 16 voxels along $x$ only {x16[0]} and {x16[1]}).", "review_grid_phase/*/grid_phase_summary.csv")
        check("Grid phase", "resampling controls", "A half-voxel translation by resampling gave ",
              f"{sb[0]} and {sb[1]}, and resampling the unshifted map alone by half a voxel gave {mi[0]} and {mi[1]}.", "review_grid_phase/*/grid_phase_summary.csv")
        pmin = min(g_[a].spearman_pooled16.min() for a in ARCH)
        check("Grid phase", "pooled agreement", "Agreement pooled to 16~mm cells stayed at ", f"{f(pmin)} or higher in every variant.", "review_grid_phase/*/grid_phase_summary.csv")
        ab = pd.concat([g_[a].abs_dlogit_median for a in ARCH]).replace(0, np.nan).dropna()
        check("Grid phase", "logit change", "median absolute logit change ", f"{f(ab.min())}--{f(ab.max())} across variants", "review_grid_phase/*/grid_phase_summary.csv")

    # ---- report ----------------------------------------------------------------------------------
    df = pd.DataFrame(rows)
    out = ROOT / paper_dir
    df.to_csv(out / "validation_sheet.csv", index=False)
    md = ["# CMC manuscript validation sheet", "", f"{len(df)} checks: {df.status.value_counts().to_dict()}", "",
          "| section | claim | status | paper | recomputed | source |", "|---|---|---|---|---|---|"]
    md += [f"| {r.section} | {r.claim} | {r.status} | {str(r.paper).replace('|', '/')} | {str(r.recomputed).replace('|', '/')} | {r.source} |" for r in df.itertuples()]
    (out / "validation_sheet.md").write_text("\n".join(md) + "\n")
    bad = df[~df.status.isin(["MATCH", "INFO"])]
    with pd.option_context("display.width", 250, "display.max_colwidth", 90):
        print(bad[["section", "claim", "status", "paper", "recomputed"]].to_string(index=False) if len(bad) else "all checks MATCH")
    print(f"\n{len(df)} checks: {df.status.value_counts().to_dict()}\nWrote {out / 'validation_sheet.md'} and .csv")


if __name__ == "__main__":
    main()
