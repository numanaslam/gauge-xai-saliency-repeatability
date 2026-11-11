#!/usr/bin/env python3
"""
GAUGE-XAI step 21 (Diagnostics review round): two targeted reanalyses requested by the pre-submission review.

A. Variance-decomposition specification sensitivity for IG LAF (per architecture), SD ratio of the patient x run to the
   patient x scan component under:
     primary   fixed kernel, thickness and run; random patient + 4 variance components (as 20_vc_inference.py)
     runRE     run as a RANDOM effect (crossed with patient): random patient, run, patient x scan/run/kernel/thickness
     reduced   fixed kernel/thickness; random patient, run, patient x scan, patient x run (no kernel/thickness interactions)
     logitLAF  primary specification on logit(LAF) (LAF is bounded in (0,1))
   Point estimates (bootstrap CIs with --n-boot > 0, patient-cluster refits).
B. Incremental information of LAF beyond lesion size: LAF and the model logit are each regressed on log equivalent radius
   (and log volume proxy) with a quadratic term; the partial Spearman correlation of the two residuals is reported per run,
   its median over runs, and a patient-cluster bootstrap CI of that median. Also: share of LAF variance explained by size.

Usage (from code/):  python 21_review_sensitivity.py --n-boot 0     (quick)   |   --n-boot 200 --jobs 4
Writes results/journal/review_sensitivity/{vc_spec_sensitivity.csv, laf_incremental.csv}.
"""
import argparse
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from gauge_lib import PRIMARY_EXCLUDE, RESULTS_DIR, SECONDARY_SETTINGS, SETTINGS

P = [s for s in SETTINGS if s not in SECONDARY_SETTINGS]


def load(sal):
    m = pd.read_csv(sal / "measurands.csv")
    d = m[(m.explainer == "ig") & ~m.PatientID.isin(PRIMARY_EXCLUDE) & m.setting.isin(P)].copy()
    d["kernel"], d["thick"] = d.setting.str[-1], d.setting.str[:-1]
    return d


def _vc(res, model):
    return dict(zip(model.exog_vc.names, res.vcomp))


def fit_spec(d, spec, meas="laf"):
    import statsmodels.formula.api as smf
    d = d.copy().reset_index(drop=True)
    if spec == "logitLAF":
        d["y"] = np.log(d[meas].clip(1e-3, 1 - 1e-3) / (1 - d[meas].clip(1e-3, 1 - 1e-3)))
    else:
        d["y"] = d[meas]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if spec in ("primary", "logitLAF"):
            vc = {"patient_x_scan": "0 + C(acq)", "patient_x_seed": "0 + C(seed)",
                  "patient_x_kernel": "0 + C(kernel)", "patient_x_thickness": "0 + C(thick)"}
            mod = smf.mixedlm("y ~ C(kernel) + C(thick) + C(seed)", d, groups="PatientID", re_formula="1", vc_formula=vc)
            res = mod.fit(reml=True, method=["lbfgs", "powell"])
            comps = _vc(res, mod)
            run, scan = comps["patient_x_seed"], comps["patient_x_scan"]
        else:
            d["all"] = 1
            d["pid_acq"] = d.PatientID + "_" + d.acq
            d["pid_seed"] = d.PatientID + "_" + d.seed.astype(str)
            d["pid_kernel"] = d.PatientID + "_" + d.kernel
            d["pid_thick"] = d.PatientID + "_" + d.thick
            vc = {"patient": "0 + C(PatientID)", "run": "0 + C(seed)", "patient_x_scan": "0 + C(pid_acq)",
                  "patient_x_seed": "0 + C(pid_seed)"}
            if spec == "runRE":
                vc.update({"patient_x_kernel": "0 + C(pid_kernel)", "patient_x_thickness": "0 + C(pid_thick)"})
            mod = smf.mixedlm("y ~ C(kernel) + C(thick)", d, groups="all", vc_formula=vc)
            res = mod.fit(reml=True, method=["lbfgs", "powell"])
            comps = _vc(res, mod)
            run, scan = comps["patient_x_seed"], comps["patient_x_scan"]
    return dict(sd_run=np.sqrt(max(run, 0)), sd_scan=np.sqrt(max(scan, 0)),
                sd_ratio=np.sqrt(max(run, 0) / scan) if scan > 0 else np.nan,
                sd_run_wide=np.sqrt(max(comps.get("run", np.nan), 0)) if "run" in comps else np.nan,
                converged=bool(res.converged))


def _boot(job):
    d, spec, seed = job
    rng = np.random.default_rng(seed)
    pats = d.PatientID.unique()
    pick = rng.choice(pats, len(pats), replace=True)
    db = pd.concat([d[d.PatientID == p].assign(PatientID=f"{p}#{i}") for i, p in enumerate(pick)], ignore_index=True)
    try:
        return fit_spec(db, spec)["sd_ratio"]
    except Exception:  # noqa: BLE001
        return np.nan


def part_a(d, arch, n_boot, jobs):
    rows = []
    for spec in ("primary", "runRE", "reduced", "logitLAF"):
        est = fit_spec(d, spec)
        lo = hi = np.nan
        if n_boot:
            with ProcessPoolExecutor(jobs) as ex:
                b = np.array(list(ex.map(_boot, [(d, spec, i) for i in range(n_boot)], chunksize=4)))
            b = b[~np.isnan(b)]
            lo, hi = np.percentile(b, [2.5, 97.5])
        rows.append(dict(arch=arch, spec=spec, sd_run=est["sd_run"], sd_scan=est["sd_scan"], sd_ratio=est["sd_ratio"],
                         ci_lo=lo, ci_hi=hi, sd_run_wide=est["sd_run_wide"], converged=est["converged"]))
        print(rows[-1], flush=True)
    return rows


def resid(y, X):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return y - X @ beta


def partial_spearman(g):
    r = np.log(g.eq_radius_mm.values)
    X = np.column_stack([np.ones(len(g)), r, r ** 2])
    ra, rb = resid(g.laf.values, X), resid(g.logit.values, X)
    r2 = 1 - np.var(resid(g.laf.values, X)) / np.var(g.laf.values)
    return spearmanr(ra, rb)[0], r2


def part_b(d, arch, n_boot):
    # one row per (patient, acq, setting, seed); primary 1.25L endpoint plus all primary settings
    out = []
    def stat(df):
        per = [partial_spearman(g) for _, g in df.groupby("seed")]
        return float(np.median([p[0] for p in per])), float(np.median([p[1] for p in per]))
    est, r2 = stat(d)
    rng = np.random.default_rng(0)
    pats = d.PatientID.unique()
    bs = []
    for i in range(n_boot):
        pick = rng.choice(pats, len(pats), replace=True)
        db = pd.concat([d[d.PatientID == p].assign(PatientID=f"{p}#{k}") for k, p in enumerate(pick)], ignore_index=True)
        bs.append(stat(db)[0])
    lo, hi = (np.percentile(bs, [2.5, 97.5]) if bs else (np.nan, np.nan))
    raw = float(np.median([spearmanr(g.laf, g.logit)[0] for _, g in d.groupby("seed")]))
    out.append(dict(arch=arch, raw_spearman_laf_logit=raw, partial_spearman_laf_logit_given_size=est, ci_lo=lo, ci_hi=hi,
                    laf_var_explained_by_size_median=r2, n=len(d)))
    print(out[-1], flush=True)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-boot", type=int, default=0, help="patient-cluster refits for the SD-ratio CIs in part A (slow for runRE)")
    ap.add_argument("--n-boot-b", type=int, default=500, help="patient-cluster resamples for part B")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--archs", default="resnet,densenet")
    args = ap.parse_args()
    outd = RESULTS_DIR / "journal" / "review_sensitivity"
    outd.mkdir(parents=True, exist_ok=True)
    A, B = [], []
    for arch in args.archs.split(","):
        d = load(RESULTS_DIR / "journal" / f"saliency_{arch}").dropna(subset=["laf", "logit"])
        d = d[["PatientID", "acq", "seed", "kernel", "thick", "laf", "logit", "eq_radius_mm"]]
        A += part_a(d, arch, args.n_boot, args.jobs)
        B += part_b(d, arch, args.n_boot_b)
    pd.DataFrame(A).round(4).to_csv(outd / "vc_spec_sensitivity.csv", index=False)
    pd.DataFrame(B).round(4).to_csv(outd / "laf_incremental.csv", index=False)
    print(pd.DataFrame(A).round(3).to_string(index=False))
    print(pd.DataFrame(B).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
