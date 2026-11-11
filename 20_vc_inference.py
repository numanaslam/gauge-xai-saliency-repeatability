#!/usr/bin/env python3
"""
GAUGE-XAI step 20 (journal, review round 3): inference for the variance-component ratios and the run offsets.

Same mixed model as 10_robustness.py (fixed kernel, thickness and run; random patient; variance components
patient x scan / run / kernel / thickness). For IG LAF and the logit, per architecture:
  - SD ratio and variance ratio of the patient x run to the patient x scan component, with a patient-cluster
    bootstrap CI computed directly on the ratio (--n-boot refits, default 1000);
  - the fixed run offsets (run-wide shifts removed by the fixed run effect): their SD across runs;
  - total run-to-run SD = sqrt(SD(offsets)^2 + SD(patient x run)^2), i.e. the variation between two separately
    trained models' measurements of the same lesion including the global offset, and the share of it that is
    lesion-specific.
Writes <out>/vc_inference.csv.  Usage:
  python 20_vc_inference.py --saliency D:\\paper8\\results\\journal\\saliency_resnet --out D:\\paper8\\results\\journal\\analysis_resnet\\robustness --jobs 8
"""
import argparse
import importlib.util
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from gauge_lib import PRIMARY_EXCLUDE, RESULTS_DIR, SECONDARY_SETTINGS, SETTINGS

P = [s for s in SETTINGS if s not in SECONDARY_SETTINGS]
VC = {"patient_x_scan": "0 + C(acq)", "patient_x_seed": "0 + C(seed)", "patient_x_kernel": "0 + C(kernel)",
      "patient_x_thickness": "0 + C(thick)"}


def fit(d, meas):
    import statsmodels.formula.api as smf

    d = d.rename(columns={meas: "y"}).reset_index(drop=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = smf.mixedlm("y ~ C(kernel) + C(thick) + C(seed)", d, groups="PatientID", re_formula="1", vc_formula=VC)
        res = model.fit(reml=True, method=["lbfgs", "powell"])
    comps = dict(zip(model.exog_vc.names, res.vcomp))
    offs = np.array([0.0] + [v for k, v in res.fe_params.items() if k.startswith("C(seed)")])
    run, scan = comps["patient_x_seed"], comps["patient_x_scan"]
    sd_off = float(np.std(offs, ddof=1))
    total = np.sqrt(sd_off ** 2 + max(run, 0))
    return dict(sd_run=np.sqrt(max(run, 0)), sd_scan=np.sqrt(max(scan, 0)),
                sd_ratio=np.sqrt(max(run, 0) / scan) if scan > 0 else np.nan,
                var_ratio=run / scan if scan > 0 else np.nan, sd_run_offsets=sd_off,
                offset_range=float(offs.max() - offs.min()), sd_run_total=total,
                lesion_specific_share=max(run, 0) / total ** 2 if total > 0 else np.nan, converged=bool(res.converged))


def _boot(job):
    d, meas, seed = job
    rng = np.random.default_rng(seed)
    pats = d.PatientID.unique()
    pick = rng.choice(pats, len(pats), replace=True)
    db = pd.concat([d[d.PatientID == p].assign(PatientID=f"{p}#{i}") for i, p in enumerate(pick)], ignore_index=True)
    try:
        return fit(db, meas)
    except Exception:  # noqa: BLE001
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--saliency", default=RESULTS_DIR / "journal" / "saliency_resnet", type=Path)
    ap.add_argument("--out", default=RESULTS_DIR / "journal" / "analysis_resnet" / "robustness", type=Path)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--jobs", type=int, default=4)
    args = ap.parse_args()
    m = pd.read_csv(args.saliency / "measurands.csv")
    d0 = m[(m.explainer == "ig") & ~m.PatientID.isin(PRIMARY_EXCLUDE) & m.setting.isin(P)].copy()
    d0["kernel"], d0["thick"] = d0.setting.str[-1], d0.setting.str[:-1]
    rows = []
    for meas in ("laf", "logit"):
        d = d0.dropna(subset=[meas])[["PatientID", "acq", "seed", "kernel", "thick", meas]]
        est = fit(d, meas)
        with ProcessPoolExecutor(args.jobs) as ex:
            boots = [b for b in ex.map(_boot, [(d, meas, i) for i in range(args.n_boot)], chunksize=8) if b]
        bdf = pd.DataFrame(boots)
        for k, v in est.items():
            if k == "converged":
                continue
            lo, hi = (np.nanpercentile(bdf[k], [2.5, 97.5]) if k in bdf else (np.nan, np.nan))
            rows.append(dict(measure=meas, quantity=k, estimate=v, ci_lo=lo, ci_hi=hi, n_boot_ok=len(bdf),
                             converged=est["converged"]))
        print(f"{meas}: {len(bdf)}/{args.n_boot} refits", flush=True)
    out = pd.DataFrame(rows)
    args.out.mkdir(parents=True, exist_ok=True)
    out.round(5).to_csv(args.out / "vc_inference.csv", index=False)
    print(out.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
