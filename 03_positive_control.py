#!/usr/bin/env python3
"""
GAUGE-XAI step 3 (positive control): scan-rescan repeatability of conventional lesion
measurements, per reconstruction setting, with patient-level bootstrap CIs.

If this pipeline cannot reproduce sensible repeatability for volume and diameter, do not
move on to saliency. The same functions are reused later for the saliency measurands.

Ratio-type measurands (volume, diameter) are analysed on the log scale; RC is then a
percentage. HU is analysed on its own scale.

Usage:
  python 03_positive_control.py --stats audit/lesion_stats.csv --out audit
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from gauge_lib import PRIMARY_EXCLUDE, RESULTS_DIR, setting_role

SETTINGS = ["1.25L", "1.25S", "2.5L", "2.5S", "5L", "5S"]
MEASURES = {"volume_ml": "log", "longest_axial_mm": "log", "hu_mean": "linear"}


def icc_1_1(y):
    """One-way random, absolute agreement, single measurement. y: (n_subjects, k)."""
    n, k = y.shape
    m = y.mean(1)
    msb = k * ((m - y.mean()) ** 2).sum() / (n - 1)
    msw = ((y - m[:, None]) ** 2).sum() / (n * (k - 1))
    return (msb - msw) / (msb + (k - 1) * msw), np.sqrt(msw)


def summarise(pairs, scale):
    """pairs: (n, 2) array of scan1/scan2 values."""
    y = np.log(pairs) if scale == "log" else pairs
    icc, wsd = icc_1_1(y)
    rc = 2.77 * wsd
    if scale == "log":
        pct = 100 * (pairs[:, 1] - pairs[:, 0]) / pairs.mean(1)
        return dict(icc=icc, wCV_pct=100 * np.sqrt(np.exp(wsd ** 2) - 1),
                    RC_pct_lo=100 * (np.exp(-rc) - 1), RC_pct_hi=100 * (np.exp(rc) - 1),
                    BA_mean_pct=pct.mean(), BA_loa_lo=pct.mean() - 1.96 * pct.std(ddof=1),
                    BA_loa_hi=pct.mean() + 1.96 * pct.std(ddof=1))
    d = pairs[:, 1] - pairs[:, 0]
    return dict(icc=icc, wSD=wsd, RC=rc, BA_mean=d.mean(),
                BA_loa_lo=d.mean() - 1.96 * d.std(ddof=1), BA_loa_hi=d.mean() + 1.96 * d.std(ddof=1))


def bootstrap(pairs, scale, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(pairs)
    iccs, rcs = [], []
    for _ in range(n_boot):
        s = pairs[rng.integers(0, n, n)]
        y = np.log(s) if scale == "log" else s
        icc, wsd = icc_1_1(y)
        iccs.append(icc)
        rcs.append(2.77 * wsd)
    return np.percentile(iccs, [2.5, 97.5]), np.percentile(rcs, [2.5, 97.5])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stats", default=RESULTS_DIR / "audit" / "lesion_stats.csv", type=Path)
    ap.add_argument("--out", default=RESULTS_DIR / "audit", type=Path)
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()

    df = pd.read_csv(args.stats)
    if "qc_ok" in df:
        print(f"Dropping {(~df['qc_ok']).sum()} cells rejected by 02_rider_qc.py (qc_ok=False)")
        df = df[df["qc_ok"]]
    # primary: pre-registered exclusions applied; all: every patient (sensitivity analysis)
    populations = {"primary": df[~df["PatientID"].isin(PRIMARY_EXCLUDE)], "all": df}
    out = []
    for pop, dpop in populations.items():
        for measure, scale in MEASURES.items():
            for s in SETTINGS:
                sub = dpop[dpop.setting == s]
                if sub.empty:
                    continue
                w = sub.pivot_table(index="PatientID", columns="acq", values=measure)
                if not {"scan1", "scan2"} <= set(w.columns):
                    continue
                w = w[["scan1", "scan2"]].dropna()
                if len(w) < 5:
                    continue
                pairs = w[["scan1", "scan2"]].to_numpy(float)
                if scale == "log" and (pairs <= 0).any():
                    continue
                res = summarise(pairs, scale)
                (ilo, ihi), (rlo, rhi) = bootstrap(pairs, scale, args.n_boot)
                out.append(dict(population=pop, measure=measure, setting=s, setting_role=setting_role(s),
                                n=len(pairs), **res, icc_ci_lo=ilo, icc_ci_hi=ihi, rc_ci_lo=rlo, rc_ci_hi=rhi))

    res = pd.DataFrame(out).round(3)
    res.to_csv(args.out / "positive_control_repeatability.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(res[res.population == "primary"].to_string(index=False))
    print(f"\nPrimary population excludes: {PRIMARY_EXCLUDE}. Rows with population=all are the sensitivity "
          f"analysis. setting_role=secondary marks reconstructions reported as secondary (small n).")
    print("\nSanity check: Zhao et al. (Radiology 2009) report 95% limits of agreement of about "
          "-12% to +13% for computer-aided volume on the 1.25L series. Your volume BA limits for "
          "1.25L should be in that neighbourhood (contours differ, so not identical). "
          "rc_ci_* are on the analysis scale (log units for volume/diameter).")


if __name__ == "__main__":
    main()
