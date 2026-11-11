#!/usr/bin/env python3
"""
GAUGE-XAI step 13: paired patient-cluster bootstrap for the centre-prior ablation.

Both arms explain the same RIDER cells with models trained from the same seeds on the same LIDC split, so
the comparison is paired by patient: each bootstrap resample draws patients with replacement and recomputes
every metric in BOTH arms on that same sample. Reported: estimate per arm, difference (arm B - arm A) and
its 95% CI. This quantifies how different the arms are; it is not a pre-specified equivalence test.

Metrics (IG unless stated; primary population and reconstructions):
  wSD_scan      pooled scan-rescan within-lesion SD (all seeds and reconstructions)
  wSD_kernel    pooled L-vs-S within-lesion SD at fixed thickness (scan 1)
  wSD_run       pooled between-run within-lesion SD (same cell, k runs)
  icc_primary   scan-rescan ICC(1,1) at 1.25L, mean over runs
  outside_share median share of the own-SEG LAF that a centred size-matched foreign SEG captures outside the
                own lesion (needs controls.csv with laf_mismatch_outside)
  logit_wSD_scan  pooled scan-rescan SD of the logit

Usage:
  python 13_ablation_ci.py --a ../results/ablation/measurands_prior64.csv --b ../results/ablation/measurands_noprior64.csv \\
      --controls-a ../results/ablation/analysis_prior64/robustness/controls.csv \\
      --controls-b ../results/ablation/analysis_noprior64/robustness/controls.csv --names prior no-prior \\
      --out ../results/ablation/ablation_ci.csv
"""
import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

from gauge_lib import PRIMARY_EXCLUDE, PRIMARY_SETTING, SECONDARY_SETTINGS, SETTINGS

P = [s for s in SETTINGS if s not in SECONDARY_SETTINGS]


def load_by_path(name, fname):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(fname))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


PC = load_by_path("positive_control", "03_positive_control.py")


def prep(m):
    m = m[~m.PatientID.isin(PRIMARY_EXCLUDE) & m.setting.isin(P)]
    ig = m[m.explainer == "ig"]
    scan = ig.pivot_table(index=["PatientID", "seed", "setting"], columns="acq", values=["laf", "logit"]).dropna()
    s1 = ig[ig.acq == "scan1"]
    ker = []
    for t in ("1.25", "2.5", "5"):
        w = s1[s1.setting.isin([t + "L", t + "S"])].pivot_table(index=["PatientID", "seed"], columns="setting", values="laf")
        if {t + "L", t + "S"} <= set(w.columns):
            ker.append(pd.DataFrame({"PatientID": w.index.get_level_values(0), "d": (w[t + "S"] - w[t + "L"]).values}).dropna())
    ker = pd.concat(ker, ignore_index=True)
    run = ig.pivot_table(index=["PatientID", "acq", "setting"], columns="seed", values="laf").dropna()
    return scan, ker, run


def stats(scan, ker, run, pats, ctl=None):
    sc = scan[scan.index.get_level_values(0).isin(pats)] if pats is not None else scan
    d = sc[("laf", "scan2")] - sc[("laf", "scan1")]
    dl = sc[("logit", "scan2")] - sc[("logit", "scan1")]
    kk = ker[ker.PatientID.isin(pats)] if pats is not None else ker
    rr = run[run.index.get_level_values(0).isin(pats)] if pats is not None else run
    out = dict(wSD_scan=np.sqrt(np.mean(d ** 2) / 2), wSD_kernel=np.sqrt(np.mean(kk.d ** 2) / 2),
               wSD_run=np.sqrt(np.mean(rr.var(axis=1, ddof=1))), logit_wSD_scan=np.sqrt(np.mean(dl ** 2) / 2))
    iccs = []
    for seed in sc.index.get_level_values(1).unique():
        x = sc.xs((seed, PRIMARY_SETTING), level=("seed", "setting"))["laf"][["scan1", "scan2"]].to_numpy(float)
        if len(x) >= 5:
            iccs.append(PC.icc_1_1(x)[0])
    out["icc_primary"] = float(np.mean(iccs)) if iccs else np.nan
    if ctl is not None:
        c = ctl[ctl.PatientID.isin(pats)] if pats is not None else ctl
        out["outside_share"] = float((c.laf_mismatch_outside / c.laf_check).median())
    return out


def resample(pats_all, rng):
    """Bootstrap patients; duplicates are kept by repeating their rows (implemented via a weight table)."""
    return rng.choice(pats_all, len(pats_all), replace=True)


def boot_stats(scan, ker, run, pick, ctl):
    """Statistics on a bootstrap sample with duplicated patients (rows repeated per draw)."""
    counts = pd.Series(pick).value_counts()
    rep = lambda df, key: df.loc[np.repeat(df.index.values, df[key].map(counts).fillna(0).astype(int).values)]  # noqa: E731
    sc = scan.reset_index()
    sc_b = rep(sc, "PatientID").set_index(["PatientID", "seed", "setting"])
    sc_b.columns = scan.columns
    rr = run.reset_index()
    rr_b = rep(rr, "PatientID").set_index(["PatientID", "acq", "setting"])
    rr_b.columns = run.columns
    kk_b = rep(ker, "PatientID")
    cc_b = rep(ctl.reset_index(drop=True), "PatientID") if ctl is not None else None
    return stats(sc_b, kk_b, rr_b, None, cc_b)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", required=True, type=Path)
    ap.add_argument("--b", required=True, type=Path)
    ap.add_argument("--controls-a", type=Path)
    ap.add_argument("--controls-b", type=Path)
    ap.add_argument("--names", nargs=2, default=["A", "B"])
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    arms = {}
    for name, mpath, cpath in ((args.names[0], args.a, args.controls_a), (args.names[1], args.b, args.controls_b)):
        m = pd.read_csv(mpath)
        ctl = None
        if cpath and cpath.exists():
            ctl = pd.read_csv(cpath)
            ctl = ctl[(ctl.explainer == "ig") & ~ctl.PatientID.isin(PRIMARY_EXCLUDE) & ctl.setting.isin(P)]
            ctl = ctl.dropna(subset=["laf_mismatch_outside"]) if "laf_mismatch_outside" in ctl else None
        arms[name] = (*prep(m), ctl)
    pats = sorted(set(arms[args.names[0]][0].index.get_level_values(0)) & set(arms[args.names[1]][0].index.get_level_values(0)))
    est = {n: stats(*arms[n][:3], pats, arms[n][3]) for n in args.names}
    rng = np.random.default_rng(0)
    boots = {n: [] for n in args.names}
    for _ in range(args.n_boot):
        pick = resample(np.array(pats), rng)
        for n in args.names:
            boots[n].append(boot_stats(*arms[n][:3], pick, arms[n][3]))
    rows = []
    for metric in est[args.names[0]]:
        a = np.array([b[metric] for b in boots[args.names[0]]])
        b = np.array([b[metric] for b in boots[args.names[1]]])
        diff = b - a
        rows.append(dict(metric=metric, n_patients=len(pats),
                         **{f"{args.names[0]}": est[args.names[0]][metric], f"{args.names[1]}": est[args.names[1]][metric]},
                         **{f"{args.names[0]}_ci_lo": np.nanpercentile(a, 2.5), f"{args.names[0]}_ci_hi": np.nanpercentile(a, 97.5),
                            f"{args.names[1]}_ci_lo": np.nanpercentile(b, 2.5), f"{args.names[1]}_ci_hi": np.nanpercentile(b, 97.5)},
                         difference=est[args.names[1]][metric] - est[args.names[0]][metric],
                         diff_ci_lo=np.nanpercentile(diff, 2.5), diff_ci_hi=np.nanpercentile(diff, 97.5)))
    res = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    res.round(5).to_csv(args.out, index=False)
    with pd.option_context("display.width", 220, "display.max_columns", 20):
        print(res.round(4).to_string(index=False))
    print(f"\nWrote {args.out}  ({len(pats)} patients, {args.n_boot} paired patient-cluster resamples)")


if __name__ == "__main__":
    main()
