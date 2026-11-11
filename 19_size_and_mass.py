#!/usr/bin/env python3
"""
GAUGE-XAI step 19 (journal, review round 3): what the lesion-level summary depends on. From measurands.csv only
(no maps, no GPU), per architecture, IG (air), primary population and reconstructions:

  size.csv         patient-level Spearman of LAF, LAF_adj, fixed-10 mm-sphere fraction, entropy and logit with the
                   equivalent lesion diameter (and LAF with the logit), 2,000 patient bootstrap CIs
  size_adj.csv     scan-rescan ICC of LAF before and after removing the part of the between-lesion variation that is
                   explained by log diameter (patient-level regression; the within-lesion error is unchanged)
  decision.csv     scan-rescan agreement for pairs classified correctly (logit > 0) in both scans vs. missed in both
  mass.csv         repeatability (scan-rescan ICC) and between-run ICC of the absolute positive attribution mass
                   (log), the negative share of |attribution|, the fixed-10 mm-sphere fraction and the entropy

Usage:  python 19_size_and_mass.py [--journal ../results/journal]
"""
import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from gauge_lib import PRIMARY_EXCLUDE, RESULTS_DIR, SECONDARY_SETTINGS, SETTINGS

P = [s for s in SETTINGS if s not in SECONDARY_SETTINGS]
spec = importlib.util.spec_from_file_location("pc", Path(__file__).with_name("03_positive_control.py"))
PC = importlib.util.module_from_spec(spec)
spec.loader.exec_module(PC)


def icc_pairs(g, value):
    w = g.pivot_table(index="PatientID", columns="acq", values=value).dropna()
    return PC.icc_1_1(w[["scan1", "scan2"]].to_numpy(float)) if len(w) >= 5 else (np.nan, np.nan)


def size(ig, n_boot):
    pl = ig.groupby("PatientID").agg(laf=("laf", "median"), diam=("diam", "median"), logit=("logit", "median"),
                                     sphere10=("laf_sphere10", "median"), entropy=("entropy_norm", "median"),
                                     chance=("laf_chance", "median"))
    pl["laf_adj"] = (pl.laf - pl.chance) / (1 - pl.chance)
    rows = []
    for x, y in (("diam", "laf"), ("diam", "laf_adj"), ("diam", "sphere10"), ("diam", "entropy"), ("logit", "laf"), ("diam", "logit")):
        bs = [spearmanr(*pl.sample(len(pl), replace=True, random_state=i)[[x, y]].T.values).correlation for i in range(n_boot)]
        rows.append(dict(x=x, y=y, n_patients=len(pl), spearman=spearmanr(pl[x], pl[y]).correlation,
                         ci_lo=np.percentile(bs, 2.5), ci_hi=np.percentile(bs, 97.5)))
    return pd.DataFrame(rows)


def size_adjusted(ig):
    rows = []
    for (seed, st), g in ig.groupby(["seed", "setting"]):
        w = g.pivot_table(index="PatientID", columns="acq", values=["laf", "eq_radius_mm"]).dropna()
        if len(w) < 5:
            continue
        y = w["laf"][["scan1", "scan2"]].to_numpy(float)
        x = np.log(w["eq_radius_mm"].mean(axis=1).to_numpy(float))
        mean = y.mean(1)
        res = mean - np.polyval(np.polyfit(x, mean, 1), x)
        msw = ((y - y.mean(1, keepdims=True)) ** 2).sum() / len(y)
        vb_adj, vb_raw = max(res.var(ddof=2) - msw / 2, 0), max(mean.var(ddof=1) - msw / 2, 0)
        rows.append(dict(seed=seed, setting=st, n=len(y), icc_raw=vb_raw / (vb_raw + msw), icc_size_adj=vb_adj / (vb_adj + msw),
                         wsd=np.sqrt(msw), sd_between_raw=np.sqrt(vb_raw), sd_between_size_adj=np.sqrt(vb_adj)))
    return pd.DataFrame(rows)


def decision(ig):
    w = ig.pivot_table(index=["seed", "setting", "PatientID"], columns="acq", values=["laf", "logit"]).dropna()
    groups = {"correct in both scans": w[(w[("logit", "scan1")] > 0) & (w[("logit", "scan2")] > 0)],
              "missed in both scans": w[(w[("logit", "scan1")] <= 0) & (w[("logit", "scan2")] <= 0)]}
    rows = []
    for lab, ww in groups.items():
        iccs, wsd = [], []
        for _, g in ww.groupby(level=[0, 1]):
            y = g["laf"][["scan1", "scan2"]].to_numpy(float)
            if len(y) >= 5:
                i, s = PC.icc_1_1(y)
                iccs.append(i)
                wsd.append(s)
        rows.append(dict(group=lab, share_of_pairs=len(ww) / len(w), n_patients=ww.index.get_level_values(2).nunique(),
                         n_strata=len(iccs), icc_median=np.median(iccs), wsd_mean=np.mean(wsd)))
    rows.append(dict(group="discordant between scans", share_of_pairs=1 - sum(len(v) for v in groups.values()) / len(w)))
    return pd.DataFrame(rows)


def mass(ig):
    ig = ig.assign(log_pos_mass=np.log(ig.pos_total), neg_share=-ig.neg_total / (ig.pos_total - ig.neg_total))
    rows = []
    for meas in ("laf", "log_pos_mass", "neg_share", "laf_sphere10", "entropy_norm"):
        scan = [icc_pairs(g, meas)[0] for _, g in ig.groupby(["seed", "setting"])]
        run = []
        for _, g in ig.groupby(["acq", "setting"]):
            w = g.pivot_table(index="PatientID", columns="seed", values=meas).dropna()
            if len(w) >= 5 and w.shape[1] >= 2:
                run.append(PC.icc_1_1(w.to_numpy(float))[0])
        rows.append(dict(measure=meas, median=ig[meas].median(), icc_scan_median=np.nanmedian(scan), icc_scan_min=np.nanmin(scan),
                         icc_scan_max=np.nanmax(scan), icc_run_median=np.nanmedian(run)))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--journal", default=RESULTS_DIR / "journal", type=Path)
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()
    for a in ("resnet", "densenet"):
        m = pd.read_csv(args.journal / f"saliency_{a}" / "measurands.csv")
        ig = m[(m.explainer == "ig") & ~m.PatientID.isin(PRIMARY_EXCLUDE) & m.setting.isin(P)].assign(diam=lambda d: 2 * d.eq_radius_mm)
        out = args.journal / f"analysis_{a}" / "size_mass"
        out.mkdir(parents=True, exist_ok=True)
        for name, df in (("size", size(ig, args.n_boot)), ("size_adj", size_adjusted(ig)), ("decision", decision(ig)), ("mass", mass(ig))):
            df.round(5).to_csv(out / f"{name}.csv", index=False)
            print(f"\n{a} / {name}\n{df.round(3).to_string(index=False)}")


if __name__ == "__main__":
    main()
