#!/usr/bin/env python3
"""
GAUGE-XAI step 23 (Diagnostics review round, experiments 2 and 3).

Part 2, undefined LAF: LAF is undefined when a map has no positive attribution in the ROI (``empty_map``; in this
study only Grad-CAM, for confidently benign cases). For every primary reconstruction (empty maps occur only at 1.25S, 5L and 5S, not at the
1.25L endpoint) and every explainer, the scan-rescan ICC(1,1) and wSD of LAF are computed per run and summarised by the median over runs under
  (a) drop       pairs with an undefined LAF in either scan are dropped        (the manuscript's handling)
  (b) zero       undefined LAF coded as 0
  (c) common     only patients with a defined LAF in both scans in EVERY run
plus the share of empty maps by setting and by sign of the logit.
Decision rule: if ICC changes by < 0.02 and wSD by < 20% between (a)-(c), the explainer comparison is robust to the
choice; otherwise all three are reported.

Part 3, paired AUC against diameter: per architecture and run, AUC of the CNN logit minus AUC of the nodule diameter
on the LIDC validation split (all nodules, <= 10 mm, > 10 mm); the difference averaged over runs, with a patient-
cluster bootstrap CI (patients = LIDC patients; resample patients with all their nodules, same resample for all runs).
Decision rule: if the CI of a stratum includes 0, the CNN is not shown to add to diameter in that stratum.

Usage (from code/):  python 23_empty_laf_and_auc.py [--n-boot 2000]
Writes results/journal/review_empty_laf_auc/{empty_laf_icc.csv, empty_share.csv, auc_vs_diameter.csv}.
"""
import argparse

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from gauge_lib import PRIMARY_EXCLUDE, RESULTS_DIR, SECONDARY_SETTINGS

J = RESULTS_DIR / "journal"


def icc_wsd(a, b):
    """One-way ICC(1,1) and within-subject SD for k = 2 replicates (arrays of equal length)."""
    x = np.column_stack([a, b])
    n = len(x)
    gm = x.mean()
    msb = 2 * ((x.mean(1) - gm) ** 2).sum() / (n - 1)
    msw = ((x - x.mean(1, keepdims=True)) ** 2).sum() / n
    return float((msb - msw) / (msb + msw)), float(np.sqrt(msw))


def auc(y, s):
    y = np.asarray(y).astype(bool)
    r = rankdata(s)
    n1, n0 = y.sum(), (~y).sum()
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else np.nan


def part2(arch):
    m = pd.read_csv(J / f"saliency_{arch}" / "measurands.csv")
    m = m[~m.PatientID.isin(PRIMARY_EXCLUDE)]
    prim = m[~m.setting.isin(SECONDARY_SETTINGS)]
    share = (prim.assign(neg_logit=prim.logit < 0).groupby(["explainer", "setting", "neg_logit"]).empty_map.agg(["mean", "sum", "size"])
             .reset_index().assign(arch=arch))
    rows = []
    for (expl, setting), d in prim.groupby(["explainer", "setting"]):
        piv = d.set_index(["seed", "PatientID", "acq"]).laf.unstack("acq").reindex(columns=["scan1", "scan2"])
        present = d.groupby(["seed", "PatientID"]).acq.nunique()
        both = piv.loc[present[present == 2].index]          # patients with a cell in both scans
        defined = both.notna().all(axis=1)
        n_seeds = both.index.get_level_values("seed").nunique()
        cnt = both[defined].reset_index().groupby("PatientID").seed.nunique()
        common_pats = set(cnt[cnt == n_seeds].index)
        res = {"drop": [], "zero": [], "common": []}
        for seed, g in both.groupby(level="seed"):
            gd = g[g.notna().all(axis=1)]
            res["drop"].append(icc_wsd(gd.scan1, gd.scan2))
            res["zero"].append(icc_wsd(g.scan1.fillna(0), g.scan2.fillna(0)))
            gc = gd[gd.index.get_level_values("PatientID").isin(common_pats)]
            res["common"].append(icc_wsd(gc.scan1, gc.scan2))
        for k, v in res.items():
            v = np.array(v)
            rows.append(dict(arch=arch, explainer=expl, setting=setting, handling=k, icc_median=np.median(v[:, 0]), wsd_median=np.median(v[:, 1]),
                             n_pairs_median=int(np.median([len(g) for _, g in both.groupby(level="seed")])) if k == "zero" else np.nan,
                             n_empty_pairs=int((~defined).sum())))
    return pd.DataFrame(rows), share


def part3(arch, n_boot, rng):
    runs = sorted((J / f"runs_{arch}").glob("seed*/val_predictions.csv"))
    dfs = [pd.read_csv(p).assign(run=p.parent.name) for p in runs]
    d = pd.concat(dfs)
    strata = {"all": lambda x: np.ones(len(x), bool), "<=10 mm": lambda x: x.diameter_mm.values <= 10,
              ">10 mm": lambda x: x.diameter_mm.values > 10}

    def diff(frame):
        out = {}
        for name, f in strata.items():
            vals = []
            for _, g in frame.groupby("run"):
                s = g[f(g)]
                vals.append(auc(s.label, s.logit) - auc(s.label, s.diameter_mm))
            out[name] = float(np.nanmean(vals))
        return out

    est = diff(d)
    pats = d.PatientID.unique()
    by_p = {p: g for p, g in d.groupby("PatientID")}
    boots = []
    for _ in range(n_boot):
        pick = rng.choice(pats, len(pats), replace=True)
        fr = pd.concat([by_p[p].assign(PatientID=f"{p}#{i}") for i, p in enumerate(pick)], ignore_index=True)
        boots.append(diff(fr))
    b = pd.DataFrame(boots)
    rows = []
    for name in strata:
        lo, hi = np.nanpercentile(b[name], [2.5, 97.5])
        sub = d[strata[name](d)]
        rows.append(dict(arch=arch, stratum=name, auc_cnn=np.nanmean([auc(g[strata[name](g)].label, g[strata[name](g)].logit) for _, g in d.groupby("run")]),
                         auc_diameter=np.nanmean([auc(g[strata[name](g)].label, g[strata[name](g)].diameter_mm) for _, g in d.groupby("run")]),
                         diff=est[name], ci_lo=lo, ci_hi=hi, n_nodules=int(len(sub) / d.run.nunique()),
                         claim="CNN adds to diameter" if lo > 0 else ("CNN worse" if hi < 0 else "not shown to add")))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--archs", default="resnet,densenet")
    args = ap.parse_args()
    out = J / "review_empty_laf_auc"
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    icc, share, auc_rows = [], [], []
    for arch in args.archs.split(","):
        i, s = part2(arch)
        icc.append(i)
        share.append(s)
        auc_rows.append(part3(arch, args.n_boot, rng))
    icc, share, auc_rows = pd.concat(icc), pd.concat(share), pd.concat(auc_rows)
    icc.round(4).to_csv(out / "empty_laf_icc.csv", index=False)
    share.round(4).to_csv(out / "empty_share.csv", index=False)
    auc_rows.round(4).to_csv(out / "auc_vs_diameter.csv", index=False)
    print(icc.round(3).to_string(index=False))
    print(share[share["sum"] > 0].round(3).to_string(index=False))
    print(auc_rows.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
