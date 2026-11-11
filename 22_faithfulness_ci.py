#!/usr/bin/env python3
"""
GAUGE-XAI step 22 (Diagnostics review round, experiment 1): patient-cluster bootstrap CIs for the faithfulness scores.

Reads <analysis>/<variant>/faithfulness.csv (one row per cell x run x explainer; written by 14_faithfulness.py) for
the deletion variants and the insertion test, restricted to the primary population (RIDER exclusions, no 2.5S) as in
the manuscript. For every variant x explainer it reports
  median_score   median over runs of the per-run median deletion/insertion score   (as in the manuscript tables)
  share_pos      median over runs of the share of maps with score > 0
each with a 95% CI from a patient-cluster bootstrap (patients resampled with all their cells and runs).
Paired contrasts (same patient x scan x reconstruction x run): blurred minus mean replacement for each explainer, and
IG (air) minus every other explainer, within one variant.
Decision rule: a sign claim ("positive", "negative") is made only if the whole CI is on that side of 0.

Usage (from code/):  python 22_faithfulness_ci.py [--n-boot 2000] [--archs resnet,densenet]
Writes results/journal/review_faithfulness_ci/{faithfulness_ci.csv, paired_contrasts.csv}.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from gauge_lib import PRIMARY_EXCLUDE, RESULTS_DIR, SECONDARY_SETTINGS

VARIANTS = ["faithfulness", "faithfulness_blur_matched", "faithfulness_mean_matched", "faithfulness_mean_voxel",
            "faithfulness_mean_cube", "faithfulness_mean_lesion", "insertion_mean"]
KEYS = ["PatientID", "acq", "setting", "seed"]


def load(adir, v):
    p = adir / v / "faithfulness.csv"
    if not p.exists():
        return None
    d = pd.read_csv(p)
    d = d[~d.PatientID.isin(PRIMARY_EXCLUDE) & ~d.setting.isin(SECONDARY_SETTINGS)]
    return d[["explainer", *KEYS, "deletion_score"]]


def stats_from_groups(groups_by_patient, pick, n_seeds):
    """groups_by_patient: {patient: (scores array per row, seed index per row)}; pick: list of patients (with repeats)."""
    per_seed = [[] for _ in range(n_seeds)]
    for p in pick:
        s, k = groups_by_patient[p]
        for i in range(n_seeds):
            per_seed[i].append(s[k == i])
    meds, pos = [], []
    for i in range(n_seeds):
        v = np.concatenate(per_seed[i]) if per_seed[i] else np.array([])
        if len(v):
            meds.append(np.median(v))
            pos.append(np.mean(v > 0))
    return float(np.median(meds)), float(np.median(pos))


def cluster(d):
    seeds = sorted(d.seed.unique())
    idx = {s: i for i, s in enumerate(seeds)}
    out = {}
    for p, g in d.groupby("PatientID"):
        out[p] = (g.deletion_score.to_numpy(), g.seed.map(idx).to_numpy())
    return out, len(seeds)


def boot_ci(d, n_boot, rng):
    groups, ns = cluster(d)
    pats = list(groups)
    est = stats_from_groups(groups, pats, ns)
    bs = np.array([stats_from_groups(groups, list(rng.choice(pats, len(pats), replace=True)), ns) for _ in range(n_boot)])
    lo, hi = np.percentile(bs, [2.5, 97.5], axis=0)
    return est, lo, hi


def paired(a, b, n_boot, rng):
    m = a.merge(b, on=["explainer", *KEYS], suffixes=("_a", "_b"))
    m["deletion_score"] = m.deletion_score_a - m.deletion_score_b
    groups, ns = cluster(m)
    pats = list(groups)
    est = stats_from_groups(groups, pats, ns)[0]
    bs = np.array([stats_from_groups(groups, list(rng.choice(pats, len(pats), replace=True)), ns)[0] for _ in range(n_boot)])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return est, lo, hi, len(m)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--archs", default="resnet,densenet")
    args = ap.parse_args()
    out = RESULTS_DIR / "journal" / "review_faithfulness_ci"
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    rows, prow = [], []
    for arch in args.archs.split(","):
        adir = RESULTS_DIR / "journal" / f"analysis_{arch}"
        data = {v: load(adir, v) for v in VARIANTS}
        for v, d in data.items():
            if d is None:
                continue
            for e, de in d.groupby("explainer"):
                (m, sp), lo, hi = boot_ci(de, args.n_boot, rng)
                rows.append(dict(arch=arch, variant=v, explainer=e, median_score=m, ci_lo=lo[0], ci_hi=hi[0],
                                 share_pos=sp, share_lo=lo[1], share_hi=hi[1],
                                 claim="positive" if lo[0] > 0 else ("negative" if hi[0] < 0 else "not distinguishable from random")))
                print(rows[-1], flush=True)
        # paired: blurred minus mean replacement (matched random), per explainer
        if data["faithfulness_blur_matched"] is not None and data["faithfulness_mean_matched"] is not None:
            for e in sorted(data["faithfulness_blur_matched"].explainer.unique()):
                a = data["faithfulness_blur_matched"].query("explainer == @e")
                b = data["faithfulness_mean_matched"].query("explainer == @e")
                est, lo, hi, n = paired(a, b, args.n_boot, rng)
                prow.append(dict(arch=arch, contrast=f"{e}: blurred minus mean replacement", est=est, ci_lo=lo, ci_hi=hi, n=n))
        # paired: IG(air) minus the other explainers within each of two variants
        for v in ("faithfulness_blur_matched", "faithfulness_mean_matched", "insertion_mean"):
            d = data[v]
            if d is None:
                continue
            ref = d.query("explainer == 'ig'")
            for e in sorted(set(d.explainer) - {"ig"}):
                other = d.query("explainer == @e").assign(explainer="ig")  # align on explainer key
                est, lo, hi, n = paired(ref, other, args.n_boot, rng)
                prow.append(dict(arch=arch, contrast=f"{v}: IG (air) minus {e}", est=est, ci_lo=lo, ci_hi=hi, n=n))
    pd.DataFrame(rows).round(4).to_csv(out / "faithfulness_ci.csv", index=False)
    pc = pd.DataFrame(prow).round(4)
    pc["claim"] = np.where(pc.ci_lo > 0, "first larger", np.where(pc.ci_hi < 0, "second larger", "not distinguishable"))
    pc.to_csv(out / "paired_contrasts.csv", index=False)
    print(pc.to_string(index=False))


if __name__ == "__main__":
    main()
