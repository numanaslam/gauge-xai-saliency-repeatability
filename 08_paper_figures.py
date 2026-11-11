#!/usr/bin/env python3
"""
GAUGE-XAI step 8: paper figures and the summary table, from the CSVs written by 07 (no images needed,
so this runs anywhere the results/analysis folder is).

Writes (default ../results/analysis):
  fig_variance_sources.png  gauge R&R view: within-lesion SD of LAF (IG, Grad-CAM) and of the logit for each
                            source of variation - scan-rescan, slice thickness, kernel, training seed.
                            Bars = mean over settings / contrasts / seeds, whiskers = min-max.
  fig_icc_forest_seeds.png  scan-rescan ICC with 95% CI per reconstruction, one marker per seed
  table_main.csv / .txt     one row per explainer x measure: ICC by source (median over seeds, range),
                            RC as % of median, share of lesions whose kernel change exceeds the RC

Usage:
  python 08_paper_figures.py
  python 08_paper_figures.py --analysis /path/to/results/analysis
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from gauge_lib import RESULTS_DIR, SECONDARY_SETTINGS, SETTINGS  # noqa: E402

PRIMARY_SETTINGS = [s for s in SETTINGS if s not in SECONDARY_SETTINGS]
SOURCES = ["scan-rescan", "thickness", "kernel", "training seed"]
COLORS = dict(zip(SOURCES, ["#4C72B0", "#55A868", "#DD8452", "#C44E52"]))
NAMES = {"ig": "Integrated Gradients", "gradcam": "Grad-CAM"}


def load(a):
    rep = pd.read_csv(a / "repeatability.csv")
    rep = rep[(rep.population == "primary") & rep.setting.isin(PRIMARY_SETTINGS)]
    con = pd.read_csv(a / "conditions.csv")
    con = con[~con.contrast.str.contains("2.5S")]
    sd = pd.read_csv(a / "seeds.csv")
    sd = sd[sd.setting.isin(PRIMARY_SETTINGS)]
    return rep, con, sd


def source_table(rep, con, sd, expl, meas):
    """wSD and ICC for each source of variation: mean, min, max over settings / contrasts / seeds."""
    parts = {
        "scan-rescan": rep[(rep.explainer == expl) & (rep.measure == meas)],
        "thickness": con[(con.explainer == expl) & (con.measure == meas) & (con.kind == "thickness")],
        "kernel": con[(con.explainer == expl) & (con.measure == meas) & (con.kind == "kernel")],
        "training seed": sd[(sd.explainer == expl) & (sd.measure == meas)],
    }
    rows = []
    for src, g in parts.items():
        if g.empty:
            continue
        rows.append(dict(explainer=expl, measure=meas, source=src, n_estimates=len(g),
                         wSD=g.wSD.mean(), wSD_min=g.wSD.min(), wSD_max=g.wSD.max(),
                         icc_median=g.icc.median(), icc_min=g.icc.min(), icc_max=g.icc.max(),
                         median_value=g["median"].median(),
                         exceed_rc=g["exceed_rc"].mean() if "exceed_rc" in g else np.nan))
    return pd.DataFrame(rows)


def fig_sources(tab, out_png, fs=1.0):
    panels = [("ig", "laf", "IG: LAF"), ("gradcam", "laf", "Grad-CAM: LAF"), ("ig", "logit", "model logit")]
    fig, axes = plt.subplots(1, 3, figsize=(7.0 if fs == 1.0 else 7.6, 2.3 * (1 + 0.6 * (fs - 1))))  # drawn at print width (ISBI full width): fonts stay legible
    for ax, (e, ms, title) in zip(axes, panels):
        g = tab[(tab.explainer == e) & (tab.measure == ms)].set_index("source").reindex(SOURCES)
        x = np.arange(len(SOURCES))
        ax.bar(x, g.wSD, color=[COLORS[s] for s in SOURCES], width=0.65)
        ax.errorbar(x, g.wSD, yerr=[g.wSD - g.wSD_min, g.wSD_max - g.wSD], fmt="none", ecolor="k", capsize=3, lw=0.8)
        for xi, (top, icc) in enumerate(zip(g.wSD_max, g.icc_median)):
            if np.isfinite(top):
                ax.annotate(f"ICC\n{icc:.2f}", (xi, top), ha="center", va="bottom", fontsize=6.5 * fs,
                            xytext=(0, 3), textcoords="offset points")
        ax.set_xticks(x, ["scan-\nrescan", "thick-\nness", "kernel", "seed"] if fs == 1.0 else ["scan", "thick-\nness", "kernel", "run"], fontsize=7 * min(fs, 1.2))
        ax.tick_params(axis="y", labelsize=7 * fs)
        unit = "logit units" if ms == "logit" else "LAF units"
        ax.set_ylabel(f"within-lesion SD ({unit})", fontsize=7 * fs)
        med = g.median_value.median()
        ax.set_title(f"{title}" + ("" if ms == "logit" else f" (median {med:.3f})"), fontsize=8 * fs)
        ax.set_ylim(0, np.nanmax(g.wSD_max) * 1.35)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(pad=0.4, w_pad=1.2)  # no suptitle: the paper caption says what the bars are
    fig.savefig(out_png, dpi=400)
    plt.close(fig)


def fig_forest_seeds(rep_all, out_png):
    r = rep_all[(rep_all.population == "primary") & rep_all.measure.isin(["laf", "logit"])]
    panels = [("ig", "laf", "IG: LAF"), ("gradcam", "laf", "Grad-CAM: LAF"), ("ig", "logit", "model logit")]
    seeds = sorted(r.seed.unique())
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.4), sharey=True)
    for ax, (e, ms, title) in zip(axes, panels):
        for lo, hi, c in ((0, 0.5, "#f3d6d6"), (0.5, 0.75, "#f6e8c8"), (0.75, 0.9, "#e3eed6"), (0.9, 1, "#cfe6cf")):
            ax.axvspan(lo, hi, color=c, lw=0)
        for k, sd in enumerate(seeds):
            g = r[(r.explainer == e) & (r.measure == ms) & (r.seed == sd)].set_index("setting").reindex(SETTINGS)
            y = np.arange(len(SETTINGS))[::-1] + (k - (len(seeds) - 1) / 2) * 0.22
            ax.errorbar(g.icc, y, xerr=[g.icc - g.icc_ci_lo, g.icc_ci_hi - g.icc], fmt="o", ms=3, capsize=1.5,
                        lw=0.8, label=f"seed {sd}")
        ax.set_yticks(np.arange(len(SETTINGS))[::-1], [s + ("*" if s in SECONDARY_SETTINGS else "") for s in SETTINGS])
        ax.set_xlim(0.0, 1.0)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("scan-rescan ICC(1,1), 95% CI", fontsize=8)
    axes[-1].legend(fontsize=7, frameon=False, loc="lower left")
    fig.suptitle("Scan-rescan repeatability per reconstruction and seed (Koo & Li bands; * secondary)", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--analysis", default=RESULTS_DIR / "analysis", type=Path)
    ap.add_argument("--font-scale", type=float, default=1.0, help="journal version: 1.35 (larger labels, 'training run')")
    args = ap.parse_args()
    rep, con, sd = load(args.analysis)
    tab = pd.concat([source_table(rep, con, sd, e, ms) for e in ("ig", "gradcam") for ms in ("laf", "laf_sphere_patient", "logit")],
                    ignore_index=True)
    tab = tab[~((tab.explainer == "gradcam") & (tab.measure == "logit"))]  # same logit as for IG
    tab.round(4).to_csv(args.analysis / "table_main.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        txt = tab[["explainer", "measure", "source", "n_estimates", "icc_median", "icc_min", "icc_max", "wSD",
                   "wSD_min", "wSD_max", "median_value", "exceed_rc"]].round(3).to_string(index=False)
    (args.analysis / "table_main.txt").write_text(txt + "\n")
    print(txt)
    fig_sources(tab, args.analysis / "fig_variance_sources.png", args.font_scale)
    fig_forest_seeds(pd.read_csv(args.analysis / "repeatability.csv"), args.analysis / "fig_icc_forest_seeds.png")
    print(f"\nWrote {args.analysis}: table_main.csv/.txt, fig_variance_sources.png, fig_icc_forest_seeds.png")


if __name__ == "__main__":
    main()
