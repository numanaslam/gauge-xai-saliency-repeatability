#!/usr/bin/env python3
"""
GAUGE-XAI step 16 (journal): faithfulness figure from the outputs of 14_faithfulness.py for both architectures.

fig_deletion.png: median logit drop after deleting the top-k% ROI voxels vs. random voxels, ResNet, with the blurred
(a) and the patch-mean (b) replacement. fig_randomisation.png: IG and Grad-CAM under cascading model randomisation
(map Spearman with the trained model; IG LAF relative to the trained LAF), ResNet and DenseNet.

Usage:
  python 16_journal_figures.py --journal ../results/journal --out ../paper_cmc/figs/fig_faithfulness.png
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from gauge_lib import PRIMARY_EXCLUDE, RESULTS_DIR  # noqa: E402

NAMES = {"ig": "IG (air)", "ig_blur": "IG (blur)", "ig_mean": "IG (mean)", "gradcam": "Grad-CAM", "occlusion": "Occlusion"}
COLORS = {"ig": "#1f77b4", "ig_blur": "#6baed6", "ig_mean": "#9ecae1", "gradcam": "#d62728", "occlusion": "#2ca02c"}
LEVELS = ["head", "head+layer4", "all"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--journal", default=RESULTS_DIR / "journal", type=Path)
    ap.add_argument("--out", default=Path(__file__).resolve().parent.parent / "paper_cmc" / "figs" / "fig_faithfulness.png", type=Path)
    args = ap.parse_args()

    plt.rcParams.update({"font.size": 10})
    out_del = args.out.with_name("fig_deletion.png")
    out_ran = args.out.with_name("fig_randomisation.png")
    # Figure A: deletion curves with the blurred and with the patch-mean replacement
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4), sharey=False)
    for ax, (sub, title) in zip(axes, (("faithfulness", "(a) blurred replacement"), ("faithfulness_mean_voxel", "(b) patch-mean replacement"))):
        f = pd.read_csv(args.journal / "analysis_resnet" / sub / "faithfulness.csv")
        f = f[~f.PatientID.isin(PRIMARY_EXCLUDE) & f.explainer.isin(NAMES)]
        ks = sorted(float(c.replace("drop_top_", "")) for c in f.columns if c.startswith("drop_top_"))
        x = [100 * k for k in ks]
        for e, g in f.groupby("explainer"):
            ax.plot(x, [g[f"drop_top_{k:g}"].median() for k in ks], "-o", ms=3, color=COLORS[e], label=NAMES[e])
        ax.plot(x, [f[f"drop_rand_{k:g}"].median() for k in ks], "--", color="0.4", label="random voxels")
        ax.axhline(0, color="0.8", lw=0.8)
        ax.set_xscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{v:g}" for v in x])
        ax.set_xlabel("deleted ROI voxels (%)")
        ax.set_title(title, fontsize=10.5, loc="left")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("logit drop (median, ResNet)")
    axes[1].legend(fontsize=8.5, frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(out_del, dpi=600)
    plt.close(fig)
    # Figure B: cascading randomisation
    fig, a2 = plt.subplots(figsize=(5.2, 3.4))
    width = 0.2
    pos = np.arange(len(LEVELS))
    for i, (arch, e) in enumerate([("resnet", "ig"), ("densenet", "ig"), ("resnet", "gradcam"), ("densenet", "gradcam")]):
        s = pd.read_csv(args.journal / f"analysis_{arch}" / "faithfulness" / "sanity_summary.csv")
        s = s[s.explainer == e].set_index("randomised").reindex(LEVELS)
        xx = pos + (i - 1.5) * width
        a2.bar(xx, s.spearman_median, width, color=COLORS[e], alpha=1.0 if arch == "resnet" else 0.55,
               label=f"{NAMES[e]} {'ResNet' if arch == 'resnet' else 'DenseNet'}: Spearman")
        if e == "ig":
            a2.plot(xx, s.laf_randomised / s.laf_trained, "k^", ms=5, label="IG: LAF / trained LAF" if i == 0 else None)
    a2.axhline(0, color="0.8", lw=0.8)
    a2.set_xticks(pos)
    a2.set_xticklabels(["head", "head + layer4", "whole network"])
    a2.set_xlabel("re-initialised weights (cascade)")
    a2.set_ylim(-0.1, 1.05)
    a2.set_ylabel("value relative to trained model")
    a2.legend(fontsize=8, frameon=False, loc="upper right")
    a2.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_ran, dpi=600)
    args.out = f"{out_del} and {out_ran}"
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
