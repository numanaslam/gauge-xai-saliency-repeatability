#!/usr/bin/env python3
"""
GAUGE-XAI step 9: compact one-lesion figure for the paper (replaces the 2 x 6 figures of 07).

Rows: Integrated Gradients, Grad-CAM. Columns (default): scan 1 and scan 2 at 1.25 mm lung kernel, then
scan 1 and scan 2 at 1.25 mm standard kernel, so the eye reads repeatability (within a pair) and kernel
reproducibility (between pairs). Axial centre slice of the 128 mm patch, cropped to --crop-mm, radiological
orientation (patient right on image left). The attribution colour scale is shared within a row, so a change
in intensity between panels is real; positive attribution below 5% of the row maximum is hidden. Cyan: SEG.
Each panel is labelled with its LAF.

Needs the RIDER DICOM (CT is re-sampled for display) plus results/saliency (maps, masks, measurands.csv).
Writes ../results/analysis/fig_lesion_compact.png (and .pdf). Copy the .png to paper/figs/.

Usage:
  python 09_lesion_figure.py                                   # RIDER-1825099523, seed 0, column width
  python 09_lesion_figure.py --patient RIDER-1500037140 --seed 1 --width 7.0
  python 09_lesion_figure.py --cells scan1:1.25L scan2:1.25L scan1:5L scan2:5L
"""
import argparse
import importlib.util
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from gauge_lib import PATCH_SIZE, PATCH_SPACING_MM, RESULTS_DIR, extract_patch, load_ct  # noqa: E402

LUNG_WINDOW = (-1350, 150)
ROWS = (("ig", "IG"), ("gradcam", "Grad-CAM"))


def load_by_path(name, fname):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(fname))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def label(cell):
    acq, setting = cell
    kern = {"L": "lung", "S": "std"}[setting[-1]]
    return f"Scan {acq[-1]}\n{setting[:-1]} mm {kern}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--patient", default="RIDER-1825099523")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cells", nargs="+", default=["scan1:1.25L", "scan2:1.25L", "scan1:1.25S", "scan2:1.25S"],
                    help="columns as acq:setting")
    ap.add_argument("--crop-mm", type=float, default=96.0, help="field of view shown (the patch is 128 mm)")
    ap.add_argument("--width", type=float, default=3.4, help="figure width in inches (3.4 = one ISBI column)")
    ap.add_argument("--audit", default=RESULTS_DIR / "audit", type=Path)
    ap.add_argument("--saliency", default=RESULTS_DIR / "saliency", type=Path)
    ap.add_argument("--out", default=RESULTS_DIR / "analysis" / "fig_lesion_compact.png", type=Path)
    args = ap.parse_args()

    s7 = load_by_path("gauge_analysis", "07_gauge_analysis.py")  # load_map (Grad-CAM upsampling as in 06)
    cells = [tuple(c.split(":")) for c in args.cells]
    cond = pd.read_csv(args.audit / "condition_manifest.csv")
    cond = cond[(cond.PatientID == args.patient) & cond.usable & cond.seg_file.notna()].set_index(["acq", "setting"])
    meas = pd.read_csv(args.saliency / "measurands.csv")
    meas = meas[(meas.PatientID == args.patient) & (meas.seed == args.seed)]
    shape = (PATCH_SIZE,) * 3
    mid = PATCH_SIZE // 2
    h = int(round(args.crop_mm / PATCH_SPACING_MM / 2))
    sl = slice(mid - h, mid + h)

    panels = {}
    for cell in cells:
        name = f"{args.patient}_{cell[0]}_{cell[1]}"
        mk = args.saliency / "masks" / f"{name}.npz"
        if cell not in cond.index or not mk.exists():
            print(f"{name}: not available (unusable cell or not explained) - panel left empty")
            continue
        z = np.load(mk)
        r = cond.loc[cell]
        ct = load_ct(r["ct_folder"], r["ct_series_uid"])
        img = extract_patch(ct, z["centroid"])[mid, sl, sl]
        del ct
        maps = {}
        for expl, _ in ROWS:
            a = s7.load_map(args.saliency, expl, args.seed, name, shape)
            maps[expl] = None if a is None else np.clip(a[mid, sl, sl], 0, None)
        laf = {e: meas[(meas.explainer == e) & (meas.acq == cell[0]) & (meas.setting == cell[1])]["laf"] for e, _ in ROWS}
        panels[cell] = dict(img=img, seg=z["mask"].astype(bool)[mid, sl, sl], maps=maps,
                            laf={e: (float(v.iloc[0]) if len(v) else np.nan) for e, v in laf.items()})

    ncol = len(cells)
    pw = args.width / (ncol + 0.25)
    fig, axes = plt.subplots(2, ncol, figsize=(args.width, 2 * pw + 0.45), squeeze=False,
                             gridspec_kw=dict(wspace=0.03, hspace=0.04, left=0.25 / (ncol + 0.25), right=0.995,
                                              top=1 - 0.30 / (2 * pw + 0.45), bottom=0.01))
    for i, (expl, row_name) in enumerate(ROWS):
        vals = [p["maps"][expl] for p in panels.values() if p["maps"][expl] is not None]
        pos = np.concatenate([v[v > 0] for v in vals]) if vals else np.array([])
        vmax = float(np.percentile(pos, 99)) if pos.size else 1.0  # shared within the row
        for j, cell in enumerate(cells):
            ax = axes[i, j]
            ax.set_xticks([]), ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_linewidth(0.4)
            if i == 0:
                ax.set_title(label(cell), fontsize=6, pad=2)
            if j == 0:
                ax.set_ylabel(row_name, fontsize=7, labelpad=2)
            p = panels.get(cell)
            if p is None or p["maps"][expl] is None:
                ax.imshow(np.full((2 * h, 2 * h), 0.85), cmap="gray", vmin=0, vmax=1)  # same size as image panels
                ax.text(0.5, 0.5, "n/a", transform=ax.transAxes, fontsize=6, ha="center", va="center", color="0.4")
                continue
            a = p["maps"][expl]
            ax.imshow(p["img"], cmap="gray", vmin=LUNG_WINDOW[0], vmax=LUNG_WINDOW[1], interpolation="nearest")
            ax.imshow(np.ma.masked_less(a, 0.05 * vmax), cmap="inferno", alpha=0.65, vmin=0, vmax=vmax,
                      interpolation="nearest")
            if p["seg"].any():
                ax.contour(p["seg"], levels=[0.5], colors="c", linewidths=0.5)
            laf = p["laf"][expl]
            txt = f"LAF {laf:.2f}" if np.isfinite(laf) else "empty map"
            ax.text(0.03, 0.04, txt, transform=ax.transAxes, fontsize=5.5, color="w",
                    va="bottom", ha="left", bbox=dict(boxstyle="round,pad=0.15", fc="k", ec="none", alpha=0.6))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=400)
    fig.savefig(args.out.with_suffix(".pdf"))
    plt.close(fig)
    print(f"Wrote {args.out} and {args.out.with_suffix('.pdf')} ({args.patient}, seed {args.seed}, "
          f"{args.crop_mm:g} mm shown). Copy the .png to paper/figs/.")


if __name__ == "__main__":
    main()
