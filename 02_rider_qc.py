#!/usr/bin/env python3
"""
GAUGE-XAI step 2: load every CT + SEG pair from the audit, check alignment,
compute lesion statistics and draw QC montages.

Writes:
  <out>/lesion_stats.csv      one row per condition: volume, diameters, centroid, mean HU
  <out>/qc/<PatientID>.png    2 x 6 montage (scan x reconstruction), axial slice through
                              the lesion centroid with the SEG contour
  <out>/qc_problems.txt       pairs that failed, misaligned SEGs, out-of-LIDC-range lesions

Usage:
  python 02_rider_qc.py --audit audit --out audit
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pydicom  # noqa: E402

from gauge_lib import RESULTS_DIR  # noqa: E402

SETTINGS = ["1.25L", "1.25S", "2.5L", "2.5S", "5L", "5S"]
ACQS = ["scan1", "scan2"]
LUNG_WINDOW = (-1350, 150)       # level -600, width 1500
CROP_MM = 48                     # half-width of the montage crop
LIDC_MAX_MM = 30                 # LIDC nodule annotations cover ~3-30 mm
CENTROID_TOL_MM = 3.0            # same acquisition, different recon: centroids should agree
CENTROID_OUTLIER_MM = 20.0       # further than this from the acquisition median: not the same lesion
MIN_SEG_FRAMES = 2               # single-frame RIDER SEGs outline other organs


def load_ct(folder, series_uid):
    slices = []
    for p in Path(folder).iterdir():
        if p.name.startswith(".") or not p.is_file():
            continue
        try:
            ds = pydicom.dcmread(p)
        except Exception:
            continue
        if getattr(ds, "SeriesInstanceUID", None) == series_uid and hasattr(ds, "ImagePositionPatient"):
            slices.append(ds)
    if not slices:
        raise RuntimeError("no CT slices found")
    iop = np.array(slices[0].ImageOrientationPatient, float)
    row_dir, col_dir = iop[:3], iop[3:]
    normal = np.cross(row_dir, col_dir)
    slices.sort(key=lambda d: float(np.dot(normal, np.array(d.ImagePositionPatient, float))))
    vol = np.stack([
        d.pixel_array.astype(np.float32) * float(getattr(d, "RescaleSlope", 1))
        + float(getattr(d, "RescaleIntercept", 0))
        for d in slices
    ])
    ipps = np.array([d.ImagePositionPatient for d in slices], float)
    zs = ipps @ normal
    dz = float(np.median(np.diff(zs))) if len(zs) > 1 else float(slices[0].SliceThickness)
    ps = np.array(slices[0].PixelSpacing, float)  # [row spacing (mm per row), column spacing]
    return dict(vol=vol, ipps=ipps, zs=zs, dz=abs(dz), ps=ps, row_dir=row_dir,
                col_dir=col_dir, normal=normal, iop=iop)


def paste(dst, src, r0, c0):
    H, W = dst.shape
    h, w = src.shape
    rs, cs, re_, ce = max(r0, 0), max(c0, 0), min(r0 + h, H), min(c0 + w, W)
    if re_ > rs and ce > cs:
        dst[rs:re_, cs:ce] |= src[rs - r0:re_ - r0, cs - c0:ce - c0]


def load_seg_mask(seg_file, ct):
    """Decode a DICOM SEG onto the CT grid by matching frame positions (no extra libraries)."""
    ds = pydicom.dcmread(seg_file)
    frames = ds.pixel_array
    if frames.ndim == 2:
        frames = frames[None]
    if str(getattr(ds, "SegmentationType", "BINARY")).upper() == "FRACTIONAL":
        frames = frames.astype(np.float32) / float(getattr(ds, "MaximumFractionalValue", 255)) >= 0.5
    else:
        frames = frames > 0

    shared = ds.SharedFunctionalGroupsSequence[0] if "SharedFunctionalGroupsSequence" in ds else None

    def attr(fg, seq, kw):
        for src in (fg, shared):
            if src is not None and seq in src:
                return getattr(getattr(src, seq)[0], kw, None)
        return None

    mask = np.zeros(ct["vol"].shape, bool)
    seg_numbers, unplaced, worst_subpixel = set(), 0, 0.0
    tol = 0.5 * ct["dz"]
    for i, fg in enumerate(ds.PerFrameFunctionalGroupsSequence):
        f_iop = attr(fg, "PlaneOrientationSequence", "ImageOrientationPatient")
        if f_iop is not None and not np.allclose(np.array(f_iop, float), ct["iop"], atol=1e-3):
            raise RuntimeError("SEG orientation differs from CT")
        f_ps = attr(fg, "PixelMeasuresSequence", "PixelSpacing")
        if f_ps is not None and not np.allclose(np.array(f_ps, float), ct["ps"], atol=1e-3):
            raise RuntimeError(f"SEG pixel spacing {list(f_ps)} differs from CT {list(ct['ps'])}")
        if "SegmentIdentificationSequence" in fg:
            seg_numbers.add(int(fg.SegmentIdentificationSequence[0].ReferencedSegmentNumber))
        ipp = np.array(fg.PlanePositionSequence[0].ImagePositionPatient, float)
        k = int(np.argmin(np.abs(ct["zs"] - ipp @ ct["normal"])))
        if abs(ct["zs"][k] - ipp @ ct["normal"]) > tol:
            unplaced += 1
            continue
        d = ipp - ct["ipps"][k]
        dc = d @ ct["row_dir"] / ct["ps"][1]   # column offset
        dr = d @ ct["col_dir"] / ct["ps"][0]   # row offset
        oc, orr = int(round(dc)), int(round(dr))
        worst_subpixel = max(worst_subpixel, abs(dc - oc), abs(dr - orr))
        paste(mask[k], frames[i], orr, oc)
    return mask, dict(n_frames=len(frames), frames_unplaced=unplaced,
                      seg_numbers=len(seg_numbers), worst_subpixel_offset=round(worst_subpixel, 3))


def longest_axial_diameter(slice_mask, ps):
    m = slice_mask
    inner = m.copy()
    inner[1:, :] &= m[:-1, :]
    inner[:-1, :] &= m[1:, :]
    inner[:, 1:] &= m[:, :-1]
    inner[:, :-1] &= m[:, 1:]
    pts = np.argwhere(m & ~inner) * ps  # boundary points in mm (row, col)
    if len(pts) < 2:
        return 0.0
    if len(pts) > 1500:
        pts = pts[np.random.default_rng(0).choice(len(pts), 1500, replace=False)]
    d = np.sqrt(((pts[:, None, :] - pts[None, :, :]) ** 2).sum(-1))
    return float(d.max())


def lesion_stats(mask, ct):
    n = int(mask.sum())
    if n == 0:
        return dict(voxels=0), None
    dy, dx = ct["ps"]
    kk, rr, cc = np.nonzero(mask)
    ck, cr, ccol = kk.mean(), rr.mean(), cc.mean()
    k0 = int(round(ck))
    centroid = (ct["ipps"][k0] + (ck - k0) * ct["dz"] * ct["normal"]
                + ccol * dx * ct["row_dir"] + cr * dy * ct["col_dir"])
    vol_ml = n * dx * dy * ct["dz"] / 1000.0
    kmax = int(np.argmax(mask.sum(axis=(1, 2))))
    stats = dict(
        voxels=n,
        volume_ml=vol_ml,
        eq_sphere_diam_mm=(6 * vol_ml * 1000 / np.pi) ** (1 / 3),
        longest_axial_mm=longest_axial_diameter(mask[kmax], ct["ps"]),
        z_extent_mm=(kk.max() - kk.min() + 1) * ct["dz"],
        hu_mean=float(ct["vol"][mask].mean()),
        hu_sd=float(ct["vol"][mask].std()),
        centroid_x=centroid[0], centroid_y=centroid[1], centroid_z=centroid[2],
    )
    half_r, half_c = int(CROP_MM / dy), int(CROP_MM / dx)
    r0, c0 = int(round(cr)), int(round(ccol))
    sl = (slice(max(r0 - half_r, 0), r0 + half_r), slice(max(c0 - half_c, 0), c0 + half_c))
    crop = (ct["vol"][k0][sl].copy(), mask[k0][sl].copy())
    return stats, crop


def draw_montage(pid, crops, out_png):
    fig, axes = plt.subplots(2, 6, figsize=(15, 5.6))
    for i, acq in enumerate(ACQS):
        for j, s in enumerate(SETTINGS):
            ax = axes[i, j]
            ax.set_xticks([]), ax.set_yticks([])
            item = crops.get((acq, s))
            if item is None:
                ax.set_facecolor("0.85")
                ax.set_title(f"{acq} {s}\nMISSING", fontsize=9, color="red")
                continue
            (img, m), vol = item
            ax.imshow(img, cmap="gray", vmin=LUNG_WINDOW[0], vmax=LUNG_WINDOW[1])
            if m.any():
                ax.contour(m, levels=[0.5], colors="r", linewidths=0.8)
            ax.set_title(f"{acq} {s}\n{vol:.1f} mL", fontsize=9)
    fig.suptitle(pid)
    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--audit", default=RESULTS_DIR / "audit", type=Path, help="folder with condition_manifest.csv")
    ap.add_argument("--out", default=RESULTS_DIR / "audit", type=Path)
    ap.add_argument("--patients", nargs="*", help="only these PatientIDs (for a quick first run)")
    args = ap.parse_args()
    (args.out / "qc").mkdir(parents=True, exist_ok=True)

    cond = pd.read_csv(args.audit / "condition_manifest.csv")
    cond = cond[cond["usable"] & cond["seg_file"].notna()]
    if args.patients:
        cond = cond[cond["PatientID"].isin(args.patients)]

    rows, problems = [], []
    for pid, g in cond.groupby("PatientID"):
        crops = {}
        print(f"{pid}: {len(g)} conditions")
        for _, r in g.iterrows():
            tag = f"{pid} {r['acq']} {r['setting']}"
            try:
                ct = load_ct(r["ct_folder"], r["ct_series_uid"])
                mask, info = load_seg_mask(r["seg_file"], ct)
                stats, crop = lesion_stats(mask, ct)
            except Exception as e:  # noqa: BLE001
                problems.append(f"{tag}: FAILED {type(e).__name__}: {e}")
                continue
            if stats["voxels"] == 0:
                problems.append(f"{tag}: SEG mask is empty after placement on CT grid")
            if info["frames_unplaced"]:
                problems.append(f"{tag}: {info['frames_unplaced']}/{info['n_frames']} SEG frames did not match a CT slice")
            if info["worst_subpixel_offset"] > 0.1:
                problems.append(f"{tag}: SEG in-plane offset is not a whole pixel ({info['worst_subpixel_offset']} px)")
            if info["seg_numbers"] > 1:
                problems.append(f"{tag}: {info['seg_numbers']} segments merged into one mask")
            rows.append(dict(PatientID=pid, acq=r["acq"], setting=r["setting"], **stats, **info))
            if crop is not None:
                crops[(r["acq"], r["setting"])] = (crop, stats["volume_ml"])
            del ct, mask
        draw_montage(pid, crops, args.out / "qc" / f"{pid}.png")

    df = pd.DataFrame(rows)

    # Reject SEGs that cannot be the tumour: RIDER v3 has single-frame SEGs that outline the liver or
    # chest wall, far from the contours of the other reconstructions of the same acquisition.
    if len(df) and "centroid_x" in df:
        xyz = ["centroid_x", "centroid_y", "centroid_z"]
        df["qc_ok"] = (df["voxels"] > 0) & (df["n_frames"] >= MIN_SEG_FRAMES)
        med = df[df["qc_ok"]].groupby(["PatientID", "acq"])[xyz].median()
        ref = med.reindex(pd.MultiIndex.from_frame(df[["PatientID", "acq"]])).to_numpy()
        df["centroid_dist_mm"] = np.sqrt(((df[xyz].to_numpy() - ref) ** 2).sum(1))
        df.loc[df["centroid_dist_mm"] > CENTROID_OUTLIER_MM, "qc_ok"] = False
        rejected = df[~df["qc_ok"]]
        problems.append(f"Rejected cells (qc_ok=False; < {MIN_SEG_FRAMES} SEG frames, empty, or centroid "
                        f"> {CENTROID_OUTLIER_MM:g} mm from the acquisition median): {len(rejected)}")
        problems += [f"  {r.PatientID} {r.acq} {r.setting}: {r.n_frames} frame(s), "
                     f"{r.centroid_dist_mm:.0f} mm from median" for r in rejected.itertuples()]
        problems.append("")
    df.to_csv(args.out / "lesion_stats.csv", index=False)
    if "qc_ok" in df:
        df = df[df["qc_ok"]]

    # Same acquisition, different reconstruction: the lesion should sit in the same place.
    if len(df) and "centroid_x" in df:
        for (pid, acq), g in df.dropna(subset=["centroid_x"]).groupby(["PatientID", "acq"]):
            c = g[["centroid_x", "centroid_y", "centroid_z"]].to_numpy()
            spread = float(np.sqrt(((c - c.mean(0)) ** 2).sum(1)).max())
            if spread > CENTROID_TOL_MM:
                problems.append(f"{pid} {acq}: lesion centroid moves {spread:.1f} mm across reconstructions "
                                f"(possible SEG/CT mismatch)")
        big = df[df["longest_axial_mm"] > LIDC_MAX_MM].groupby("PatientID")["longest_axial_mm"].max()
        problems.append("")
        problems.append(f"Lesions larger than the LIDC annotation range (> {LIDC_MAX_MM} mm longest axial): "
                        f"{len(big)} / {df['PatientID'].nunique()} patients")
        problems += [f"  {p}: up to {d:.0f} mm" for p, d in big.items()]
        problems.append("")
        problems.append("Longest axial diameter (mm), 1.25L scan1: "
                        + df[(df.setting == "1.25L") & (df.acq == "scan1")]["longest_axial_mm"]
                        .describe().round(1).to_string().replace("\n", " | "))

    (args.out / "qc_problems.txt").write_text("\n".join(problems) + "\n")
    print("\n".join(problems))
    print(f"\nWrote {args.out}/lesion_stats.csv, qc_problems.txt and qc/*.png - look at every montage.")


if __name__ == "__main__":
    main()
