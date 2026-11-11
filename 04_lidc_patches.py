#!/usr/bin/env python3
"""
GAUGE-XAI step 4: LIDC-IDRI nodules -> 1 mm isotropic patches for training.

Parses the LIDC XML reads, places every reader's nodule outline on its CT, groups the up-to-four
readers' marks into physical nodules, and samples one patch per nodule around the consensus centroid.
No label rule is applied here (05_train.py does that), so every nodule with a malignancy rating is kept.

Writes (in --out):
  nodules.csv          one row per nodule: patient, series, centroid (mm), size, reader ratings
  reader_marks.csv     one row per reader mark (before grouping), for auditing the grouping
  patches.npy          uint8 (N, 144, 144, 144), HU clipped to [-1000, 400] and scaled to 0-255 (1 mm);
                       row i matches nodules.csv patch_index i (144 = 128 + room for shift augmentation)
                       (~3 MB per nodule; written incrementally, so RAM use stays small)
  patches_qc.png       centre slices of a random sample of patches, with the consensus diameter
  problems.txt         series without XML, marks that could not be placed, failed series

Usage:
  python 04_lidc_patches.py --series audit/lidc_series.csv --xml-root /data/LIDC-XML-only --out lidc_patches
  (--xml-root may also be the CT download root if the XML files sit next to the DICOM)
"""
import argparse
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

from gauge_lib import (RESULTS_DIR, PATCH_SPACING_MM, STORE_SIZE, extract_patch, from_uint8, load_ct, normalize_hu,
                       pixel_to_world, to_uint8)

GROUP_MIN_MM = 4.0      # two readers' marks are the same nodule if centroids are closer than
GROUP_FRAC = 0.5        # max(GROUP_MIN_MM, GROUP_FRAC * larger diameter)
CHARACTERISTICS = ["subtlety", "internalStructure", "calcification", "sphericity", "margin",
                   "lobulation", "spiculation", "texture", "malignancy"]


# ---- XML ---------------------------------------------------------------------------------------
def local(tag):
    return tag.rsplit("}", 1)[-1]


def child(el, name):
    for e in el:
        if local(e.tag) == name:
            return e
    return None


def children(el, name):
    return [e for e in el if local(e.tag) == name]


def text(el, name):
    c = child(el, name) if el is not None else None
    return c.text.strip() if c is not None and c.text else None


def series_uid_of_xml(path):
    """Cheap header read: SeriesInstanceUid of a CT read (None for chest X-ray reads)."""
    try:
        for _, el in ET.iterparse(path, events=("end",)):
            if local(el.tag).lower() == "seriesinstanceuid":
                return el.text.strip() if el.text else None
    except ET.ParseError:
        return None
    return None


def parse_reads(path):
    """All reader nodule marks (>= 3 mm nodules, which carry characteristics) in one LIDC XML."""
    root = ET.parse(path).getroot()
    marks = []
    for s, session in enumerate(children(root, "readingSession")):
        reader = text(session, "servicingRadiologistID") or f"session{s}"
        for nod in children(session, "unblindedReadNodule"):
            chars = child(nod, "characteristics")
            if chars is None or text(chars, "malignancy") is None:
                continue  # < 3 mm nodule: no characteristics, no outline
            ratings = {}
            for c in CHARACTERISTICS:
                v = text(chars, c)
                ratings[c] = int(v) if v and v.isdigit() else None
            if not ratings["malignancy"]:
                continue
            rois = []
            for roi in children(nod, "roi"):
                pts = [(float(text(e, "xCoord")), float(text(e, "yCoord"))) for e in children(roi, "edgeMap")]
                rois.append(dict(sop=text(roi, "imageSOP_UID"), z=float(text(roi, "imageZposition") or "nan"),
                                 inclusion=(text(roi, "inclusion") or "TRUE").upper() == "TRUE",
                                 pts=np.array(pts, float)))
            marks.append(dict(session=s, reader=reader, nodule_id=text(nod, "noduleID"), rois=rois, **ratings))
    return marks


# ---- geometry ------------------------------------------------------------------------------------
def fill_polygon(pts, shape):
    """Pixels inside (or on) a closed outline given as (x=column, y=row) points."""
    from matplotlib.path import Path as MplPath

    c0, r0 = np.floor(pts.min(0)).astype(int)
    c1, r1 = np.ceil(pts.max(0)).astype(int)
    c0, r0 = max(c0, 0), max(r0, 0)
    c1, r1 = min(c1, shape[1] - 1), min(r1, shape[0] - 1)
    cc, rr = np.meshgrid(np.arange(c0, c1 + 1), np.arange(r0, r1 + 1))
    q = np.stack([cc.ravel(), rr.ravel()], 1)
    inside = MplPath(pts).contains_points(q, radius=0.5) if len(pts) >= 3 else np.zeros(len(q), bool)
    rc = np.concatenate([q[inside][:, ::-1], np.rint(pts[:, ::-1]).astype(int)])  # + boundary
    return np.unique(rc, axis=0)


def place_mark(mark, ct, sop_to_k):
    """Voxels of one reader's mark on the CT grid -> centroid (mm), volume, longest axial diameter."""
    vox, n_unplaced, longest = [], 0, 0.0
    for roi in mark["rois"]:
        if not roi["inclusion"] or len(roi["pts"]) == 0:
            continue
        k = sop_to_k.get(roi["sop"])
        if k is None and np.isfinite(roi["z"]):
            j = int(np.argmin(np.abs(ct["ipps"][:, 2] - roi["z"])))
            k = j if abs(ct["ipps"][j, 2] - roi["z"]) < 0.5 * ct["dz"] + 1e-3 else None
        if k is None:
            n_unplaced += 1
            continue
        rc = fill_polygon(roi["pts"], ct["vol"].shape[1:])
        vox.append(np.column_stack([np.full(len(rc), k), rc]))
        e = roi["pts"] * ct["ps"][::-1]  # (x, y) pixels -> mm
        if len(e) > 1:
            if len(e) > 600:
                e = e[np.random.default_rng(0).choice(len(e), 600, replace=False)]
            longest = max(longest, float(np.sqrt(((e[:, None] - e[None]) ** 2).sum(-1)).max()))
    if not vox:
        return None, n_unplaced
    v = np.concatenate(vox)
    world = pixel_to_world(ct, v[:, 0], v[:, 1], v[:, 2])
    vol_ml = len(v) * ct["ps"][0] * ct["ps"][1] * ct["dz"] / 1000.0
    return dict(centroid=world.mean(0), volume_ml=vol_ml, longest_axial_mm=longest,
                eq_sphere_diam_mm=(6 * vol_ml * 1000 / np.pi) ** (1 / 3), n_slices=len(vox)), n_unplaced


def group_marks(marks):
    """Union-find over marks from different readers whose centroids are close -> list of index groups."""
    parent = list(range(len(marks)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(marks)):
        for j in range(i + 1, len(marks)):
            if marks[i]["session"] == marks[j]["session"]:
                continue
            d = np.linalg.norm(marks[i]["centroid"] - marks[j]["centroid"])
            size = max(marks[i]["eq_sphere_diam_mm"], marks[j]["eq_sphere_diam_mm"])
            if d < max(GROUP_MIN_MM, GROUP_FRAC * size):
                parent[find(i)] = find(j)
    groups = defaultdict(list)
    for i in range(len(marks)):
        groups[find(i)].append(i)
    return list(groups.values())


# ---- per series ------------------------------------------------------------------------------------
def process_series(row, xml_paths):
    problems, marks = [], []
    try:
        ct = load_ct(row["folder"], row["SeriesInstanceUID"])
    except Exception as e:  # noqa: BLE001
        return [], [], None, [f"{row['PatientID']} {row['SeriesInstanceUID']}: CT load failed: {e}"]
    sop_to_k = {u: i for i, u in enumerate(ct["sop_uids"])}
    for xp in xml_paths:
        try:
            raw = parse_reads(xp)
        except Exception as e:  # noqa: BLE001
            problems.append(f"{xp}: XML parse failed: {e}")
            continue
        for m in raw:
            geo, n_unplaced = place_mark(m, ct, sop_to_k)
            if n_unplaced:
                problems.append(f"{row['PatientID']} reader {m['session']} nodule {m['nodule_id']}: "
                                f"{n_unplaced} outline slice(s) not matched to a CT slice")
            if geo is None:
                continue
            m = {k: v for k, v in m.items() if k != "rois"}
            m.update(geo, xml=os.path.basename(xp))
            marks.append(m)

    nodules, patches = [], []
    for gi, idx in enumerate(group_marks(marks)):
        g = [marks[i] for i in idx]
        mal = np.array([m["malignancy"] for m in g], float)
        c = np.mean([m["centroid"] for m in g], 0)
        if len({m["session"] for m in g}) < len(g):
            problems.append(f"{row['PatientID']} nodule group {gi}: one reader marked it twice (kept)")
        patch = extract_patch(ct, c, size=STORE_SIZE, spacing=PATCH_SPACING_MM)
        patches.append(to_uint8(normalize_hu(patch)))
        eqd = float(np.mean([m["eq_sphere_diam_mm"] for m in g]))
        nod = dict(PatientID=row["PatientID"], SeriesInstanceUID=row["SeriesInstanceUID"], group=gi,
                   n_readers=len({m["session"] for m in g}), n_marks=len(g),
                   malignancy_ratings=" ".join(str(int(x)) for x in mal),
                   malignancy_median=float(np.median(mal)), malignancy_mean=float(mal.mean()),
                   centroid_x=c[0], centroid_y=c[1], centroid_z=c[2],
                   diameter_mm=float(np.mean([m["longest_axial_mm"] for m in g])),
                   eq_sphere_diam_mm=eqd, centre_hu=centre_hu(patch, eqd),
                   volume_ml=float(np.mean([m["volume_ml"] for m in g])),
                   slice_thickness_mm=ct["dz"], pixel_spacing_mm=float(ct["ps"][0]))
        for ch in CHARACTERISTICS:
            vals = [m[ch] for m in g if m.get(ch)]
            nod[f"{ch}_median"] = float(np.median(vals)) if vals else None
        nodules.append(nod)
        for i in idx:
            marks[i]["group"] = gi
    for m in marks:
        m.update(PatientID=row["PatientID"], SeriesInstanceUID=row["SeriesInstanceUID"])
        cx, cy, cz = m.pop("centroid")
        m.update(centroid_x=cx, centroid_y=cy, centroid_z=cz)
    return nodules, marks, (np.stack(patches) if patches else None), problems


def centre_hu(patch_hu, eq_diam_mm):
    """Mean HU in a small sphere at the patch centre: a placed nodule should be denser than lung."""
    r_mm = max(1.5, 0.35 * eq_diam_mm)
    n = patch_hu.shape[0]
    g = (np.arange(n) - (n - 1) / 2.0) * PATCH_SPACING_MM
    zz, yy, xx = np.meshgrid(g, g, g, indexing="ij", sparse=True)
    return float(patch_hu[zz ** 2 + yy ** 2 + xx ** 2 <= r_mm ** 2].mean())


def draw_qc(patches, nod, out_png, n=48, seed=0):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(seed)
    pick = rng.choice(len(patches), min(n, len(patches)), replace=False)
    cols = 8
    rows = int(np.ceil(len(pick) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(2 * cols, 2.2 * rows))
    for ax in np.ravel(axes):
        ax.axis("off")
    mid = STORE_SIZE // 2
    for ax, i in zip(np.ravel(axes), pick):
        r = nod.iloc[i]
        h = 32  # show the central 64 mm so small nodules are visible
        ax.imshow(from_uint8(patches[i][mid, mid - h:mid + h, mid - h:mid + h]), cmap="gray", vmin=0, vmax=1)
        rad = r["diameter_mm"] / 2 / PATCH_SPACING_MM
        ax.add_patch(plt.Circle((h - 0.5, h - 0.5), max(rad, 1), fill=False, color="r", lw=0.8))
        ax.set_title(f"{r['PatientID'][-4:]} m={r['malignancy_median']:g}\n{r['diameter_mm']:.0f} mm, "
                     f"{r['n_readers']}R", fontsize=7)
    fig.suptitle(f"LIDC patches (centre axial slice, {PATCH_SPACING_MM:g} mm, central 64 mm shown); "
                 f"red = mean reader diameter")
    fig.tight_layout()
    fig.savefig(out_png, dpi=100)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--series", default=RESULTS_DIR / "audit" / "lidc_series.csv", type=Path, help="from 00_lidc_check.py")
    ap.add_argument("--xml-root", required=True, type=Path, help="folder searched recursively for LIDC *.xml")
    ap.add_argument("--out", default=RESULTS_DIR / "lidc_patches", type=Path)
    ap.add_argument("--workers", type=int, default=4, help="parallel series (HDD: 2-4; SSD: more)")
    ap.add_argument("--limit", type=int, help="only the first N series (quick test)")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    series = pd.read_csv(args.series)
    series = series[series["Modality"] == "CT"]
    if args.limit:
        series = series.head(args.limit)

    print(f"Indexing XML under {args.xml_root} ...")
    xml_by_series = defaultdict(list)
    for dirpath, _, files in os.walk(args.xml_root):
        for f in files:
            if f.lower().endswith(".xml") and not f.startswith("."):
                uid = series_uid_of_xml(os.path.join(dirpath, f))
                if uid:
                    xml_by_series[uid].append(os.path.join(dirpath, f))
    have = series["SeriesInstanceUID"].isin(xml_by_series)
    print(f"  {len(xml_by_series)} CT reads indexed; {have.sum()} / {len(series)} CT series have XML")
    problems = [f"{r.PatientID} {r.SeriesInstanceUID}: no XML found" for r in series[~have].itertuples()]
    todo = series[have]

    parts = args.out / "patches_parts"
    parts.mkdir(exist_ok=True)
    for f in parts.glob("*.npy"):
        f.unlink()
    all_nod, all_marks, part_files = [], [], []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(process_series, dict(r), xml_by_series[r["SeriesInstanceUID"]]): r["PatientID"]
                for _, r in todo.iterrows()}
        for n_done, f in enumerate(as_completed(futs), 1):
            nod, marks, patches, probs = f.result()
            problems += probs
            if patches is not None:
                all_nod += nod
                all_marks += marks
                pf = parts / f"{len(part_files):05d}.npy"
                np.save(pf, patches)
                part_files.append((pf, len(patches)))
            if n_done % 20 == 0 or n_done == len(futs):
                print(f"  {n_done}/{len(futs)} series, {len(all_nod)} nodules")

    if not all_nod:
        raise SystemExit("No nodules extracted - see problems.txt")
    nod = pd.DataFrame(all_nod)
    nod["patch_index"] = np.arange(len(nod))
    n_total = sum(n for _, n in part_files)
    assert n_total == len(nod)
    patches = np.lib.format.open_memmap(args.out / "patches.npy", mode="w+", dtype=np.uint8,
                                        shape=(n_total,) + (STORE_SIZE,) * 3)
    i = 0
    for pf, n in part_files:  # same order as all_nod
        patches[i:i + n] = np.load(pf)
        i += n
        pf.unlink()
    patches.flush()
    parts.rmdir()
    nod.to_csv(args.out / "nodules.csv", index=False)
    pd.DataFrame(all_marks).to_csv(args.out / "reader_marks.csv", index=False)
    draw_qc(patches, nod, args.out / "patches_qc.png")

    lab = nod[nod["n_readers"] >= 3]
    summary = [
        f"Series processed: {len(todo)}; patients: {nod['PatientID'].nunique()}; nodules: {len(nod)}",
        f"Readers per nodule: {nod['n_readers'].value_counts().sort_index().to_dict()}",
        f"Median malignancy (>= 3 readers): {lab['malignancy_median'].value_counts().sort_index().to_dict()}",
        f"  -> benign (< 3): {(lab['malignancy_median'] < 3).sum()}, malignant (> 3): "
        f"{(lab['malignancy_median'] > 3).sum()}, excluded (= 3): {(lab['malignancy_median'] == 3).sum()}",
        "Nodule diameter (mm): " + nod["diameter_mm"].describe().round(1).to_string().replace("\n", " | "),
        f"Patch centre denser than -600 HU (nodule placed at the centre): {(nod['centre_hu'] > -600).mean():.0%} "
        f"of all nodules, {(nod.loc[nod['diameter_mm'] >= 6, 'centre_hu'] > -600).mean():.0%} of nodules >= 6 mm "
        f"(expect most; low values mean misplaced outlines)",
    ]
    (args.out / "problems.txt").write_text("\n".join(summary + [""] + problems) + "\n")
    print("\n".join(summary))
    print(f"{len(problems)} problem lines in {args.out}/problems.txt")
    print(f"Wrote {args.out}/nodules.csv, reader_marks.csv, patches.npy {patches.shape}, patches_qc.png")


if __name__ == "__main__":
    main()
