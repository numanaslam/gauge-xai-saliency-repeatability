#!/usr/bin/env python3
"""
GAUGE-XAI step 1: audit the RIDER Lung CT (v3) download.

Reads DICOM headers (plus one slice per CT series for a duplicate check) and writes:
  <out>/series_manifest.csv     one row per DICOM series (CT, SEG, anything else)
  <out>/condition_manifest.csv  one row per CT series in the design grid
                                (patient x acquisition x reconstruction) with its SEG
  <out>/audit_report.txt        everything that departs from the expected 32 x 2 x 6 design
  <out>/complete_patients.txt   patients with all 12 cells usable
  <out>/acquisitions.csv        every acquisition found per patient and which two became scan1/scan2

The download also contains series from derived collections (QIN segmentation challenge, RTSTRUCT
conversions). These are excluded: CT copies without Manufacturer and SEGs other than the RIDER v3
tumour contours (SeriesDescription "Tumor Segmentation").

An acquisition is (StudyDate, FrameOfReferenceUID, AcquisitionTime). AcquisitionTime is shared by
all reconstructions of one acquisition, while Study and FrameOfReference are not reliable (some
repeat scans share both). Some patients have 3+ acquisitions; the same-day pair with the best
tumour-SEG coverage (1.25L in both first) becomes scan1/scan2.

Usage:
  python 01_rider_audit.py --root /data/rider_lung_ct --out audit
  python 01_rider_audit.py --from-manifest audit/series_manifest.csv --out audit   # re-grid, no rescan
"""
import argparse
import hashlib
import os
from collections import defaultdict
from datetime import datetime, timedelta
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom

from gauge_lib import RESULTS_DIR

THICKNESSES = (1.25, 2.5, 5.0)
KERNELS = ("L", "S")
SETTINGS = [f"{t:g}{k}" for t in THICKNESSES for k in KERNELS]  # 1.25L 1.25S 2.5L 2.5S 5L 5S
ACQS = ("scan1", "scan2")
FAST_TAGS = ["SeriesInstanceUID", "SOPInstanceUID", "Modality", "InstanceNumber"]
SKIP_SUFFIXES = {".txt", ".csv", ".xlsx", ".xls", ".json", ".pdf", ".zip", ".tcia", ".md", ".py"}
TUMOR_SEG_DESCRIPTION = "Tumor Segmentation"
PRIMARY_SETTING = "1.25L"


def get(ds, kw, default=None):
    v = getattr(ds, kw, None)
    if v is None or (isinstance(v, str) and v == ""):
        return default
    return v


def scan_files(root: Path):
    """Index every DICOM file by series with a fast partial header read."""
    series_files = defaultdict(list)  # suid -> [(path, instance_number)]
    sop_to_series = {}
    n_bad = 0
    for dirpath, _, files in os.walk(root):
        for f in files:
            p = Path(dirpath) / f
            if f.startswith(".") or p.suffix.lower() in SKIP_SUFFIXES:
                continue
            try:
                ds = pydicom.dcmread(p, stop_before_pixels=True, specific_tags=FAST_TAGS)
                suid = ds.SeriesInstanceUID
            except Exception:
                n_bad += 1
                continue
            inst = get(ds, "InstanceNumber")
            series_files[suid].append((p, int(inst) if inst is not None else 0))
            if get(ds, "SOPInstanceUID"):
                sop_to_series[ds.SOPInstanceUID] = suid
    return series_files, sop_to_series, n_bad


def norm_kernel(k):
    if k is None:
        return None
    s = " ".join(map(str, k)) if isinstance(k, pydicom.multival.MultiValue) else str(k)
    s = s.upper().strip()
    if "LUNG" in s:
        return "L"
    if "STANDARD" in s or s in {"STD", "STND"}:
        return "S"
    return s or None


def norm_thickness(t):
    if t is None:
        return None
    try:
        t = float(t)
    except (TypeError, ValueError):
        return None
    for ref in THICKNESSES:
        if abs(t - ref) < 0.15:
            return ref
    return round(t, 3)


def seg_referenced_series(ds, sop_to_series):
    """CT SeriesInstanceUID a SEG points at: direct reference first, else via source SOP UIDs."""
    for item in get(ds, "ReferencedSeriesSequence", []) or []:
        uid = get(item, "SeriesInstanceUID")
        if uid:
            return uid, "ReferencedSeriesSequence"
    hits = defaultdict(int)
    groups = list(get(ds, "PerFrameFunctionalGroupsSequence", []) or [])
    groups += list(get(ds, "SharedFunctionalGroupsSequence", []) or [])
    for g in groups:
        for der in get(g, "DerivationImageSequence", []) or []:
            for src in get(der, "SourceImageSequence", []) or []:
                s = sop_to_series.get(get(src, "ReferencedSOPInstanceUID"))
                if s:
                    hits[s] += 1
    if hits:
        return max(hits, key=hits.get), "SourceImageSequence"
    return None, "none"


def summarize_series(suid, files, sop_to_series):
    paths = [p for p, _ in files]
    ds = pydicom.dcmread(paths[0], stop_before_pixels=True)
    mod = get(ds, "Modality")
    image_type = get(ds, "ImageType", [])
    row = dict(
        PatientID=get(ds, "PatientID"),
        StudyInstanceUID=get(ds, "StudyInstanceUID"),
        SeriesInstanceUID=suid,
        Modality=mod,
        SeriesNumber=get(ds, "SeriesNumber"),
        SeriesDescription=get(ds, "SeriesDescription"),
        ImageType="\\".join(map(str, image_type)),
        StudyDate=get(ds, "StudyDate"),
        SeriesTime=get(ds, "SeriesTime"),
        AcquisitionTime=get(ds, "AcquisitionTime"),
        FrameOfReferenceUID=get(ds, "FrameOfReferenceUID"),
        Manufacturer=get(ds, "Manufacturer"),
        Model=get(ds, "ManufacturerModelName"),
        n_files=len(paths),
        folder=str(paths[0].parent),
    )
    if mod == "CT":
        row.update(
            SliceThickness_raw=get(ds, "SliceThickness"),
            thickness=norm_thickness(get(ds, "SliceThickness")),
            ConvolutionKernel_raw=str(get(ds, "ConvolutionKernel")),
            kernel=norm_kernel(get(ds, "ConvolutionKernel")),
            PixelSpacing=str([float(x) for x in (get(ds, "PixelSpacing", []) or [])]),
            Rows=get(ds, "Rows"),
            Columns=get(ds, "Columns"),
            KVP=get(ds, "KVP"),
        )
    elif mod == "SEG":
        ref, how = seg_referenced_series(ds, sop_to_series)
        segs = get(ds, "SegmentSequence", []) or []
        row.update(
            seg_ref_series=ref,
            seg_ref_method=how,
            SegmentationType=get(ds, "SegmentationType"),
            n_segments=len(segs),
            segment_labels="; ".join(str(get(s, "SegmentLabel", "")) for s in segs),
            NumberOfFrames=get(ds, "NumberOfFrames"),
            seg_file=str(paths[0]),
        )
    return row


def norm_time(t):
    """DICOM TM (or the float pandas makes of it) -> 'HHMMSS', or None."""
    if t is None or (isinstance(t, float) and np.isnan(t)):
        return None
    s = str(t).strip().split(".")[0]
    return s.zfill(6) if s.isdigit() and len(s) <= 6 else None


def norm_date(d):
    s = str(d).strip().split(".")[0]
    return s if s.isdigit() and len(s) == 8 else None


def acq_start(date, time):
    if date is None or time is None:
        return None
    return datetime.strptime(date + time, "%Y%m%d%H%M%S")


def assign_acquisitions(ct):
    """Group each patient's grid CT series into acquisitions and pick the scan1/scan2 pair.

    ct needs a boolean column has_seg. Returns (ct with acq/acq_id columns, acquisitions table, problems).
    """
    ct = ct.copy()
    ct["acq_date"] = ct["StudyDate"].map(norm_date)
    ct["acq_time"] = ct["AcquisitionTime"].map(norm_time)
    ct["acq_time"] = ct["acq_time"].fillna(ct["SeriesTime"].map(norm_time))
    ct["acq_id"] = [f"{d}_{t}_{str(f)[-6:]}" for d, t, f in
                    zip(ct["acq_date"], ct["acq_time"], ct["FrameOfReferenceUID"])]
    ct["acq"] = None
    acq_rows, problems = [], []
    for pid, g in ct.groupby("PatientID"):
        acqs = []
        for aid, gg in g.groupby("acq_id"):
            r0 = gg.iloc[0]
            acqs.append(dict(PatientID=pid, acq_id=aid, date=r0["acq_date"],
                             start=acq_start(r0["acq_date"], r0["acq_time"]),
                             n_series=len(gg), seg_settings=set(gg.loc[gg["has_seg"], "setting"])))
        acqs.sort(key=lambda a: (a["start"] is None, a["start"] or datetime.min))
        if len(acqs) < 2:
            problems.append(f"{pid}: only {len(acqs)} acquisition(s) found -> no repeat scan")
            acq_rows += [dict(a, role="unpaired") for a in acqs]
            continue

        def score(pair):
            a, b = acqs[pair[0]], acqs[pair[1]]
            both = a["seg_settings"] & b["seg_settings"]
            return (a["date"] == b["date"], PRIMARY_SETTING in both, len(both),
                    len(a["seg_settings"] | b["seg_settings"]), -(pair[0] + pair[1]))

        i, j = max(combinations(range(len(acqs)), 2), key=score)
        a, b = acqs[i], acqs[j]
        ct.loc[g.index[g["acq_id"] == a["acq_id"]], "acq"] = "scan1"
        ct.loc[g.index[g["acq_id"] == b["acq_id"]], "acq"] = "scan2"
        gap = (b["start"] - a["start"]) if a["start"] and b["start"] else None
        for k, x in enumerate(acqs):
            acq_rows.append(dict(x, role={i: "scan1", j: "scan2"}.get(k, "dropped"),
                                 rescan_interval_min=round(gap / timedelta(minutes=1), 1) if gap else None))
        if len(acqs) > 2:
            problems.append(f"{pid}: {len(acqs)} acquisitions, kept {a['acq_id']} + {b['acq_id']}; dropped "
                            + ", ".join(f"{x['acq_id']} ({len(x['seg_settings'])} SEG)"
                                        for k, x in enumerate(acqs) if k not in (i, j)))
        if a["date"] != b["date"]:
            problems.append(f"{pid}: scan1 and scan2 are on different dates ({a['date']} vs {b['date']}) "
                            f"-> not a same-day repeat; decide whether to exclude")
        elif gap is not None and gap > timedelta(minutes=60):
            problems.append(f"{pid}: scan1 -> scan2 interval is {gap}")
    acq_table = pd.DataFrame(acq_rows)
    if len(acq_table):
        acq_table["seg_settings"] = acq_table["seg_settings"].map(lambda s: " ".join(sorted(s)))
    return ct, acq_table, problems


def series_files_from_folder(folder, suid):
    out = []
    if not folder or not Path(folder).is_dir():
        return out
    for p in Path(folder).iterdir():
        if p.name.startswith(".") or not p.is_file():
            continue
        try:
            ds = pydicom.dcmread(p, stop_before_pixels=True, specific_tags=FAST_TAGS)
        except Exception:
            continue
        if getattr(ds, "SeriesInstanceUID", None) == suid:
            inst = get(ds, "InstanceNumber")
            out.append((p, int(inst) if inst is not None else 0))
    return out


def middle_slice_hash(files):
    if not files:
        return "NA"
    files = sorted(files, key=lambda x: x[1])
    p = files[len(files) // 2][0]
    try:
        ds = pydicom.dcmread(p)
        return hashlib.md5(ds.pixel_array.tobytes()).hexdigest()
    except Exception as e:  # noqa: BLE001
        return f"ERR:{type(e).__name__}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--root", type=Path, help="folder you downloaded rider_lung_ct into")
    src.add_argument("--from-manifest", type=Path,
                     help="reuse an existing series_manifest.csv instead of rescanning the DICOM tree")
    ap.add_argument("--out", default=RESULTS_DIR / "audit", type=Path)
    ap.add_argument("--expected-patients", type=int, default=32)
    ap.add_argument("--seg-description", default=TUMOR_SEG_DESCRIPTION,
                    help="SeriesDescription of the SEGs to use (RIDER v3 tumour contours)")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    if args.root:
        print(f"Indexing DICOM files under {args.root} (a few minutes for ~80k files) ...")
        series_files, sop_to_series, n_bad = scan_files(args.root)
        n_files = sum(len(v) for v in series_files.values())
        print(f"  {len(series_files)} series, {n_files} files, {n_bad} unreadable/non-DICOM skipped")
        if not series_files:
            raise SystemExit("No DICOM series found. Check --root.")
        series = pd.DataFrame([summarize_series(s, f, sop_to_series) for s, f in series_files.items()])
        series["SeriesNumber"] = pd.to_numeric(series["SeriesNumber"], errors="coerce")
        series = series.sort_values(["PatientID", "Modality", "SeriesNumber"], na_position="last")
        series.to_csv(args.out / "series_manifest.csv", index=False)
    else:
        series_files = None
        series = pd.read_csv(args.from_manifest, low_memory=False,
                             dtype={"StudyDate": str, "SeriesTime": str, "AcquisitionTime": str})
        if args.from_manifest.resolve() != (args.out / "series_manifest.csv").resolve():
            series.to_csv(args.out / "series_manifest.csv", index=False)

    def files_of(suid, folder):
        if series_files is not None:
            return series_files[suid]
        return series_files_from_folder(folder, suid)

    ct_all = series[series.Modality == "CT"].copy()
    seg_all = series[series.Modality == "SEG"].copy() if "seg_ref_series" in series else series.iloc[0:0]
    seg = seg_all[seg_all["SeriesDescription"] == args.seg_description]

    # Grid-eligible CT: an original scanner series (derived-collection copies have no Manufacturer),
    # not a localizer, a real volume, and one of the six expected settings.
    ct_all["setting"] = [
        f"{t:g}{k}" if t in THICKNESSES and k in KERNELS else None
        for t, k in zip(ct_all["thickness"], ct_all["kernel"])
    ]
    is_loc = ct_all["ImageType"].str.upper().str.contains("LOCALIZER", na=False)
    is_copy = ct_all["Manufacturer"].isna()
    eligible = ct_all[(~is_loc) & (~is_copy) & (ct_all["n_files"] >= 10) & ct_all["setting"].notna()].copy()
    excluded_ct = ct_all.drop(eligible.index)

    seg_by_ref = defaultdict(list)
    for _, r in seg.iterrows():
        if isinstance(r.get("seg_ref_series"), str):
            seg_by_ref[r["seg_ref_series"]].append(r)
    eligible["has_seg"] = eligible["SeriesInstanceUID"].map(lambda u: len(seg_by_ref.get(u, [])) > 0)

    eligible, acq_table, acq_problems = assign_acquisitions(eligible)
    acq_table.to_csv(args.out / "acquisitions.csv", index=False)

    rows = []
    for _, r in eligible.iterrows():
        segs = seg_by_ref.get(r["SeriesInstanceUID"], [])
        rows.append(dict(
            PatientID=r["PatientID"], acq=r["acq"], acq_id=r["acq_id"], setting=r["setting"],
            SeriesNumber=r["SeriesNumber"],
            ct_series_uid=r["SeriesInstanceUID"], ct_folder=r["folder"], n_slices=r["n_files"],
            Model=r["Model"], PixelSpacing=r["PixelSpacing"],
            seg_series_uid=segs[0]["SeriesInstanceUID"] if segs else None,
            seg_file=segs[0]["seg_file"] if segs else None,
            n_seg_for_series=len(segs),
            n_segments=segs[0]["n_segments"] if segs else None,
            # only the paired acquisitions are hashed (reads one slice per series)
            mid_slice_md5=(middle_slice_hash(files_of(r["SeriesInstanceUID"], r["folder"]))
                           if r["acq"] in ACQS else None),
        ))
    cond = pd.DataFrame(rows)

    # ---- per-cell checks --------------------------------------------------------------
    report = []
    pids = sorted(series["PatientID"].dropna().unique())
    report.append(f"Patients found: {len(pids)} (expected {args.expected_patients})")
    report.append(f"Series by modality: {series['Modality'].value_counts().to_dict()}")
    report.append(f"Scanner models: {series['Model'].value_counts().to_dict()}")
    report.append("CT thickness x kernel (all CT series):")
    report.append(ct_all.groupby(["SliceThickness_raw", "ConvolutionKernel_raw"]).size().to_string())
    report.append(f"CT series excluded from the grid: {len(excluded_ct)} "
                  f"(derived copies without Manufacturer: {int(is_copy.sum())}; "
                  f"localizers, <10 files, other settings: {len(excluded_ct) - int(is_copy.sum())})")
    report.append(f"SEG series: {len(seg_all)}; used ('{args.seg_description}'): {len(seg)}; "
                  f"ignored by source: {seg_all[seg_all['SeriesDescription'] != args.seg_description]['Model'].value_counts().to_dict()}")
    report.append("")
    n_acq = acq_table.groupby("PatientID").size() if len(acq_table) else pd.Series(dtype=int)
    report.append(f"Acquisitions per patient: {n_acq.value_counts().sort_index().to_dict()}")
    if "rescan_interval_min" in acq_table:
        gaps = acq_table[acq_table.role == "scan2"]["rescan_interval_min"]
        report.append("scan1 -> scan2 interval (min): "
                      + gaps.describe().round(1).to_string().replace("\n", " | "))
    report.append("Acquisition pairing notes:" + ("" if acq_problems else " none"))
    report += ["  " + p for p in acq_problems]

    cond["usable"] = False
    cell_issues, complete = [], []
    for pid in pids:
        missing = []
        for acq in ACQS:
            for s in SETTINGS:
                cell = cond[(cond.PatientID == pid) & (cond.acq == acq) & (cond.setting == s)]
                with_seg = cell[cell.seg_file.notna()]
                if cell.empty:
                    missing.append(f"{acq}/{s}: no CT")
                elif with_seg.empty:
                    missing.append(f"{acq}/{s}: CT but no SEG")
                else:
                    pick = with_seg.sort_values("SeriesNumber").index[0]
                    cond.loc[pick, "usable"] = True
                    if len(with_seg) > 1:
                        h = with_seg["mid_slice_md5"]
                        same = "identical pixels" if h.nunique() == 1 and not str(h.iloc[0]).startswith(("NA", "ERR")) \
                            else "pixels differ or unchecked"
                        missing.append(f"{acq}/{s}: {len(with_seg)} CT+SEG candidates "
                                       f"(series {with_seg.SeriesNumber.astype(int).tolist()}, {same}); "
                                       f"using series {int(cond.loc[pick, 'SeriesNumber'])}")
        # duplicate-repeat check: identical middle slice in scan1 and scan2 for a setting
        for s in SETTINGS:
            sel = (cond.PatientID == pid) & (cond.setting == s) & cond.usable
            h1 = set(cond[sel & (cond.acq == "scan1")].mid_slice_md5.dropna())
            h2 = set(cond[sel & (cond.acq == "scan2")].mid_slice_md5.dropna())
            if {h for h in h1 & h2 if not str(h).startswith(("NA", "ERR"))}:
                missing.append(f"{s}: scan1 and scan2 have identical pixel data (not a real repeat)")
                cond.loc[(cond.PatientID == pid) & (cond.setting == s), "usable"] = False
        n_usable = int(cond[(cond.PatientID == pid) & cond.usable].shape[0])
        if n_usable == 2 * len(SETTINGS):
            complete.append(pid)
        if missing:
            cell_issues.append(f"{pid}: ({n_usable}/12 usable)")
            cell_issues += ["    " + m for m in missing]

    report.append("")
    report.append(f"Patients with all 12 cells usable: {len(complete)} / {len(pids)}")
    paired = (cond[cond.usable].groupby(["setting", "PatientID"]).acq.nunique() == 2).groupby("setting").sum()
    report.append("Patients with a usable scan1+scan2 pair, per setting (n for per-reconstruction stats):")
    report.append("  " + "  ".join(f"{s}={int(paired.get(s, 0))}" for s in SETTINGS))
    report.append("Cell notes:" + ("" if cell_issues else " none"))
    report += ["  " + c for c in cell_issues]

    if len(seg):
        linked = set(cond.loc[cond.usable, "seg_series_uid"].dropna())
        orphan = seg[~seg["SeriesInstanceUID"].isin(linked)]
        report.append("")
        report.append(f"Tumour SEGs: {len(seg)}; used in the grid: {len(linked)}; not used: {len(orphan)}")
        for _, r in orphan.iterrows():
            ref = r["seg_ref_series"] if isinstance(r["seg_ref_series"], str) else None
            why = "no referenced CT found" if ref is None else (
                "references a CT outside the grid" if ref not in set(eligible.SeriesInstanceUID)
                else "its CT is in a dropped acquisition or not the chosen duplicate")
            report.append(f"  {r['PatientID']} {r['SeriesInstanceUID'][-12:]}: {why}")
        report.append(f"SEG reference method: {seg['seg_ref_method'].value_counts().to_dict()}")
        report.append(f"Segments per SEG: {seg['n_segments'].value_counts().to_dict()}")
        report.append(f"SegmentationType: {seg['SegmentationType'].value_counts().to_dict()}")
    if len(cond):
        report.append("")
        report.append("Slices per series by setting (min / median / max):")
        report.append(cond[cond.usable].groupby("setting")["n_slices"].agg(["min", "median", "max"]).to_string())

    cond.sort_values(["PatientID", "acq", "setting"]).to_csv(args.out / "condition_manifest.csv", index=False)
    (args.out / "complete_patients.txt").write_text("\n".join(complete) + "\n")
    (args.out / "audit_report.txt").write_text("\n".join(report) + "\n")
    print("\n".join(report))
    print(f"\nWrote {args.out}/series_manifest.csv, condition_manifest.csv, acquisitions.csv, audit_report.txt")


if __name__ == "__main__":
    main()
