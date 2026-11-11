#!/usr/bin/env python3
"""
Quick sanity check of the LIDC-IDRI download (headers only).

Expected (TCIA v4): 1,010 patients, 1,018 CT scans, annotation XML alongside the CT series.
Writes <out>/lidc_series.csv for later patch extraction.

Usage:
  python 00_lidc_check.py --root /data/lidc_idri --out audit
"""
import argparse
import os
from collections import defaultdict
from pathlib import Path

import pandas as pd
import pydicom

from gauge_lib import RESULTS_DIR

TAGS = ["SeriesInstanceUID", "PatientID", "Modality", "SliceThickness", "Manufacturer"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--out", default=RESULTS_DIR / "audit", type=Path)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    series = {}
    counts = defaultdict(int)
    folders = {}
    n_xml = 0
    for dirpath, _, files in os.walk(args.root):
        for f in files:
            p = Path(dirpath) / f
            if f.lower().endswith(".xml"):
                n_xml += 1
                continue
            try:
                ds = pydicom.dcmread(p, stop_before_pixels=True, specific_tags=TAGS)
                suid = ds.SeriesInstanceUID
            except Exception:
                continue
            counts[suid] += 1
            if suid not in series:
                series[suid] = dict(PatientID=getattr(ds, "PatientID", None),
                                    Modality=getattr(ds, "Modality", None),
                                    SliceThickness=getattr(ds, "SliceThickness", None),
                                    Manufacturer=getattr(ds, "Manufacturer", None))
                folders[suid] = str(p.parent)

    df = pd.DataFrame([dict(SeriesInstanceUID=s, n_files=counts[s], folder=folders[s], **v)
                       for s, v in series.items()])
    if df.empty:
        raise SystemExit("No DICOM found. Check --root.")
    ct = df[df.Modality == "CT"]
    print(f"Series by modality: {df.Modality.value_counts().to_dict()}")
    print(f"CT series: {len(ct)} (expect ~1,018); patients with CT: {ct.PatientID.nunique()} (expect 1,010)")
    print(f"XML annotation files: {n_xml}. If 0 (normal for an IDC download), download "
          f"LIDC-XML-only.zip from the TCIA collection page; annotations are needed for training.")
    print(f"CT series with < 50 slices: {(ct.n_files < 50).sum()}")
    print("Slice thickness (mm):", pd.to_numeric(ct.SliceThickness, errors="coerce").describe().round(2).to_dict())
    print("Manufacturers:", ct.Manufacturer.value_counts().to_dict())
    ct.to_csv(args.out / "lidc_series.csv", index=False)
    print(f"Wrote {args.out}/lidc_series.csv")


if __name__ == "__main__":
    main()
