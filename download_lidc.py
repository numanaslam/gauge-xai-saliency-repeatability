#!/usr/bin/env python3
"""
Download the LIDC-IDRI CT series that are missing locally, from IDC (s5cmd via idc-index),
plus the LIDC XML reader annotations from TCIA (IDC does not host the XML).

Only CT series are fetched (IDC also hosts LIDC SEG/SR conversions; not needed here). Uses the
idc-index default layout, which matches the existing download:
  <download-dir>/lidc_idri/<PatientID>/<StudyInstanceUID>/CT_<SeriesInstanceUID>/*.dcm
so --download-dir is the folder that CONTAINS lidc_idri (e.g. D:\\data).

A series counts as present when its folder holds at least as many files as IDC lists instances;
partial folders are downloaded again (s5cmd skips files already there). Safe to re-run.

Usage:
  pip install --upgrade idc-index
  python download_lidc.py --download-dir D:\\data --dry-run      # list what is missing + size
  python download_lidc.py --download-dir D:\\data                # download CT + XML
  python download_lidc.py --download-dir D:\\data --skip-xml
"""
import argparse
import urllib.request
import zipfile
from pathlib import Path

XML_URL = "https://www.cancerimagingarchive.net/wp-content/uploads/LIDC-XML-only.zip"
COLLECTION = "lidc_idri"


def local_counts(coll_dir):
    """SeriesInstanceUID -> number of files, from folders named <Modality>_<SeriesInstanceUID>."""
    counts = {}
    if not coll_dir.is_dir():
        return counts
    for d in coll_dir.glob("*/*/*"):
        if d.is_dir() and "_" in d.name:
            counts[d.name.split("_", 1)[1]] = sum(1 for f in d.iterdir() if f.is_file())
    return counts


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--download-dir", required=True, type=Path, help="folder that contains (or will contain) lidc_idri")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--skip-xml", action="store_true")
    ap.add_argument("--xml-dir", type=Path, help="where to unzip the XML (default: <download-dir>/LIDC-XML-only)")
    args = ap.parse_args()

    from idc_index import IDCClient

    client = IDCClient()
    idx = client.index
    ct = idx[(idx["collection_id"] == COLLECTION) & (idx["Modality"] == "CT")]
    count_col = next((c for c in ("instanceCount", "instance_count") if c in ct.columns), None)
    size_col = next((c for c in ("series_size_MB", "series_size_mb") if c in ct.columns), None)
    print(f"IDC {COLLECTION}: {len(ct)} CT series, {ct['PatientID'].nunique()} patients"
          + (f", {ct[size_col].sum() / 1024:.0f} GB" if size_col else ""))

    have = local_counts(args.download_dir / COLLECTION)
    def missing(r):
        n = have.get(r["SeriesInstanceUID"])
        return n is None or (count_col is not None and n < int(r[count_col]))
    todo = ct[ct.apply(missing, axis=1)]
    print(f"Local complete CT series: {len(ct) - len(todo)}; to download: {len(todo)} series, "
          f"{todo['PatientID'].nunique()} patients"
          + (f", {todo[size_col].sum() / 1024:.1f} GB" if size_col else ""))
    todo[["PatientID", "SeriesInstanceUID"]].to_csv(args.download_dir / "lidc_missing_series.csv", index=False)

    if not args.dry_run and len(todo):
        uids = todo["SeriesInstanceUID"].tolist()
        for i in range(0, len(uids), 200):  # batches keep each s5cmd manifest small; re-run resumes
            print(f"Downloading series {i + 1}-{min(i + 200, len(uids))} of {len(uids)} ...")
            client.download_from_selection(downloadDir=str(args.download_dir), seriesInstanceUID=uids[i:i + 200],
                                           dirTemplate="%collection_id/%PatientID/%StudyInstanceUID/%Modality_%SeriesInstanceUID")
        have = local_counts(args.download_dir / COLLECTION)
        left = ct[ct.apply(missing, axis=1)]
        print(f"After download: {len(ct) - len(left)} / {len(ct)} CT series complete"
              + (" - re-run to retry the rest" if len(left) else ""))

    if not args.skip_xml and not args.dry_run:
        xml_dir = args.xml_dir or args.download_dir / "LIDC-XML-only"
        if any(xml_dir.rglob("*.xml")):
            print(f"XML already present in {xml_dir}")
        else:
            zpath = args.download_dir / "LIDC-XML-only.zip"
            if not zpath.exists():
                print(f"Downloading {XML_URL} ...")
                urllib.request.urlretrieve(XML_URL, zpath)
            with zipfile.ZipFile(zpath) as z:
                z.extractall(xml_dir)
            print(f"XML: {sum(1 for _ in xml_dir.rglob('*.xml'))} files in {xml_dir}")


if __name__ == "__main__":
    main()
