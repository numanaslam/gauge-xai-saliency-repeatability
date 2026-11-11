#!/usr/bin/env python3
"""
Repair a patches.npy that has unreadable rows (bad disk sectors): copy every readable row into a new
file and recompute unreadable rows from the LIDC DICOM + XML with the same code as 04_lidc_patches.py.
Row order and patch_index are kept, so the repaired file is byte-identical to the original wherever the
original was readable and the models trained on it stay comparable.

A recomputed nodule is matched to its nodules.csv row by centroid (< 0.01 mm) and number of readers, so
the order in which the XML files are found does not matter. The patch-centre HU is checked as well.

Writes <dst>/patches.npy, nodules.csv, reader_marks.csv and repair_report.txt.

Usage:
  python repair_patches.py --xml-root D:\\data\\LIDC-XML-only
      (defaults: --src ../results/lidc_patches  --dst ../results/lidc_patches_v2)
Do NOT delete the source folder before the drive's bad sectors are marked (chkdsk D: /r): the old file
keeps them occupied, so new data is not written onto them.
"""
import argparse
import importlib.util
import os
import shutil
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from gauge_lib import RESULTS_DIR, npy_layout


def load_04():
    spec = importlib.util.spec_from_file_location("lidc_patches", Path(__file__).with_name("04_lidc_patches.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def read_row(f, offset, row_bytes, retries=2):
    for attempt in range(retries):
        try:
            f.seek(offset)
            buf = f.read(row_bytes)
            if len(buf) == row_bytes:
                return buf
        except OSError:
            time.sleep(0.5)
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default=RESULTS_DIR / "lidc_patches", type=Path)
    ap.add_argument("--dst", default=RESULTS_DIR / "lidc_patches_v2", type=Path)
    ap.add_argument("--xml-root", required=True, type=Path)
    ap.add_argument("--series", default=RESULTS_DIR / "audit" / "lidc_series.csv", type=Path)
    args = ap.parse_args()
    if args.dst.resolve() == args.src.resolve():
        raise SystemExit("--dst must differ from --src")
    args.dst.mkdir(parents=True, exist_ok=True)
    m04 = load_04()

    src = args.src / "patches.npy"
    shape, dtype, off = npy_layout(src)
    row_bytes = int(np.prod(shape[1:])) * dtype.itemsize
    nod = pd.read_csv(args.src / "nodules.csv")
    if len(nod) != shape[0]:
        raise SystemExit(f"nodules.csv has {len(nod)} rows but patches.npy has {shape[0]}")
    series = pd.read_csv(args.series).set_index("SeriesInstanceUID")
    xml_index = None
    recomputed = {}  # SeriesInstanceUID -> (nodules list, patches array)
    report, repaired, failed = [], [], []

    def recompute(i):
        nonlocal xml_index
        r = nod.iloc[i]
        suid = r["SeriesInstanceUID"]
        if suid not in recomputed:
            if xml_index is None:
                print("  indexing XML (once) ...", flush=True)
                xml_index = defaultdict(list)
                for dirpath, _, files in os.walk(args.xml_root):
                    for fn in files:
                        if fn.lower().endswith(".xml") and not fn.startswith("."):
                            uid = m04.series_uid_of_xml(os.path.join(dirpath, fn))
                            if uid:
                                xml_index[uid].append(os.path.join(dirpath, fn))
            if suid not in series.index:
                return None, f"series {suid} not in {args.series}"
            srow = dict(series.loc[suid])
            srow["SeriesInstanceUID"] = suid
            nods, _, patches, probs = m04.process_series(srow, xml_index.get(suid, []))
            recomputed[suid] = (nods, patches, probs)
        nods, patches, probs = recomputed[suid]
        if patches is None:
            return None, "recompute produced no patches: " + "; ".join(probs[:2])
        c = np.array([r["centroid_x"], r["centroid_y"], r["centroid_z"]])
        d = [np.linalg.norm(np.array([n["centroid_x"], n["centroid_y"], n["centroid_z"]]) - c) for n in nods]
        k = int(np.argmin(d))
        if d[k] > 0.01 or nods[k]["n_readers"] != r["n_readers"]:
            return None, f"no matching nodule (closest {d[k]:.3f} mm, readers {nods[k]['n_readers']} vs {r['n_readers']})"
        if abs(nods[k]["centre_hu"] - r["centre_hu"]) > 1.0:
            return None, f"centre HU differs ({nods[k]['centre_hu']:.1f} vs {r['centre_hu']:.1f})"
        return patches[k].tobytes(), f"recomputed from {r['PatientID']} (match {d[k]:.4f} mm)"

    t0 = time.time()
    with open(src, "rb", buffering=0) as fs, open(args.dst / "patches.npy", "wb") as fd:
        fs.seek(0)
        fd.write(fs.read(off))  # identical .npy header
        for i in range(shape[0]):
            buf = read_row(fs, off + i * row_bytes, row_bytes)
            if buf is None:
                buf, msg = recompute(i)
                if buf is None:
                    failed.append(i)
                    report.append(f"row {i}: FAILED - {msg}")
                    buf = b"\0" * row_bytes
                else:
                    repaired.append(i)
                    report.append(f"row {i}: {msg}")
                print(f"  {report[-1]}", flush=True)
            fd.write(buf)
            if (i + 1) % 200 == 0 or i + 1 == shape[0]:
                print(f"  {i + 1}/{shape[0]} rows ({(time.time() - t0) / 60:.1f} min), repaired {len(repaired)}, "
                      f"failed {len(failed)}", flush=True)
    for extra in ("nodules.csv", "reader_marks.csv", "patches_qc.png"):
        if (args.src / extra).exists():
            shutil.copy2(args.src / extra, args.dst / extra)

    # read back the repaired rows from the new file
    bad_back = []
    with open(args.dst / "patches.npy", "rb", buffering=0) as f:
        for i in repaired:
            if read_row(f, off + i * row_bytes, row_bytes) is None:
                bad_back.append(i)
    summary = [f"source {src}: {shape[0]} rows; repaired {len(repaired)}, failed {len(failed)}, "
               f"unreadable in the new file {len(bad_back)}"]
    if failed:
        summary.append(f"FAILED rows (zeros in the new file - do not train until fixed): {failed}")
    if bad_back:
        summary.append(f"repaired rows unreadable in the NEW file (the disk is still failing): {bad_back}")
    (args.dst / "repair_report.txt").write_text("\n".join(summary + [""] + report) + "\n")
    print("\n".join(summary))
    if not failed and not bad_back:
        print(f"\nRepaired file: {args.dst / 'patches.npy'}. Check it with\n"
              f"  python verify_patches.py --file {args.dst / 'patches.npy'}\n"
              f"then train with --patches {args.dst}")


if __name__ == "__main__":
    main()
