#!/usr/bin/env python3
"""
Read every row of a large .npy (e.g. results/lidc_patches/patches.npy) with ordinary file reads and report
rows the disk cannot read. Optionally copy the file to another drive at the same time (--copy-to), so a
healthy copy is made in one pass. Also checks the file size against its header.

A "Windows fatal exception: page error" in 05_train.py means a read of this file failed; run this first.

Usage:
  python verify_patches.py                                         # check results/lidc_patches/patches.npy
  python verify_patches.py --copy-to D:\\some\\folder             # check + copy patches.npy and nodules.csv
"""
import argparse
import os
import shutil
import time
from pathlib import Path

import numpy as np

from gauge_lib import RESULTS_DIR, npy_layout


WIN_HINTS = {
    5: "ERROR_ACCESS_DENIED: a security filter (antivirus, Controlled Folder Access, a sync client) refuses the read",
    23: "ERROR_CRC: data error (cyclic redundancy check) - BAD SECTORS on the disk",
    33: "ERROR_LOCK_VIOLATION: another process holds a byte-range lock",
    225: "ERROR_VIRUS_INFECTED: antivirus blocks the file",
    362: "cloud-file error: the file is an online-only placeholder of a sync client (OneDrive, ...)",
    388: "cloud-file error: the sync client cannot provide the data",
    1117: "ERROR_IO_DEVICE: hardware I/O error - failing disk, cable or USB enclosure",
    1392: "ERROR_FILE_CORRUPT: file system corruption - run chkdsk D: /f",
    483: "ERROR_DEVICE_HARDWARE_ERROR: failing disk",
}


def win_read_error(path, offset, nbytes):
    """Windows only: read nbytes at offset with ReadFile and return (GetLastError code, message).
    Python maps several different Windows errors to errno 13, so this gives the real cause."""
    import ctypes
    from ctypes import wintypes

    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.CreateFileW.restype = wintypes.HANDLE
    k32.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, wintypes.LPVOID,
                                wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
    k32.SetFilePointerEx.argtypes = [wintypes.HANDLE, ctypes.c_longlong, ctypes.c_void_p, wintypes.DWORD]
    k32.SetFilePointerEx.restype = wintypes.BOOL
    k32.ReadFile.argtypes = [wintypes.HANDLE, wintypes.LPVOID, wintypes.DWORD, ctypes.POINTER(wintypes.DWORD),
                             wintypes.LPVOID]
    k32.ReadFile.restype = wintypes.BOOL
    k32.CloseHandle.argtypes = [wintypes.HANDLE]
    h = k32.CreateFileW(str(path), 0x80000000, 0x1 | 0x2, None, 3, 0, None)  # GENERIC_READ, share r/w, OPEN_EXISTING
    if h is None or h == wintypes.HANDLE(-1).value:
        e = ctypes.get_last_error()
        return e, "CreateFile: " + ctypes.FormatError(e).strip()
    try:
        if not k32.SetFilePointerEx(h, offset, None, 0):
            e = ctypes.get_last_error()
            return e, "SetFilePointerEx: " + ctypes.FormatError(e).strip()
        buf = ctypes.create_string_buffer(nbytes)
        got = wintypes.DWORD(0)
        if k32.ReadFile(h, buf, nbytes, ctypes.byref(got), None):
            return 0, f"ReadFile succeeded ({got.value} bytes) - the failure is intermittent"
        e = ctypes.get_last_error()
        return e, ctypes.FormatError(e).strip()
    finally:
        k32.CloseHandle(h)


def first_bad_offset(path, start, nbytes, chunk=65536):
    """Scan [start, start+nbytes) in 64 KB chunks; return the first offset whose read fails."""
    with open(path, "rb", buffering=0) as f:
        for o in range(start, start + nbytes, chunk):
            try:
                f.seek(o)
                f.read(min(chunk, start + nbytes - o))
            except OSError:
                return o
    return None


def ranges(rows):
    out, start = [], None
    for a, b in zip(rows, rows[1:] + [None]):
        start = a if start is None else start
        if b != a + 1:
            out.append((start, a))
            start = None
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--file", default=RESULTS_DIR / "lidc_patches" / "patches.npy", type=Path)
    ap.add_argument("--copy-to", type=Path, help="folder on another drive; patches.npy (+ nodules.csv) is copied there")
    ap.add_argument("--retries", type=int, default=3, help="re-reads of a failing row before giving up on it")
    ap.add_argument("--max-consecutive-bad", type=int, default=20,
                    help="stop after this many unreadable rows in a row (the cause is then not a few bad sectors)")
    ap.add_argument("--map-bad", action="store_true",
                    help="scan the whole file fast (one try per row, no waiting, no early stop) and list bad row ranges")
    args = ap.parse_args()
    if args.map_bad:
        args.retries, args.max_consecutive_bad = 1, 10 ** 9

    shape, dtype, off = npy_layout(args.file)
    row_bytes = int(np.prod(shape[1:])) * dtype.itemsize
    expected = off + shape[0] * row_bytes
    actual = args.file.stat().st_size
    print(f"{args.file}: {shape[0]} rows of {shape[1:]} {dtype}, {actual / 1e9:.2f} GB "
          f"(header says {expected / 1e9:.2f} GB){'' if actual == expected else '  <-- SIZE MISMATCH'}")

    dst = None
    if args.copy_to:
        args.copy_to.mkdir(parents=True, exist_ok=True)
        dst = open(args.copy_to / args.file.name, "wb")
        with open(args.file, "rb") as f:
            dst.write(f.read(off))  # .npy header

    bad, t0, errnos, run, diagnosed = [], time.time(), set(), 0, False
    with open(args.file, "rb", buffering=0) as f:
        for i in range(shape[0]):
            buf = None
            for attempt in range(args.retries):
                try:
                    f.seek(off + i * row_bytes)
                    buf = f.read(row_bytes)
                    if len(buf) == row_bytes:
                        break
                    buf = None
                except OSError as e:
                    errnos.add(e.errno)
                    if attempt == 0 and (len(bad) < 5 or not args.map_bad):
                        print(f"  row {i}: read error: {e}")
                    if not diagnosed:
                        diagnosed = True
                        o = first_bad_offset(args.file, off + i * row_bytes, row_bytes)
                        if o is not None:
                            print(f"  first unreadable byte offset: {o:,} ({o / 1e6:.1f} MB into the file; "
                                  f"{o - off - i * row_bytes:,} bytes into row {i})")
                        if os.name == "nt":
                            code, msg = win_read_error(args.file, o if o is not None else off + i * row_bytes, 65536)
                            print(f"  Windows error {code}: {msg}")
                            print(f"  -> {WIN_HINTS.get(code, 'see the Windows error code above')}")
                    if not args.map_bad:
                        time.sleep(1)
            if buf is None:
                bad.append(i)
                run += 1
                buf = b"\0" * row_bytes  # keep the copy aligned; the bad rows must be rebuilt
                if run >= args.max_consecutive_bad:
                    print(f"\nStopped: {run} consecutive unreadable rows (from row {i - run + 1}).")
                    break
            else:
                run = 0
            if dst:
                dst.write(buf)
            if (i + 1) % 200 == 0 or i + 1 == shape[0]:
                el = time.time() - t0
                print(f"  {i + 1}/{shape[0]} rows, {(i + 1) * row_bytes / 1e6 / el:.0f} MB/s, {len(bad)} bad", flush=True)
    if dst:
        dst.close()
        for extra in ("nodules.csv", "reader_marks.csv"):
            if (args.file.parent / extra).exists():
                shutil.copy2(args.file.parent / extra, args.copy_to / extra)

    if 13 in errnos:
        print("\n[Errno 13] on a READ: Python reports several Windows errors this way (lock, access denied by a\n"
              "security filter or sync client, ...). The 'Windows error' line above gives the real cause. If it\n"
              "persists after a reboot, the file region itself is the problem: rebuild the patches into a new\n"
              "file (04_lidc_patches.py --out D:\\paper8\\results\\lidc_patches_v2) and check the drive.")
    if bad:
        print(f"\n{len(bad)} UNREADABLE rows: {bad[:50]}{' ...' if len(bad) > 50 else ''}")
        print("Bad row ranges: " + ", ".join(f"{a}-{b}" if a != b else f"{a}" for a, b in ranges(bad)))
        if 13 not in errnos:
            print("Read errors other than a lock: the disk may have bad sectors in this file. Rebuild the patches\n"
                  "  python 04_lidc_patches.py --xml-root D:\\data\\LIDC-XML-only --out D:\\paper8\\results\\lidc_patches_v2\n"
                  "and check the drive (chkdsk D: /scan).")
        if dst:
            print(f"(The copy in {args.copy_to} has zeros in those rows - do not train from it.)")
    else:
        print("\nAll rows readable." + (f" Copy written to {args.copy_to}; train with --patches {args.copy_to}"
                                         if dst else ""))


if __name__ == "__main__":
    main()
