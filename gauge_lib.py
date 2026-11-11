"""
GAUGE-XAI shared constants and helpers (CT loading, patch sampling).

Patches are sampled on a 1 mm isotropic grid aligned with the patient (LPS) axes and centred on a
world coordinate, so LIDC training patches and RIDER evaluation patches are built the same way.
Axis order of a patch: (z, y, x) = (inferior->superior, anterior->posterior, right->left).

Heavy imports (pydicom, scipy) are done inside functions so that analysis scripts can import the
constants with only numpy/pandas installed.
"""
from pathlib import Path

import numpy as np

# All outputs go to <project>/results (next to the code folder), keeping code/ clean.
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

# ---- study design (pre-registered choices; see PROJECT_NOTES.md) ----------------------------
PATCH_SIZE = 128                # model input, voxels per side
PATCH_SPACING_MM = 1.0          # -> 128 mm field of view. 2 mm was too coarse: the median LIDC nodule
                                #    (8 mm) was 4 voxels wide and the model could not fit the training set
STORE_SIZE = 144                # stored LIDC patches: 128 + room for +-8 voxel (8 mm) shift augmentation
HU_WINDOW = (-1000.0, 400.0)    # clipped, then scaled to [0, 1]; stored as uint8 (~5.5 HU per step)
AIR_HU = -1024.0                # fill value outside the scanned volume

SETTINGS = ["1.25L", "1.25S", "2.5L", "2.5S", "5L", "5S"]
PRIMARY_SETTING = "1.25L"
SECONDARY_SETTINGS = {"2.5S"}   # most scan1 2.5S series have no tumour SEG (n ~ 10)
PRIMARY_EXCLUDE = {
    "RIDER-2539508879": "scan1 and scan2 are 2 days apart (not a same-day repeat)",
}


ROI_RADIUS_MM_128 = 60.0  # fixed LAF ROI for the 128 mm patch; scaled with the patch for other input sizes


def roi_radius_mm(n=PATCH_SIZE):
    return ROI_RADIUS_MM_128 * n / 128.0


def patch_roi(n=PATCH_SIZE, spacing=PATCH_SPACING_MM):
    """Boolean n^3 sphere (radius roi_radius_mm(n)) centred in the patch."""
    g = (np.arange(n) - (n - 1) / 2.0) * spacing
    zz, yy, xx = np.meshgrid(g, g, g, indexing="ij", sparse=True)
    return np.sqrt(zz ** 2 + yy ** 2 + xx ** 2) <= roi_radius_mm(n)


def setting_role(setting):
    return "secondary" if setting in SECONDARY_SETTINGS else "primary"


def load_ct(folder, series_uid=None):
    """Read one CT series into HU, sorted along the slice normal. Keeps SOP UIDs for annotation lookup."""
    import pydicom

    slices = []
    for p in Path(folder).iterdir():
        if p.name.startswith(".") or not p.is_file():
            continue
        try:
            ds = pydicom.dcmread(p)
        except Exception:
            continue
        if series_uid is not None and getattr(ds, "SeriesInstanceUID", None) != series_uid:
            continue
        if hasattr(ds, "ImagePositionPatient") and hasattr(ds, "PixelData"):
            slices.append(ds)
    if not slices:
        raise RuntimeError(f"no CT slices found in {folder}")
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
    return dict(vol=vol, ipps=ipps, zs=zs, dz=abs(dz), ps=ps, row_dir=row_dir, col_dir=col_dir,
                normal=normal, iop=iop, sop_uids=[str(d.SOPInstanceUID) for d in slices],
                patient_id=str(getattr(slices[0], "PatientID", "")))


def pixel_to_world(ct, k, r, c):
    """Voxel (slice k, row r, column c) -> world mm. Accepts arrays; uses each slice's own IPP."""
    k = np.asarray(k)
    ipp = ct["ipps"][np.clip(np.rint(k).astype(int), 0, len(ct["ipps"]) - 1)]
    return (ipp + np.asarray(c)[..., None] * ct["ps"][1] * ct["row_dir"]
            + np.asarray(r)[..., None] * ct["ps"][0] * ct["col_dir"])


def world_to_voxel(ct, pts):
    """World mm (..., 3) -> fractional (k, r, c). Slice index is interpolated along the sorted normal."""
    pts = np.asarray(pts, float)
    d = pts - ct["ipps"][0]
    c = d @ ct["row_dir"] / ct["ps"][1]
    r = d @ ct["col_dir"] / ct["ps"][0]
    zn = pts @ ct["normal"]
    idx = np.arange(len(ct["zs"]), dtype=float)
    k = np.interp(zn, ct["zs"], idx, left=np.nan, right=np.nan)
    # outside the scanned range: extrapolate with the median spacing (filled with cval later)
    lo, hi = zn < ct["zs"][0], zn > ct["zs"][-1]
    k = np.where(lo, (zn - ct["zs"][0]) / ct["dz"], k)
    k = np.where(hi, len(ct["zs"]) - 1 + (zn - ct["zs"][-1]) / ct["dz"], k)
    return k, r, c


def patch_grid(center_xyz, size=PATCH_SIZE, spacing=PATCH_SPACING_MM):
    """World coordinates (size, size, size, 3) of a patch grid, axis order (z, y, x)."""
    off = (np.arange(size) - (size - 1) / 2.0) * spacing
    zz, yy, xx = np.meshgrid(off, off, off, indexing="ij")
    return np.stack([xx, yy, zz], -1) + np.asarray(center_xyz, float)


def extract_patch(ct, center_xyz, size=PATCH_SIZE, spacing=PATCH_SPACING_MM, volume=None, order=1, cval=AIR_HU):
    """Resample ct['vol'] (or another array on the CT grid, e.g. a mask) onto the patch grid."""
    from scipy.ndimage import map_coordinates

    vol = ct["vol"] if volume is None else volume
    k, r, c = world_to_voxel(ct, patch_grid(center_xyz, size, spacing))
    return map_coordinates(vol.astype(np.float32, copy=False), [k, r, c], order=order,
                           mode="constant", cval=cval).astype(np.float32)


def normalize_hu(x):
    lo, hi = HU_WINDOW
    return (np.clip(x, lo, hi) - lo) / (hi - lo)


def npy_layout(path):
    """(shape, dtype, data offset in bytes) of a .npy file, without reading the data."""
    with open(path, "rb") as f:
        version = np.lib.format.read_magic(f)
        # public readers (the private _read_array_header was removed in numpy 2.x)
        read_header = (np.lib.format.read_array_header_1_0 if version == (1, 0)
                       else np.lib.format.read_array_header_2_0)
        shape, fortran, dtype = read_header(f)
        if fortran:
            raise ValueError(f"{path}: Fortran-ordered arrays are not supported")
        return shape, dtype, f.tell()


def read_npy_rows(path, idx, retries=3):
    """Rows idx of a large .npy with ordinary file reads (no memory map). A failing disk then raises an
    OSError naming the row, instead of crashing the process ("Windows fatal exception: page error").
    Each failed read is retried (a drive that briefly drops out usually succeeds the second time)."""
    import time

    shape, dtype, off = npy_layout(path)
    row_bytes = int(np.prod(shape[1:])) * dtype.itemsize
    idx = np.asarray(idx, dtype=np.int64)
    out = np.empty((len(idx),) + tuple(shape[1:]), dtype=dtype)
    with open(path, "rb", buffering=0) as f:
        for j, i in enumerate(idx):
            if not 0 <= i < shape[0]:
                raise IndexError(f"row {i} outside {path} ({shape[0]} rows)")
            for attempt in range(retries):
                try:
                    f.seek(off + int(i) * row_bytes)
                    buf = f.read(row_bytes)
                    break
                except OSError as e:
                    if attempt == retries - 1:
                        hint = ("another process holds a lock on the file: close other Python runs or reboot"
                                if e.errno == 13 else "run verify_patches.py")
                        raise OSError(f"{path}: disk read failed at row {i} after {retries} tries ({e}); "
                                      f"{hint}") from e
                    print(f"  read error at row {i} (attempt {attempt + 1}), retrying: {e}", flush=True)
                    time.sleep(2)
            if len(buf) != row_bytes:
                raise OSError(f"{path}: row {i} is truncated ({len(buf)} of {row_bytes} bytes)")
            out[j] = np.frombuffer(buf, dtype=dtype).reshape(shape[1:])
    return out


def to_uint8(x01):
    return np.clip(np.rint(np.asarray(x01) * 255.0), 0, 255).astype(np.uint8)


def from_uint8(x):
    """Stored patch (uint8, or float in [0, 1] from older runs) -> float32 in [0, 1]."""
    x = np.asarray(x)
    return x.astype(np.float32) / 255.0 if x.dtype == np.uint8 else x.astype(np.float32)


def center_crop(x, size=PATCH_SIZE):
    """Central size^3 crop of the last three axes (STORE_SIZE patch -> model input)."""
    o = [(s - size) // 2 for s in x.shape[-3:]]
    return x[..., o[0]:o[0] + size, o[1]:o[1] + size, o[2]:o[2] + size]
