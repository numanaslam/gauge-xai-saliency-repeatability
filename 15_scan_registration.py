#!/usr/bin/env python3
"""
GAUGE-XAI step 15 (journal): registration-based analyses.

(a) Scan 1 <-> scan 2 (same reconstruction): the two patches are centred on each scan's own SEG centroid, so
    they differ by a residual shift (contour differences, breathing). The shift is found by a coarse-to-fine integer
    search (|d| <= --local-search-mm per axis) maximising the CT NCC in the tumour neighbourhood (both SEGs dilated
    by --margin-mm); zero shift is a candidate. A whole-patch phase correlation is reported for comparison only: on
    RIDER it follows the chest anatomy, not the tumour (SEG Dice fell after it in 69% of pairs).
    Reported per explainer and run: voxel-level map agreement between scans (Spearman of positive attribution
    in the ROI, at 1 mm and pooled to 8^3 cells; top-10% Dice), and LAF of the aligned scan-2 map computed with
    the SCAN-1 SEG (fixed mask). Comparing the scan-rescan ICC/wSD of fixed-mask LAF with own-mask LAF separates
    explainer variability from contour variability. Registration quality: CT NCC in the tumour neighbourhood and SEG Dice before/after;
    map agreement is reported both unregistered (centroid-aligned) and registered.
(b) Reconstructions (same scan): the 1.25L SEG of the same acquisition is moved into every reconstruction's patch
    by the centroid difference (same frame of reference) and used as a fixed reference mask, so kernel/thickness
    agreement of LAF can be computed without per-series contour differences.

Needs 06 run with --save-patches (patches/, masks/, maps/). Writes <out>/scan_pairs.csv (one row per
patient x reconstruction x run x explainer), recon_fixed_mask.csv, registration_summary.csv and recon_fixed_mask_summary.csv.

Usage:
  python 15_scan_registration.py --saliency ..\\results\\saliency --out ..\\results\\analysis\\registration
"""
import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

from gauge_lib import PATCH_SPACING_MM, PRIMARY_EXCLUDE, RESULTS_DIR, SECONDARY_SETTINGS, SETTINGS, from_uint8

P = [s for s in SETTINGS if s not in SECONDARY_SETTINGS]


def load_by_path(name, fname):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(fname))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


S7 = load_by_path("gauge_analysis", "07_gauge_analysis.py")  # load_map, _roi, align_to, map_similarity, pool8, roi8


def phase_shift(a, b, max_shift):
    """Integer shift d (z, y, x) such that a(x) ~ b(x - d); i.e. scipy.ndimage.shift(b, d) aligns b to a."""
    w = np.outer(np.hanning(a.shape[0]), np.hanning(a.shape[1]))[:, :, None] * np.hanning(a.shape[2])[None, None, :]
    fa, fb = np.fft.fftn((a - a.mean()) * w), np.fft.fftn((b - b.mean()) * w)
    r = fa * np.conj(fb)
    r /= np.abs(r) + 1e-12
    c = np.real(np.fft.ifftn(r))
    n = np.array(a.shape)
    grid = np.meshgrid(*[np.fft.fftfreq(k, 1.0 / k) for k in a.shape], indexing="ij")  # signed shifts
    ok = (np.abs(grid[0]) <= max_shift) & (np.abs(grid[1]) <= max_shift) & (np.abs(grid[2]) <= max_shift)
    c = np.where(ok, c, -np.inf)
    peak = np.array(np.unravel_index(np.argmax(c), c.shape))
    return np.where(peak > n // 2, peak - n, peak)


def shift_int(v, d, cval=0.0):
    out = np.full_like(v, cval)
    src = tuple(slice(max(0, -k), v.shape[i] - max(0, k)) for i, k in enumerate(d))
    dst = tuple(slice(max(0, k), v.shape[i] - max(0, -k)) for i, k in enumerate(d))
    out[dst] = v[src]
    return out


def ncc(a, b, m):
    x, y = a[m] - a[m].mean(), b[m] - b[m].mean()
    return float((x * y).sum() / np.sqrt((x * x).sum() * (y * y).sum() + 1e-12))


def lesion_region(mask, margin_mm):
    """Both scans' SEGs (on the shared centroid-centred grid) dilated by margin_mm: the tumour neighbourhood."""
    from scipy.ndimage import binary_dilation, generate_binary_structure, iterate_structure

    k = max(1, int(round(margin_mm / PATCH_SPACING_MM)))
    return binary_dilation(mask, iterate_structure(generate_binary_structure(3, 1), k))


def local_shift(a, b, region, max_shift):
    """Integer search |d_i| <= max_shift (2-voxel grid, then +-1 refinement) for the shift of b that maximises the CT NCC with a inside
    region (tumour neighbourhood). Zero shift (the centroid alignment) is a candidate, so it is never worse.
    Returns (d, ncc before, ncc after)."""
    idx = np.argwhere(region)
    lo, hi = np.maximum(idx.min(0) - max_shift, 0), np.minimum(idx.max(0) + max_shift + 1, a.shape)
    sl = tuple(slice(l, h) for l, h in zip(lo, hi))
    A, B, R = a[sl], b[sl], region[sl]
    n0 = ncc(A, B, R)

    def search(cands):
        best, best_d = -np.inf, np.zeros(3, int)
        for d in sorted(cands, key=lambda d: np.abs(d).sum()):  # ties -> smallest shift
            v = ncc(A, shift_int(B, np.array(d), B.min()), R)
            if v > best + 1e-9:
                best, best_d = v, np.array(d)
        return best_d, best

    r = range(-max_shift, max_shift + 1, 2)  # coarse (2 mm) grid, then +-1 voxel refinement
    d0, _ = search([(z, y, x) for z in r for y in r for x in r] + [(0, 0, 0)])
    best_d, best = search([tuple(np.clip(d0 + (z, y, x), -max_shift, max_shift)) for z in (-1, 0, 1)
                           for y in (-1, 0, 1) for x in (-1, 0, 1)] + [(0, 0, 0)])
    return best_d, n0, float(best)


def laf(a, mask, roi):
    pos = np.clip(a, 0, None) * roi
    t = pos.sum()
    return float(pos[mask].sum() / t) if t > 0 else np.nan


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--saliency", default=RESULTS_DIR / "saliency", type=Path)
    ap.add_argument("--out", default=RESULTS_DIR / "analysis" / "registration", type=Path)
    ap.add_argument("--max-shift-mm", type=float, default=20.0, help="whole-patch phase correlation (reported only)")
    ap.add_argument("--local-search-mm", type=float, default=5.0, help="tumour-local exhaustive search range")
    ap.add_argument("--margin-mm", type=float, default=10.0, help="SEG dilation defining the tumour neighbourhood")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    sal = args.saliency
    explainers = sorted(d.name for d in (sal / "maps").iterdir() if d.is_dir())
    seeds = sorted({int(p.name.replace("seed", "")) for e in explainers for p in (sal / "maps" / e).glob("seed*")})
    cells = {p.stem for p in (sal / "patches").glob("*.npz")}
    if not cells:
        raise SystemExit("no patches/ - run 06_rider_saliency.py with --save-patches first")
    split = [c.rsplit("_", 2) for c in cells]
    pairs, recon = [], []

    # (a) scan 1 <-> scan 2
    for pid, setting in sorted({(p, s) for p, a, s in split}):
        n1, n2 = f"{pid}_scan1_{setting}", f"{pid}_scan2_{setting}"
        if n1 not in cells or n2 not in cells:
            continue
        c1, c2 = from_uint8(np.load(sal / "patches" / f"{n1}.npz")["patch"]), from_uint8(np.load(sal / "patches" / f"{n2}.npz")["patch"])
        m1, m2 = np.load(sal / "masks" / f"{n1}.npz")["mask"].astype(bool), np.load(sal / "masks" / f"{n2}.npz")["mask"].astype(bool)
        n = c1.shape[0]
        roi = S7._roi(n)
        dg = phase_shift(c1, c2, int(round(args.max_shift_mm / PATCH_SPACING_MM)))  # whole patch (reported only)
        region = lesion_region(m1 | m2, args.margin_mm)
        d, ncc_b, ncc_a = local_shift(c1, c2, region, int(round(args.local_search_mm / PATCH_SPACING_MM)))
        m2a = shift_int(m2, d, False)
        dice = lambda a, b: float(2 * (a & b).sum() / (a.sum() + b.sum()))  # noqa: E731
        base = dict(PatientID=pid, setting=setting, shift_mm=float(np.linalg.norm(d) * PATCH_SPACING_MM),
                    global_shift_mm=float(np.linalg.norm(dg) * PATCH_SPACING_MM),
                    ncc_lesion_before=ncc_b, ncc_lesion_after=ncc_a,
                    ncc_lesion_global=ncc(c1, shift_int(c2, dg, c2.min()), region),
                    seg_dice_before=dice(m1, m2), seg_dice_after=dice(m1, m2a),
                    seg_dice_global=dice(m1, shift_int(m2, dg, False)))
        roi8 = S7.roi8(n)
        for e in explainers:
            for seed in seeds:
                a1 = S7.load_map(sal, e, seed, n1, (n,) * 3)
                a2 = S7.load_map(sal, e, seed, n2, (n,) * 3)
                if a1 is None or a2 is None:
                    continue
                a2a = shift_int(a2, d)
                rho0, dice0 = S7.map_similarity(a1, a2, roi)
                rho80, dice80 = S7.map_similarity(S7.pool8(a1), S7.pool8(a2), roi8)
                rho, dice_ = S7.map_similarity(a1, a2a, roi)
                rho8, dice8 = S7.map_similarity(S7.pool8(a1), S7.pool8(a2a), roi8)
                rho_l, _ = S7.map_similarity(a1, a2a, region)  # lesion neighbourhood only
                pairs.append(dict(base, explainer=e, seed=seed,
                                  spearman_unregistered=rho0, dice_top10_unregistered=dice0,
                                  spearman_8_unregistered=rho80, dice_top10_8_unregistered=dice80,
                                  spearman=rho, dice_top10=dice_, spearman_8=rho8, dice_top10_8=dice8,
                                  spearman_lesion_region=rho_l,
                                  laf1=laf(a1, m1, roi), laf2_own=laf(a2, m2, roi), laf2_fixed=laf(a2a, m1, roi)))

    # (b) fixed 1.25L reference mask across reconstructions of the same scan
    for pid, acq in sorted({(p, a) for p, a, s in split}):
        ref = f"{pid}_{acq}_1.25L"
        if ref not in cells:
            continue
        zr = np.load(sal / "masks" / f"{ref}.npz")
        mref, cref = zr["mask"].astype(np.float32), zr["centroid"]
        for setting in SETTINGS:
            name = f"{pid}_{acq}_{setting}"
            if name not in cells:
                continue
            z = np.load(sal / "masks" / f"{name}.npz")
            m_own, c_own = z["mask"].astype(bool), z["centroid"]
            m_ref = S7.align_to(mref, c_own, cref, PATCH_SPACING_MM) > 0.5  # reference SEG on this cell's grid
            n = m_own.shape[0]
            roi = S7._roi(n)
            for e in explainers:
                for seed in seeds:
                    a = S7.load_map(sal, e, seed, name, (n,) * 3)
                    if a is None:
                        continue
                    recon.append(dict(PatientID=pid, acq=acq, setting=setting, explainer=e, seed=seed,
                                      laf_own=laf(a, m_own, roi), laf_refmask=laf(a, m_ref, roi),
                                      mask_dice_own_vs_ref=float(2 * (m_own & m_ref).sum() / (m_own.sum() + m_ref.sum()))))
    sp = pd.DataFrame(pairs)
    rc = pd.DataFrame(recon)
    sp.round(5).to_csv(args.out / "scan_pairs.csv", index=False)
    rc.round(5).to_csv(args.out / "recon_fixed_mask.csv", index=False)

    # ---- summary ------------------------------------------------------------------------------
    rows = []
    prim = sp[~sp.PatientID.isin(PRIMARY_EXCLUDE) & sp.setting.isin(P)]
    for (e, seed), g in prim.groupby(["explainer", "seed"]):
        r = dict(explainer=e, seed=seed, n_pairs=len(g))
        for c in ("shift_mm", "global_shift_mm", "ncc_lesion_before", "ncc_lesion_after", "ncc_lesion_global",
                  "seg_dice_before", "seg_dice_after", "seg_dice_global", "spearman_unregistered", "spearman",
                  "spearman_8_unregistered", "spearman_8", "dice_top10_unregistered", "dice_top10",
                  "spearman_lesion_region"):
            r[c] = g[c].median()
        r["share_zero_shift"] = float((g.shift_mm == 0).mean())
        for lab, col in (("own", "laf2_own"), ("fixed", "laf2_fixed")):
            y = g[["laf1", col]].dropna().to_numpy(float)
            if len(y) >= 5:
                icc, wsd = S7.PC.icc_1_1(y)
                r[f"laf_icc_{lab}"], r[f"laf_wsd_{lab}"] = icc, wsd
        rows.append(r)
    sm = pd.DataFrame(rows)
    rows = []
    rprim = rc[~rc.PatientID.isin(PRIMARY_EXCLUDE) & (rc.acq == "scan1")]
    for (e, seed), g in rprim.groupby(["explainer", "seed"]):
        out = dict(explainer=e, seed=seed, n_patients=g.PatientID.nunique())
        for lab, col in (("own", "laf_own"), ("refmask", "laf_refmask")):
            w = g.pivot_table(index="PatientID", columns="setting", values=col)
            vals = []
            for s_ in [s for s in P if s != "1.25L"]:
                if "1.25L" in w and s_ in w:
                    y = w[["1.25L", s_]].dropna().to_numpy(float)
                    if len(y) >= 5:
                        vals.append(S7.PC.icc_1_1(y)[1])
            out[f"wsd_vs_1.25L_{lab}"] = float(np.mean(vals)) if vals else np.nan
        out["mask_dice_own_vs_ref"] = g.mask_dice_own_vs_ref.median()
        rows.append(out)
    rs = pd.DataFrame(rows)
    sm.round(4).to_csv(args.out / "registration_summary.csv", index=False)
    rs.round(4).to_csv(args.out / "recon_fixed_mask_summary.csv", index=False)
    with pd.option_context("display.width", 230, "display.max_columns", 30):
        print("scan 1 <-> scan 2 (registered), primary population and reconstructions:")
        print(sm.round(3).to_string(index=False))
        print("\nreconstruction vs 1.25L, own SEG vs fixed 1.25L reference SEG (scan 1):")
        print(rs.round(3).to_string(index=False))
    print(f"\nWrote {args.out}: scan_pairs.csv, recon_fixed_mask.csv, registration_summary.csv, recon_fixed_mask_summary.csv")


if __name__ == "__main__":
    main()
