#!/usr/bin/env python3
"""
GAUGE-XAI step 7: gauge statistics for the saliency measurands (and the logit), plus paper figures.

Reads ../results/saliency/measurands.csv (06) and writes ../results/analysis/:
  repeatability.csv   scan-rescan, per explainer x seed x setting x measurand x population:
                      ICC(1,1) + patient-bootstrap 95% CI, wSD, RC = 2.77 wSD, Bland-Altman, RC / median,
                      Koo & Li class and AIAG %GRR class, each with a CI-based "conclusive" flag
  conditions.csv      agreement between reconstructions within scan1 (kernel and thickness contrasts)
  seeds.csv           agreement between training seeds on the same cell (needs >= 2 seeds)
  stability.csv       model vs explainer instability: Spearman(d logit, d LAF), ICC of LAF for pairs
                      whose logit changed little vs a lot
  map_conditions.csv  map-level agreement between reconstructions of the SAME acquisition (same frame of
                      reference, so no registration; maps aligned on the SEG centroids): Spearman rho of the
                      positive attribution in the ROI and Dice of the top-10% voxels. LAF only sees the lesion;
                      this sees the whole map.
  Every ICC row also has a leave-one-patient-out range and the most influential patient.
  summary.txt         primary endpoint first, then the key tables in plain text
  fig_icc_forest.png, fig_dlogit_dlaf.png, fig_bland_altman_primary.png,
  fig_lesion_<explainer>.png (2 x 6: scan x reconstruction, one lesion)

Primary endpoint (pre-registered): IG LAF at 1.25L, primary population (PRIMARY_EXCLUDE removed).
Thresholds: Koo & Li 2016 ICC < 0.5 poor, 0.5-0.75 moderate, 0.75-0.9 good, > 0.9 excellent.
AIAG gauge R&R: %GRR = 100 sqrt(1 - ICC) (repeatability SD / total SD); < 10% acceptable,
10-30% marginal, > 30% unacceptable. A class is "conclusive" only if the whole CI lies in it.

Usage:
  python 07_gauge_analysis.py
  python 07_gauge_analysis.py --figure-patient RIDER-1500037140 --n-boot 2000
"""
import argparse
import importlib.util
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from gauge_lib import (PRIMARY_EXCLUDE, PRIMARY_SETTING, RESULTS_DIR, SETTINGS, extract_patch,  # noqa: E402
                       load_ct, setting_role)

MEASURES = ["laf", "laf_sphere_patient", "laf_sphere10", "laf_enrichment", "com_offset_rel", "entropy_norm",
            "logit"]
CONTRASTS = [("1.25L", "1.25S"), ("2.5L", "2.5S"), ("5L", "5S"),          # kernel
             ("1.25L", "2.5L"), ("1.25L", "5L"), ("1.25S", "5S")]          # thickness
KOO = [(0.5, "poor"), (0.75, "moderate"), (0.9, "good"), (np.inf, "excellent")]
AIAG = [(10, "acceptable"), (30, "marginal"), (np.inf, "unacceptable")]


def load_by_path(name, fname):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(fname))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


PC = load_by_path("positive_control", "03_positive_control.py")  # icc_1_1, summarise, bootstrap


def klass(v, table):
    for t, name in table:
        if v < t:
            return name
    return table[-1][1]


def verdicts(icc, lo, hi):
    grr = lambda i: 100 * np.sqrt(max(0.0, 1 - i))  # noqa: E731
    out = dict(koo=klass(icc, KOO), koo_conclusive=klass(lo, KOO) == klass(hi, KOO),
               pct_grr=grr(icc), pct_grr_lo=grr(hi), pct_grr_hi=grr(lo))
    out["aiag"] = klass(out["pct_grr"], AIAG)
    out["aiag_conclusive"] = klass(out["pct_grr_lo"], AIAG) == klass(out["pct_grr_hi"], AIAG)
    return out


def leave_one_out(y, ids):
    """ICC with each patient removed in turn: range, and the patient whose removal moves the ICC most."""
    ids = np.asarray(ids)
    full, _ = PC.icc_1_1(y)
    vals = {}
    for p in np.unique(ids):
        keep = ids != p
        if keep.sum() >= 4:
            vals[p] = PC.icc_1_1(y[keep])[0]
    if not vals:
        return {}
    who = max(vals, key=lambda p: abs(vals[p] - full))
    return dict(icc_loo_min=min(vals.values()), icc_loo_max=max(vals.values()), loo_most_influential=who,
                icc_without_most_influential=vals[who])


def agreement(y, n_boot, seed=0, ids=None):
    """y: (n, k) replicate matrix -> ICC, CI, wSD, RC (+ BA for k = 2); ids (patient per row) adds a
    leave-one-patient-out influence check."""
    icc, wsd = PC.icc_1_1(y)
    (ilo, ihi), (rlo, rhi) = PC.bootstrap(y, "linear", n_boot, seed)
    out = dict(n=len(y), icc=icc, icc_ci_lo=ilo, icc_ci_hi=ihi, wSD=wsd, RC=2.77 * wsd, RC_ci_lo=rlo, RC_ci_hi=rhi,
               median=float(np.median(y)), between_sd=float(np.std(y.mean(1), ddof=1)))
    out["RC_over_median"] = out["RC"] / abs(out["median"]) if out["median"] else np.nan
    if y.shape[1] == 2:
        d = y[:, 1] - y[:, 0]
        out.update(BA_mean=d.mean(), BA_loa_lo=d.mean() - 1.96 * d.std(ddof=1), BA_loa_hi=d.mean() + 1.96 * d.std(ddof=1))
    out.update(verdicts(icc, ilo, ihi))
    if ids is not None:
        out.update(leave_one_out(y, ids))
    return out


def pairs(df, measure, a, b, key):
    """Wide table of measure for replicate labels a, b of column key, one row per patient (NaNs dropped)."""
    w = df.pivot_table(index="PatientID", columns=key, values=measure, aggfunc="first")
    if a not in w or b not in w:
        return None, None
    w = w[[a, b]].dropna()
    return w.index.to_numpy(), w.to_numpy(float)


def repeatability(m, n_boot):
    rows = []
    for pop, d in (("primary", m[~m.PatientID.isin(PRIMARY_EXCLUDE)]), ("all", m)):
        for (expl, seed, setting), g in d.groupby(["explainer", "seed", "setting"]):
            for meas in MEASURES:
                if meas not in g:
                    continue
                ids, y = pairs(g, meas, "scan1", "scan2", "acq")
                if y is None or len(y) < 5:
                    continue
                rows.append(dict(population=pop, explainer=expl, seed=seed, setting=setting,
                                 setting_role=setting_role(setting), measure=meas, **agreement(y, n_boot, ids=ids)))
    return pd.DataFrame(rows)


def conditions(m, n_boot):
    """Agreement between reconstructions within scan1. exceed_rc = fraction of lesions whose change between
    the two reconstructions is larger than the scan-rescan RC of the first one (robust to single outliers)."""
    rows = []
    prim = m[~m.PatientID.isin(PRIMARY_EXCLUDE)]
    d = prim[prim.acq == "scan1"]
    for (expl, seed), g in d.groupby(["explainer", "seed"]):
        gall = prim[(prim.explainer == expl) & (prim.seed == seed)]
        for a, b in CONTRASTS:
            for meas in ("laf", "laf_sphere_patient", "logit"):
                ids, y = pairs(g, meas, a, b, "setting")
                if y is None or len(y) < 5:
                    continue
                kind = "kernel" if a[:-1] == b[:-1] else "thickness"
                _, ysr = pairs(gall[gall.setting == a], meas, "scan1", "scan2", "acq")
                rc = 2.77 * PC.icc_1_1(ysr)[1] if ysr is not None and len(ysr) >= 5 else np.nan
                exceed = float(np.mean(np.abs(y[:, 1] - y[:, 0]) > rc)) if np.isfinite(rc) else np.nan
                rows.append(dict(explainer=expl, seed=seed, contrast=f"{a} vs {b}", kind=kind, measure=meas,
                                 rc_scan_rescan=rc, exceed_rc=exceed, **agreement(y, n_boot, ids=ids)))
    return pd.DataFrame(rows)


def seeds(m, n_boot):
    if m.seed.nunique() < 2:
        return pd.DataFrame()
    rows = []
    d = m[~m.PatientID.isin(PRIMARY_EXCLUDE)]
    for (expl, setting), g in d.groupby(["explainer", "setting"]):
        for meas in ("laf", "logit"):
            w = g.pivot_table(index=["PatientID", "acq"], columns="seed", values=meas).dropna()
            if len(w) < 5:
                continue
            rows.append(dict(explainer=expl, setting=setting, measure=meas, n_seeds=w.shape[1],
                             **agreement(w.to_numpy(float), n_boot, ids=w.index.get_level_values(0))))
    return pd.DataFrame(rows)


def stability(m, n_boot):
    from scipy.stats import spearmanr

    rows, pts = [], []
    d = m[~m.PatientID.isin(PRIMARY_EXCLUDE)]
    for (expl, seed), g in d.groupby(["explainer", "seed"]):
        diffs = []
        for setting, gs in g.groupby("setting"):
            w = gs.pivot_table(index="PatientID", columns="acq", values=["laf", "logit"])
            if ("laf", "scan1") not in w or ("laf", "scan2") not in w:
                continue
            w = w.dropna()
            diffs.append(pd.DataFrame(dict(PatientID=w.index, setting=setting,
                                           laf1=w[("laf", "scan1")], laf2=w[("laf", "scan2")],
                                           d_logit=w[("logit", "scan2")] - w[("logit", "scan1")],
                                           d_laf=w[("laf", "scan2")] - w[("laf", "scan1")])))
        if not diffs:
            continue
        dd = pd.concat(diffs, ignore_index=True)
        dd["explainer"], dd["seed"] = expl, seed
        pts.append(dd)
        rho, p = spearmanr(dd["d_logit"].abs(), dd["d_laf"].abs())
        cut = dd["d_logit"].abs().median()
        row = dict(explainer=expl, seed=seed, n_pairs=len(dd), spearman_absdlogit_absdlaf=rho, p=p,
                   median_abs_dlogit=cut)
        for name, sub in (("stable", dd[dd.d_logit.abs() <= cut]), ("unstable", dd[dd.d_logit.abs() > cut])):
            if len(sub) >= 5:
                a = agreement(sub[["laf1", "laf2"]].to_numpy(float), n_boot)
                row.update({f"icc_laf_{name}": a["icc"], f"icc_laf_{name}_lo": a["icc_ci_lo"],
                            f"icc_laf_{name}_hi": a["icc_ci_hi"], f"n_{name}": a["n"]})
        rows.append(row)
    return pd.DataFrame(rows), (pd.concat(pts, ignore_index=True) if pts else pd.DataFrame())


# ---- map-level agreement between reconstructions --------------------------------------------------
def load_map(sal, expl, seed, name, shape):
    import torch
    import torch.nn.functional as F

    p = sal / "maps" / expl / f"seed{seed}" / f"{name}.npz"
    if not p.exists():
        return None
    a = np.load(p)["attr"].astype(np.float32)
    if a.shape != shape:  # Grad-CAM is stored at 8^3
        a = F.interpolate(torch.from_numpy(a)[None, None], size=shape, mode="trilinear",
                          align_corners=False)[0, 0].numpy()
    return a


def align_to(b, c_a, c_b, spacing):
    """Resample map b (patch centred at world c_b) onto the patch grid centred at c_a."""
    from scipy.ndimage import shift

    s_xyz = (np.asarray(c_b) - np.asarray(c_a)) / spacing
    return shift(b, s_xyz[::-1], order=1, mode="constant", cval=0.0)  # patch axes are (z, y, x)


def map_similarity(a, b, roi, top=0.10):
    """Spearman rho and top-10% Dice of the positive attribution inside roi (a and b on the same grid)."""
    from scipy.stats import spearmanr

    pa, pb = np.clip(a[roi], 0, None), np.clip(b[roi], 0, None)
    if pa.max() <= 0 or pb.max() <= 0:
        return np.nan, np.nan
    rho = spearmanr(pa, pb).correlation
    k = max(1, int(top * roi.sum()))
    ta = np.zeros(len(pa), bool)
    tb = np.zeros(len(pb), bool)
    ta[np.argpartition(-pa, k - 1)[:k]] = True
    tb[np.argpartition(-pb, k - 1)[:k]] = True
    return float(rho), float(2 * (ta & tb).sum() / (ta.sum() + tb.sum()))


GRID8 = 8  # maps are pooled to 8^3 cells (the layer4 Grad-CAM grid) for a resolution-matched comparison


def pool8(a):
    p = a.shape[0] // GRID8
    return np.clip(a, 0, None).reshape(GRID8, p, GRID8, p, GRID8, p).mean((1, 3, 5))


def roi8(n):
    from gauge_lib import PATCH_SPACING_MM, roi_radius_mm

    cell = n // GRID8 * PATCH_SPACING_MM
    g = (np.arange(GRID8) - (GRID8 - 1) / 2.0) * cell
    zz, yy, xx = np.meshgrid(g, g, g, indexing="ij", sparse=True)
    return np.sqrt(zz ** 2 + yy ** 2 + xx ** 2) <= roi_radius_mm(n)


def map_conditions(sal, m, contrasts=CONTRASTS):
    from gauge_lib import PATCH_SPACING_MM

    first = next((sal / "masks").glob("*.npz"), None)
    if first is None:
        return pd.DataFrame()
    n = np.load(first)["mask"].shape[0]  # 128 (main model) or 64 (ablation)
    shape, roi, r8, rows = (n,) * 3, _roi(n), roi8(n), []
    d = m[~m.PatientID.isin(PRIMARY_EXCLUDE)]
    cent = {}
    for (pid, acq), g in d.groupby(["PatientID", "acq"]):
        have = set(g.setting)
        for expl, seed in g[["explainer", "seed"]].drop_duplicates().itertuples(index=False):
            cache = {}
            for a, b in contrasts:
                if a not in have or b not in have:
                    continue
                for s_ in (a, b):
                    name = f"{pid}_{acq}_{s_}"
                    if s_ not in cache:
                        cache[s_] = load_map(sal, expl, seed, name, shape)
                    if name not in cent:
                        mk = sal / "masks" / f"{name}.npz"
                        cent[name] = np.load(mk)["centroid"] if mk.exists() else None
                ma, mb = cache[a], cache[b]
                ca, cb = cent[f"{pid}_{acq}_{a}"], cent[f"{pid}_{acq}_{b}"]
                if ma is None or mb is None or ca is None or cb is None:
                    continue
                mb_al = align_to(mb, ca, cb, PATCH_SPACING_MM)
                rho, dice = map_similarity(ma, mb_al, roi)
                rho8, dice8 = map_similarity(pool8(ma), pool8(mb_al), r8)
                rows.append(dict(explainer=expl, seed=seed, PatientID=pid, acq=acq, contrast=f"{a} vs {b}",
                                 kind="kernel" if a[:-1] == b[:-1] else "thickness", spearman=rho,
                                 dice_top10=dice, spearman_8=rho8, dice_top10_8=dice8,
                                 centroid_shift_mm=float(np.linalg.norm(np.asarray(ca) - cb))))
    return pd.DataFrame(rows)


# ---- figures ------------------------------------------------------------------------------------
def fig_forest(rep, out_png, seed):
    if rep.empty:
        return
    r = rep[(rep.population == "primary") & (rep.seed == seed) & rep.measure.isin(["laf", "laf_sphere_patient", "logit"])]
    if r.empty:
        return
    labels = {"laf": "LAF (SEG)", "laf_sphere_patient": "LAF (fixed sphere)", "logit": "logit"}
    groups = [(e, ms) for e in sorted(r.explainer.unique()) for ms in ("laf", "laf_sphere_patient")]
    groups.append((sorted(r.explainer.unique())[0], "logit"))  # the logit is the same for every explainer
    fig, axes = plt.subplots(1, len(groups), figsize=(2.6 * len(groups), 3.6), sharey=True)
    for ax, (e, ms) in zip(np.ravel(axes), groups):
        g = r[(r.explainer == e) & (r.measure == ms)].set_index("setting").reindex(SETTINGS)
        y = np.arange(len(SETTINGS))[::-1]
        for lo, hi, c in ((0, 0.5, "#f3d6d6"), (0.5, 0.75, "#f6e8c8"), (0.75, 0.9, "#e3eed6"), (0.9, 1, "#cfe6cf")):
            ax.axvspan(lo, hi, color=c, lw=0)
        ax.errorbar(g["icc"], y, xerr=[g["icc"] - g["icc_ci_lo"], g["icc_ci_hi"] - g["icc"]], fmt="o", color="k",
                    ms=4, capsize=2)
        ax.set_yticks(y, [f"{s}{'*' if setting_role(s) == 'secondary' else ''}" for s in SETTINGS])
        ax.set_xlim(min(0, np.nanmin(g["icc_ci_lo"].to_numpy()) - 0.05), 1.0)
        ax.set_title(f"{'IG' if e == 'ig' else 'Grad-CAM'}: {labels[ms]}" if ms != "logit" else "model logit",
                     fontsize=8)
        ax.set_xlabel("scan-rescan ICC(1,1)", fontsize=8)
    fig.suptitle(f"Scan-rescan ICC with 95% CI (seed {seed}, primary population); bands: Koo & Li; * secondary",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def fig_dlogit_dlaf(pts, out_png, seed):
    p = pts[pts.seed == seed] if len(pts) else pts
    if p.empty:
        return
    expls = sorted(p.explainer.unique())
    fig, axes = plt.subplots(1, len(expls), figsize=(4.2 * len(expls), 3.6))
    cmap = dict(zip(SETTINGS, plt.cm.tab10(np.arange(len(SETTINGS)))))
    for ax, e in zip(np.ravel(axes), expls):
        g = p[p.explainer == e]
        for s, gs in g.groupby("setting"):
            ax.scatter(gs.d_logit, gs.d_laf, s=12, color=cmap[s], label=s, alpha=0.8)
        ax.axhline(0, color="0.6", lw=0.6)
        ax.axvline(0, color="0.6", lw=0.6)
        ax.set_xlabel("scan2 - scan1: logit")
        ax.set_ylabel("scan2 - scan1: LAF")
        ax.set_title("IG" if e == "ig" else "Grad-CAM")
    np.ravel(axes)[-1].legend(fontsize=7, frameon=False)
    fig.suptitle(f"Model instability vs explainer instability (seed {seed})", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def fig_bland_altman(m, out_png, seed):
    g = m[(m.explainer == "ig") & (m.seed == seed) & (m.setting == PRIMARY_SETTING) & ~m.PatientID.isin(PRIMARY_EXCLUDE)]
    _, y = pairs(g, "laf", "scan1", "scan2", "acq")
    if y is None or len(y) < 5:
        return
    mean, d = y.mean(1), y[:, 1] - y[:, 0]
    lo, hi = d.mean() - 1.96 * d.std(ddof=1), d.mean() + 1.96 * d.std(ddof=1)
    fig, ax = plt.subplots(figsize=(4.2, 3.4))
    ax.scatter(mean, d, s=14, color="k")
    for v, ls in ((d.mean(), "-"), (lo, "--"), (hi, "--")):
        ax.axhline(v, color="C3", ls=ls, lw=0.9)
    ax.set_xlabel("mean LAF (scan1, scan2)")
    ax.set_ylabel("LAF scan2 - scan1")
    ax.set_title(f"Primary endpoint: IG LAF at {PRIMARY_SETTING} (n = {len(y)})\nbias {d.mean():+.3f}, "
                 f"LoA {lo:+.3f} to {hi:+.3f}", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def fig_lesion(audit, sal, m, pid, seed, out_dir):
    """2 x 6 (scan x reconstruction) axial centre slices: CT, SEG contour, attribution overlay."""
    import torch
    import torch.nn.functional as F

    qc = load_by_path("rider_qc", "02_rider_qc.py")
    cond = pd.read_csv(audit / "condition_manifest.csv")
    cond = cond[(cond.PatientID == pid) & cond.usable & cond.seg_file.notna()]
    if cond.empty:
        print(f"figure: no usable cells for {pid}")
        return
    cts = {}
    for r in cond.itertuples():
        name = f"{pid}_{r.acq}_{r.setting}"
        mk = sal / "masks" / f"{name}.npz"
        if not mk.exists():
            continue
        z = np.load(mk)
        try:
            ct = load_ct(r.ct_folder, r.ct_series_uid)
        except Exception as e:  # noqa: BLE001
            print(f"figure: {name} CT failed ({e})")
            continue
        cts[(r.acq, r.setting)] = (extract_patch(ct, z["centroid"]), z["mask"].astype(bool), name)
        del ct
    mid = 64
    for expl in ("ig", "gradcam"):
        fig, axes = plt.subplots(2, 6, figsize=(15, 5.8))
        for i, acq in enumerate(("scan1", "scan2")):
            for j, s in enumerate(SETTINGS):
                ax = axes[i, j]
                ax.set_xticks([]), ax.set_yticks([])
                item = cts.get((acq, s))
                mp = sal / "maps" / expl / f"seed{seed}" / f"{pid}_{acq}_{s}.npz"
                if item is None or not mp.exists():
                    ax.set_facecolor("0.85")
                    ax.set_title(f"{acq} {s}\nmissing", fontsize=8, color="red")
                    continue
                patch, lesion, name = item
                a = np.load(mp)["attr"].astype(np.float32)
                if a.shape != patch.shape:  # Grad-CAM stored at 8^3
                    a = F.interpolate(torch.from_numpy(a)[None, None], size=patch.shape, mode="trilinear",
                                      align_corners=False)[0, 0].numpy()
                a = np.clip(a[mid], 0, None)
                vmax = np.percentile(a[a > 0], 99) if (a > 0).any() else 1.0
                a = np.where(a >= 0.05 * vmax, a, 0)  # hide near-zero attribution (display only)
                ax.imshow(patch[mid], cmap="gray", vmin=-1350, vmax=150)
                ax.imshow(np.ma.masked_less_equal(a, 0), cmap="inferno", alpha=0.6, vmin=0, vmax=vmax)
                if lesion[mid].any():
                    ax.contour(lesion[mid], levels=[0.5], colors="c", linewidths=0.7)
                row = m[(m.explainer == expl) & (m.seed == seed) & (m.PatientID == pid) & (m.acq == acq) &
                        (m.setting == s)]
                t = f"LAF {row.laf.iloc[0]:.2f}, logit {row.logit.iloc[0]:+.1f}" if len(row) else ""
                ax.set_title(f"{acq} {s}\n{t}", fontsize=8)
        fig.suptitle(f"{pid}: {'Integrated Gradients' if expl == 'ig' else 'Grad-CAM'} (seed {seed}), axial centre "
                     f"slice of the 128 mm patch; cyan = SEG; positive attribution >= 5% of the 99th pct shown",
                     fontsize=10)
        fig.tight_layout()
        fig.savefig(out_dir / f"fig_lesion_{expl}.png", dpi=130)
        plt.close(fig)


def pick_figure_patient(m, audit):
    comp = audit / "complete_patients.txt"
    pids = [p for p in (comp.read_text().split() if comp.exists() else []) if p not in PRIMARY_EXCLUDE]
    g = m[(m.explainer == "ig") & (m.setting == PRIMARY_SETTING) & m.PatientID.isin(pids or m.PatientID.unique())]
    if g.empty:
        return None
    size = g.groupby("PatientID")["eq_radius_mm"].median()
    return size.sub(size.median()).abs().idxmin()  # the median-sized complete lesion


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--saliency", default=RESULTS_DIR / "saliency", type=Path)
    ap.add_argument("--audit", default=RESULTS_DIR / "audit", type=Path)
    ap.add_argument("--out", default=RESULTS_DIR / "analysis", type=Path)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--figure-patient")
    ap.add_argument("--figure-seed", type=int)
    ap.add_argument("--no-lesion-figure", action="store_true", help="skip the 2 x 6 figure (needs the RIDER DICOM)")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    m = pd.read_csv(args.saliency / "measurands.csv")
    if "laf_chance" not in m or m["laf_chance"].isna().any():  # rows from an older 06: fill from the masks
        ch = {}
        for p in (args.saliency / "masks").glob("*.npz"):
            z = np.load(p)
            r = _roi(z["mask"].shape[0])
            ch[p.stem] = float(z["mask"].astype(bool)[r].sum() / r.sum())
        key = m.PatientID + "_" + m.acq + "_" + m.setting
        m["laf_chance"] = m.get("laf_chance", pd.Series(np.nan, index=m.index)).fillna(key.map(ch))
        m["laf_enrichment"] = m["laf"] / m["laf_chance"]
        m["empty_map"] = m["pos_total"] <= 0
        ig = m.explainer == "ig"
        m.loc[ig, "ig_gap"] = ((m["ig_completeness"] - 1) * (m["logit"] - m["logit_baseline"]))[ig]
    seed_fig = args.figure_seed if args.figure_seed is not None else int(m.seed.min())

    rep = repeatability(m, args.n_boot)
    if rep.empty:
        print("WARNING: fewer than 5 scan1/scan2 pairs per group - no repeatability rows (check measurands.csv)")
        rep = pd.DataFrame(columns=["population", "explainer", "seed", "setting", "measure"])
    rep.round(4).to_csv(args.out / "repeatability.csv", index=False)
    con = conditions(m, args.n_boot)
    con.round(4).to_csv(args.out / "conditions.csv", index=False)
    sd = seeds(m, args.n_boot)
    sd.round(4).to_csv(args.out / "seeds.csv", index=False)
    print("Map-level agreement between reconstructions (loads every map once; a few minutes) ...", flush=True)
    mc = map_conditions(args.saliency, m) if (args.saliency / "maps").exists() else pd.DataFrame()
    mc.round(4).to_csv(args.out / "map_conditions.csv", index=False)
    stab, pts = stability(m, args.n_boot)
    stab.round(4).to_csv(args.out / "stability.csv", index=False)
    pts.round(4).to_csv(args.out / "stability_pairs.csv", index=False)

    # ---- summary ----------------------------------------------------------------------------------
    L = []
    prim = rep[(rep.population == "primary") & (rep.explainer == "ig") & (rep.setting == PRIMARY_SETTING) &
               (rep.measure == "laf")]
    L.append(f"PRIMARY ENDPOINT: IG LAF at {PRIMARY_SETTING}, primary population (excludes {sorted(PRIMARY_EXCLUDE)})")
    for r in prim.itertuples():
        L.append(f"  seed {r.seed}: n={r.n}  ICC {r.icc:.3f} [{r.icc_ci_lo:.3f}, {r.icc_ci_hi:.3f}]  "
                 f"wSD {r.wSD:.4f}  RC {r.RC:.4f} ({100 * r.RC_over_median:.0f}% of median {r.median:.3f})  "
                 f"BA {r.BA_mean:+.4f} [{r.BA_loa_lo:+.4f}, {r.BA_loa_hi:+.4f}]  Koo: {r.koo}"
                 f"{' (conclusive)' if r.koo_conclusive else ''}  AIAG %GRR {r.pct_grr:.0f}% {r.aiag}"
                 f"{' (conclusive)' if r.aiag_conclusive else ''}")
        if hasattr(r, "icc_loo_min"):
            L.append(f"          leave-one-patient-out ICC {r.icc_loo_min:.3f}-{r.icc_loo_max:.3f}; most influential "
                     f"{r.loo_most_influential} (ICC without it {r.icc_without_most_influential:.3f})")
    L.append("")
    L.append("Explainer sanity: ")
    for e, g in m.groupby("explainer"):
        L.append(f"  {e}: median LAF {g.laf.median():.3f} vs chance {g.laf_chance.median():.3f} "
                 f"(enrichment {g.laf_enrichment.median():.1f}x); empty maps {100 * g.empty_map.mean():.1f}% "
                 f"({int(g.empty_map.sum())}/{len(g)})")
    if "ig_gap" in m:
        gg = m[m.explainer == "ig"]["ig_gap"].abs()
        L.append(f"  IG completeness |gap|: median {gg.median():.3f}, 95th pct {gg.quantile(0.95):.3f} logit units")
    L.append("")
    cols = ["explainer", "seed", "setting", "measure", "n", "icc", "icc_ci_lo", "icc_ci_hi", "icc_loo_min",
            "loo_most_influential", "icc_without_most_influential", "wSD", "RC_over_median", "koo", "koo_conclusive",
            "pct_grr", "aiag"]
    with pd.option_context("display.width", 220, "display.max_columns", 30, "display.max_rows", 500):
        L.append("SCAN-RESCAN (primary population)")
        L.append(rep[rep.population == "primary"].reindex(columns=cols).round(3).to_string(index=False))
        L.append("")
        if len(con):
            L.append("BETWEEN RECONSTRUCTIONS (scan1; kernel and thickness contrasts)")
            L.append(con.reindex(columns=["explainer", "seed", "contrast", "kind", "measure", "n", "icc", "icc_ci_lo",
                                          "icc_ci_hi", "icc_loo_min", "loo_most_influential",
                                          "icc_without_most_influential", "BA_mean", "BA_loa_lo", "BA_loa_hi",
                                          "rc_scan_rescan", "exceed_rc", "koo"])
                     .round(3).to_string(index=False))
            L.append("")
        if len(mc):
            L.append("MAP-LEVEL AGREEMENT BETWEEN RECONSTRUCTIONS (same acquisition, centroid-aligned): median [IQR]")
            L.append("  full = 1 mm voxels; _8 = both maps pooled to the 8^3 Grad-CAM grid (16 mm cells), i.e. "
                     "resolution-matched")
            q1 = lambda x: x.quantile(0.25)  # noqa: E731
            q3 = lambda x: x.quantile(0.75)  # noqa: E731
            agg = dict(n=("spearman", "count"))
            for col in ("spearman", "dice_top10", "spearman_8", "dice_top10_8"):
                if col in mc:
                    agg.update({f"{col}_med": (col, "median"), f"{col}_q1": (col, q1), f"{col}_q3": (col, q3)})
            q = mc.groupby(["explainer", "seed", "kind", "contrast"]).agg(**agg)
            L.append(q.round(3).to_string())
            L.append("")
        if len(sd):
            L.append("BETWEEN SEEDS (same cell)")
            L.append(sd[["explainer", "setting", "measure", "n", "n_seeds", "icc", "icc_ci_lo", "icc_ci_hi", "koo"]]
                     .round(3).to_string(index=False))
            L.append("")
        if len(stab):
            L.append("MODEL vs EXPLAINER INSTABILITY (pooled over settings)")
            L.append(stab.round(3).to_string(index=False))
    (args.out / "summary.txt").write_text("\n".join(L) + "\n")
    print("\n".join(L))

    fig_forest(rep, args.out / "fig_icc_forest.png", seed_fig)
    fig_dlogit_dlaf(pts, args.out / "fig_dlogit_dlaf.png", seed_fig)
    fig_bland_altman(m, args.out / "fig_bland_altman_primary.png", seed_fig)
    if not args.no_lesion_figure:
        pid = args.figure_patient or pick_figure_patient(m, args.audit)
        if pid:
            print(f"\nLesion figure: {pid} (seed {seed_fig})")
            fig_lesion(args.audit, args.saliency, m, pid, seed_fig, args.out)
    print(f"\nWrote {args.out}: repeatability.csv, conditions.csv, seeds.csv, stability.csv, summary.txt, figures")


_ROI_CACHE = {}


def _roi(n=None):
    """LAF ROI for an n^3 patch (same rule as 06: 60 mm at 128^3, scaled for other sizes)."""
    from gauge_lib import PATCH_SIZE, patch_roi

    n = PATCH_SIZE if n is None else n
    if n not in _ROI_CACHE:
        _ROI_CACHE[n] = patch_roi(n)
    return _ROI_CACHE[n]


if __name__ == "__main__":
    main()
