#!/usr/bin/env python3
"""
GAUGE-XAI step 10: robustness analyses requested in review (reads results/saliency; writes
results/analysis/robustness/).

  predictions          RIDER logit distribution per seed and reconstruction. Every RIDER patient has a
                       confirmed NSCLC, so the share with logit > 0 is the model's sensitivity on RIDER
                       (domain-shift check; RIDER is used for measurement reliability, not diagnostic accuracy).
  laf_adj              chance-adjusted LAF = (LAF - f) / (1 - f), f = lesion volume fraction of the ROI;
                       scan-rescan and kernel/thickness agreement as in 07
  controls             per map: LAF with the SEG shifted off-centre (6 directions x --shift-mm, mean),
                       LAF with a size-matched SEG from another patient (centred, different shape), and
                       LAF of |attribution| (sign-agnostic). If laf ~ laf_mismatch, attribution is a centred,
                       size-scaled blob rather than lesion-shape specific (the centre prior's expected effect).
  seeds                between-seed agreement with a PATIENT-cluster bootstrap (07 resampled rows, and each
                       patient contributes two rows - one per scan)
  strata               scan-rescan agreement of LAF within strata: lesion size (patient's median equivalent
                       diameter <= 30 mm vs > 30 mm, the LIDC range) and predicted class (patient's mean
                       logit for that run > 0 vs <= 0), per explainer, run and reconstruction; strata_differences.csv
                       gives the pooled-wSD difference between levels with a patient-cluster CI; --only-strata
  variance_components  linear mixed model per explainer and measure (primary population and settings):
                         y ~ kernel + thickness + seed (fixed; systematic offsets)
                         random: patient, patient x scan, patient x seed, patient x kernel, patient x thickness,
                                 residual (all higher-order interactions)
                       variance of each component, % of the random variance, patient-cluster bootstrap CI,
                       and the leave-one-seed-out range of the patient x seed component.

Usage:
  python 10_robustness.py                         # all of the above (bootstrap 200)
  python 10_robustness.py --n-boot 500 --skip-controls
"""
import argparse
import importlib.util
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from gauge_lib import PATCH_SIZE, PATCH_SPACING_MM, PRIMARY_EXCLUDE, RESULTS_DIR, SECONDARY_SETTINGS, SETTINGS

PRIMARY_SETTINGS = [s for s in SETTINGS if s not in SECONDARY_SETTINGS]
SHIFT_DIRS = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]])


def load_by_path(name, fname):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(fname))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


S7 = load_by_path("gauge_analysis", "07_gauge_analysis.py")  # agreement(), pairs(), load_map(), _roi()


# ---- predictions -------------------------------------------------------------------------------------
def predictions(m):
    d = m[(m.explainer == "ig") & ~m.PatientID.isin(PRIMARY_EXCLUDE)].copy()
    d["size"] = np.where(2 * d.eq_radius_mm > 30, ">30 mm", "<=30 mm")
    rows = []
    for (seed, setting), g in d.groupby(["seed", "setting"]):
        for size, gs in [("all", g)] + list(g.groupby("size")):
            rows.append(dict(seed=seed, setting=setting, size=size, n_cells=len(gs), n_patients=gs.PatientID.nunique(),
                             logit_median=gs.logit.median(), logit_q1=gs.logit.quantile(0.25),
                             logit_q3=gs.logit.quantile(0.75), sensitivity_logit_gt0=(gs.logit > 0).mean()))
    return pd.DataFrame(rows)


# ---- chance-adjusted LAF -------------------------------------------------------------------------------
def laf_adj_tables(m, n_boot):
    m = m.copy()
    m["laf_adj"] = (m.laf - m.laf_chance) / (1 - m.laf_chance)
    rep, con = [], []
    prim = m[~m.PatientID.isin(PRIMARY_EXCLUDE)]
    for (e, s, st), g in prim.groupby(["explainer", "seed", "setting"]):
        ids, y = S7.pairs(g, "laf_adj", "scan1", "scan2", "acq")
        if y is not None and len(y) >= 5:
            rep.append(dict(explainer=e, seed=s, setting=st, **S7.agreement(y, n_boot, ids=ids)))
    for (e, s), g in prim[prim.acq == "scan1"].groupby(["explainer", "seed"]):
        for a, b in S7.CONTRASTS:
            ids, y = S7.pairs(g, "laf_adj", a, b, "setting")
            if y is not None and len(y) >= 5:
                con.append(dict(explainer=e, seed=s, contrast=f"{a} vs {b}",
                                kind="kernel" if a[:-1] == b[:-1] else "thickness", **S7.agreement(y, n_boot, ids=ids)))
    return m, pd.DataFrame(rep), pd.DataFrame(con)


# ---- spatial controls on the stored maps ------------------------------------------------------------
def shifted(mask, vox):
    """mask moved by integer voxels vox = (dz, dy, dx), zero-filled."""
    out = np.zeros_like(mask)
    src = [slice(max(0, -v), mask.shape[i] - max(0, v)) for i, v in enumerate(vox)]
    dst = [slice(max(0, v), mask.shape[i] - max(0, -v)) for i, v in enumerate(vox)]
    out[tuple(dst)] = mask[tuple(src)]
    return out


def controls(sal, m, shift_mm):
    masks = {}
    for p in (sal / "masks").glob("*.npz"):
        masks[p.stem] = np.load(p)["mask"].astype(bool)
    n = next(iter(masks.values())).shape[0] if masks else PATCH_SIZE
    shape, roi = (n,) * 3, S7._roi(n)
    if shift_mm is None:
        shift_mm = 25.0 * n / PATCH_SIZE  # 25 mm at 128^3, 12.5 mm at 64^3
    size = {k: int(v.sum()) for k, v in masks.items()}
    pid_of = {k: k.split("_")[0] for k in masks}
    keys = list(masks)
    k_shift = int(round(shift_mm / PATCH_SPACING_MM))
    rows, t0 = [], time.time()
    cells = m[["explainer", "seed", "PatientID", "acq", "setting"]].drop_duplicates()
    for n, r in enumerate(cells.itertuples(index=False), 1):
        name = f"{r.PatientID}_{r.acq}_{r.setting}"
        if name not in masks:
            continue
        a = S7.load_map(sal, r.explainer, r.seed, name, shape)
        if a is None:
            continue
        pos = np.clip(a, 0, None) * roi
        tot = pos.sum()
        les = masks[name]
        absa = np.abs(a) * roi
        row = dict(r._asdict(), laf_abs=float(absa[les].sum() / absa.sum()) if absa.sum() > 0 else np.nan)
        if tot > 0:
            row["laf_check"] = float(pos[les].sum() / tot)
            sh = [shifted(les, k_shift * d) for d in SHIFT_DIRS]
            row["laf_shift"] = float(np.mean([pos[q].sum() / tot for q in sh]))
            # a shifted (or foreign) SEG still overlaps the own lesion; split its LAF into the part inside the own
            # lesion (expected even for perfectly lesion-confined attribution) and the part outside it
            row["laf_shift_outside"] = float(np.mean([pos[q & ~les].sum() / tot for q in sh]))
            row["shift_overlap"] = float(np.mean([(q & les).sum() / max(les.sum(), 1) for q in sh]))
            donors = [k for k in keys if pid_of[k] != r.PatientID]
            if donors:
                donor = min(donors, key=lambda k: abs(size[k] - size[name]))
                dm = masks[donor]
                row["laf_mismatch"] = float(pos[dm].sum() / tot)
                row["laf_mismatch_inside_own"] = float(pos[dm & les].sum() / tot)
                row["laf_mismatch_outside"] = float(pos[dm & ~les].sum() / tot)
                row["mismatch_overlap"] = float((dm & les).sum() / max(les.sum(), 1))
                row["mismatch_donor"] = donor
        rows.append(row)
        if n % 300 == 0:
            print(f"  controls: {n}/{len(cells)} maps ({(time.time() - t0) / 60:.1f} min)", flush=True)
    return pd.DataFrame(rows)


# ---- seeds with a patient-cluster bootstrap ------------------------------------------------------
def cluster_boot_icc(y, ids, n_boot, seed=0):
    ids = np.asarray(ids)
    pats = np.unique(ids)
    idx = {p: np.where(ids == p)[0] for p in pats}
    rng = np.random.default_rng(seed)
    iccs, wsds = [], []
    for _ in range(n_boot):
        take = np.concatenate([idx[p] for p in rng.choice(pats, len(pats), replace=True)])
        icc, wsd = S7.PC.icc_1_1(y[take])
        iccs.append(icc)
        wsds.append(wsd)
    return np.percentile(iccs, [2.5, 97.5]), np.percentile(wsds, [2.5, 97.5])


def seeds_cluster(m, n_boot):
    if m.seed.nunique() < 2:
        return pd.DataFrame()
    rows = []
    d = m[~m.PatientID.isin(PRIMARY_EXCLUDE)]
    for (e, st), g in d.groupby(["explainer", "setting"]):
        for meas in ("laf", "laf_adj", "logit"):
            if meas not in g:
                continue
            w = g.pivot_table(index=["PatientID", "acq"], columns="seed", values=meas).dropna()
            if len(w) < 5:
                continue
            y, ids = w.to_numpy(float), w.index.get_level_values(0).to_numpy()
            icc, wsd = S7.PC.icc_1_1(y)
            (ilo, ihi), (wlo, whi) = cluster_boot_icc(y, ids, n_boot)
            rows.append(dict(explainer=e, setting=st, measure=meas, n_rows=len(y), n_patients=len(np.unique(ids)),
                             n_seeds=y.shape[1], icc=icc, icc_ci_lo=ilo, icc_ci_hi=ihi, wSD=wsd, wSD_ci_lo=wlo,
                             wSD_ci_hi=whi))
    return pd.DataFrame(rows)


# ---- strata: size and predicted class ------------------------------------------------------------
def strata(m, n_boot):
    """Scan-rescan agreement within lesion-size and predicted-class strata. wSD is the main quantity: a
    stratum restricts the between-lesion range, which lowers ICC without any change in measurement error."""
    d = m[~m.PatientID.isin(PRIMARY_EXCLUDE)].copy()
    size = d.groupby("PatientID").eq_radius_mm.median().mul(2)
    d["size"] = d.PatientID.map(lambda p: "<=30 mm" if size[p] <= 30 else ">30 mm")
    mean_logit = d[d.explainer == d.explainer.iloc[0]].groupby(["PatientID", "seed"]).logit.mean()
    d["pred"] = [("logit>0" if mean_logit[(p, s)] > 0 else "logit<=0") for p, s in zip(d.PatientID, d.seed)]
    rows = []
    np.seterr(invalid="ignore", divide="ignore")  # constant / empty strata (Grad-CAM) give an undefined ICC (NaN)
    for (e, seed, setting), g in d.groupby(["explainer", "seed", "setting"]):
        for kind in ("size", "pred"):
            for level, gs in g.groupby(kind):
                ids, y = S7.pairs(gs, "laf", "scan1", "scan2", "acq")
                if y is None or len(y) < 5:
                    continue
                a = S7.agreement(y, n_boot)
                rows.append(dict(explainer=e, seed=seed, setting=setting, stratum=kind, level=level, n=a["n"],
                                 icc=a["icc"], icc_ci_lo=a["icc_ci_lo"], icc_ci_hi=a["icc_ci_hi"], wSD=a["wSD"],
                                 RC=a["RC"], median=a["median"], RC_over_median=a["RC_over_median"]))
    return pd.DataFrame(rows)


def strata_summary(st):
    if st.empty:
        return pd.DataFrame()
    prim = st[st.setting.isin(PRIMARY_SETTINGS)]
    q = prim.groupby(["explainer", "stratum", "level"]).agg(
        n_patients=("n", "median"), icc_median=("icc", "median"), icc_min=("icc", "min"), icc_max=("icc", "max"),
        wSD_median=("wSD", "median"), wSD_min=("wSD", "min"), wSD_max=("wSD", "max"),
        laf_median=("median", "median"), RC_over_median=("RC_over_median", "median"))
    return q


def strata_diff(m, n_boot, seed=0):
    """Pooled scan-rescan wSD per stratum level (primary reconstructions, all runs) and the difference between
    the two levels with a patient-cluster bootstrap CI (patients resampled; a patient keeps all its rows).
    wSD_rel = wSD / mean LAF of the level."""
    d = m[~m.PatientID.isin(PRIMARY_EXCLUDE) & m.setting.isin(PRIMARY_SETTINGS)].copy()
    size = d.groupby("PatientID").eq_radius_mm.median().mul(2)
    d["size"] = d.PatientID.map(lambda p: "<=30 mm" if size[p] <= 30 else ">30 mm")
    mean_logit = d[d.explainer == d.explainer.iloc[0]].groupby(["PatientID", "seed"]).logit.mean()
    d["pred"] = [("logit>0" if mean_logit[(p, s)] > 0 else "logit<=0") for p, s in zip(d.PatientID, d.seed)]
    rng = np.random.default_rng(seed)
    rows = []
    for e, g in d.groupby("explainer"):
        w = g.pivot_table(index=["PatientID", "seed", "setting", "size", "pred"], columns="acq", values="laf").dropna()
        if not {"scan1", "scan2"} <= set(w.columns):
            continue
        w = w.reset_index()
        w["d2"], w["mean"] = (w.scan2 - w.scan1) ** 2, (w.scan1 + w.scan2) / 2
        pats = np.array(sorted(w.PatientID.unique()))
        for kind, (la, lb) in (("size", ("<=30 mm", ">30 mm")), ("pred", ("logit>0", "logit<=0"))):
            agg = w.groupby(["PatientID", kind]).agg(s=("d2", "sum"), n=("d2", "size"), mu=("mean", "sum"))
            agg = agg.unstack(kind).reindex(pats).fillna(0)
            if ("s", la) not in agg or ("s", lb) not in agg:
                continue
            S_, N_, M_ = (agg["s"].to_numpy(float), agg["n"].to_numpy(float), agg["mu"].to_numpy(float))
            cols = list(agg["s"].columns)
            ia, ib = cols.index(la), cols.index(lb)

            def vals(wt):
                s_, n_, mu_ = wt @ S_, wt @ N_, wt @ M_
                with np.errstate(invalid="ignore", divide="ignore"):
                    wsd = np.sqrt(s_ / n_ / 2)
                    return wsd, wsd / (mu_ / n_)

            est = vals(np.ones(len(pats)))
            wts = np.stack([np.bincount(rng.integers(0, len(pats), len(pats)), minlength=len(pats)) for _ in range(n_boot)])
            bw, br = [], []
            for wt in wts:
                a_, r_ = vals(wt.astype(float))
                bw.append(a_[ib] - a_[ia])
                br.append(r_[ib] - r_[ia])
            for metric, k, b in (("wSD", 0, np.array(bw)), ("wSD_rel", 1, np.array(br))):
                rows.append(dict(explainer=e, stratum=kind, metric=metric, level_a=la, level_b=lb,
                                 n_patients_a=int((N_[:, ia] > 0).sum()), n_patients_b=int((N_[:, ib] > 0).sum()),
                                 value_a=est[k][ia], value_b=est[k][ib], diff_b_minus_a=est[k][ib] - est[k][ia],
                                 diff_ci_lo=np.nanpercentile(b, 2.5), diff_ci_hi=np.nanpercentile(b, 97.5)))
    return pd.DataFrame(rows)


# ---- variance components -------------------------------------------------------------------------
VC = {"patient x scan": "0 + C(acq)", "patient x seed": "0 + C(seed)", "patient x kernel": "0 + C(kernel)",
      "patient x thickness": "0 + C(thick)"}


def fit_vc(d, meas):
    import statsmodels.formula.api as smf

    d = d.rename(columns={meas: "y"}).reset_index(drop=True)  # unique row labels (bootstrap duplicates patients)
    fixed = "y ~ C(kernel) + C(thick)" + (" + C(seed)" if d.seed.nunique() > 1 else "")
    vcf = {k.replace(" ", "_"): v for k, v in VC.items() if not (k == "patient x seed" and d.seed.nunique() < 2)}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = smf.mixedlm(fixed, d, groups="PatientID", re_formula="1", vc_formula=vcf)
        res = model.fit(reml=True, method=["lbfgs", "powell"])
    comps = {"patient": float(res.cov_re.iloc[0, 0])}
    for name, v in zip(model.exog_vc.names, res.vcomp):
        comps[name.replace("_", " ")] = float(v)
    comps["residual"] = float(res.scale)
    return comps, bool(res.converged)


def variance_components(m, n_boot):
    d0 = m[~m.PatientID.isin(PRIMARY_EXCLUDE) & m.setting.isin(PRIMARY_SETTINGS)].copy()
    d0["kernel"] = d0.setting.str[-1]
    d0["thick"] = d0.setting.str[:-1]
    targets = [(e, ms) for e in sorted(d0.explainer.unique()) for ms in ("laf", "laf_adj")] + [("ig", "logit")]
    rows = []
    for e, ms in targets:
        d = d0[d0.explainer == e].dropna(subset=[ms])
        if len(d) < 50:
            continue
        t0 = time.time()
        est, conv = fit_vc(d, ms)
        # patient-cluster bootstrap (patients resampled with replacement, relabelled so duplicates are distinct)
        pats = d.PatientID.unique()
        rng = np.random.default_rng(0)
        boots = []
        for b in range(n_boot):
            pick = rng.choice(pats, len(pats), replace=True)
            db = pd.concat([d[d.PatientID == p].assign(PatientID=f"{p}#{i}") for i, p in enumerate(pick)],
                           ignore_index=True)
            try:
                boots.append(fit_vc(db, ms)[0])
            except Exception:  # noqa: BLE001
                pass
        boots = pd.DataFrame(boots)
        # leave-one-seed-out: how much the patient x seed component depends on which seeds were trained
        loso = []
        if d.seed.nunique() >= 3:
            for s in sorted(d.seed.unique()):
                try:
                    loso.append(fit_vc(d[d.seed != s], ms)[0].get("patient x seed", np.nan))
                except Exception:  # noqa: BLE001
                    pass
        total = sum(est.values())
        for comp, v in est.items():
            row = dict(explainer=e, measure=ms, component=comp, variance=v, sd=np.sqrt(max(v, 0)),
                       pct_of_random=100 * v / total if total > 0 else np.nan, converged=conv,
                       n_obs=len(d), n_patients=d.PatientID.nunique(), n_seeds=d.seed.nunique(),
                       n_boot_ok=len(boots))
            if comp in boots:
                lo, hi = np.percentile(boots[comp].clip(lower=0), [2.5, 97.5])
                row.update(sd_ci_lo=np.sqrt(lo), sd_ci_hi=np.sqrt(hi))
                pct = 100 * boots[comp] / boots.sum(axis=1)
                row.update(pct_ci_lo=np.percentile(pct, 2.5), pct_ci_hi=np.percentile(pct, 97.5))
            if comp == "patient x seed" and loso:
                row.update(sd_leave_one_seed_out_min=np.sqrt(max(min(loso), 0)),
                           sd_leave_one_seed_out_max=np.sqrt(max(max(loso), 0)))
            rows.append(row)
        print(f"  variance components {e}/{ms}: {len(d)} obs, converged {conv}, {len(boots)} bootstrap fits "
              f"({(time.time() - t0) / 60:.1f} min)", flush=True)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--saliency", default=RESULTS_DIR / "saliency", type=Path)
    ap.add_argument("--out", default=RESULTS_DIR / "analysis" / "robustness", type=Path)
    ap.add_argument("--n-boot", type=int, default=200)
    ap.add_argument("--shift-mm", type=float, default=None, help="SEG shift for the control (default 25 mm x size/128)")
    ap.add_argument("--skip-controls", action="store_true", help="skip the map-based controls (needs maps/, slow)")
    ap.add_argument("--only-vc", action="store_true", help="only the variance components (the rest already written)")
    ap.add_argument("--only-strata", action="store_true", help="only the size / predicted-class strata")
    ap.add_argument("--only-controls", action="store_true", help="only the map-based spatial controls (controls.csv)")
    args = ap.parse_args()
    try:
        import statsmodels  # noqa: F401
    except ImportError:
        raise SystemExit("statsmodels is needed for the variance components: pip install statsmodels")
    args.out.mkdir(parents=True, exist_ok=True)
    m = pd.read_csv(args.saliency / "measurands.csv")
    if "laf_chance" not in m:
        raise SystemExit("measurands.csv has no laf_chance - rerun 06_rider_saliency.py (it backfills the column)")
    L = [f"seeds: {sorted(m.seed.unique())}; explainers: {sorted(m.explainer.unique())}; cells: "
         f"{m[['PatientID', 'acq', 'setting']].drop_duplicates().shape[0]}", ""]

    if args.only_controls:
        ctl = controls(args.saliency, m, args.shift_mm)
        ctl.round(5).to_csv(args.out / "controls.csv", index=False)
        cols = ["laf_check", "laf_shift", "laf_shift_outside", "shift_overlap", "laf_mismatch", "laf_mismatch_inside_own",
                "laf_mismatch_outside", "mismatch_overlap", "laf_abs"]
        with pd.option_context("display.width", 220, "display.max_columns", 20):
            print(ctl.groupby("explainer")[[c for c in cols if c in ctl]].median().round(4).to_string())
        print(f"\nWrote {args.out / 'controls.csv'}")
        return
    if args.only_strata:
        st = strata(m, args.n_boot)
        st.round(4).to_csv(args.out / "strata.csv", index=False)
        q = strata_summary(st)
        q.round(4).to_csv(args.out / "strata_summary.csv")
        sdf = strata_diff(m, args.n_boot)
        sdf.round(5).to_csv(args.out / "strata_differences.csv", index=False)
        with pd.option_context("display.width", 200, "display.max_columns", 20):
            print("SCAN-RESCAN LAF WITHIN STRATA (primary population and reconstructions; over runs/settings)")
            print(q.round(3).to_string())
            print("\nSTRATUM DIFFERENCES (pooled wSD, level b - level a, patient-cluster bootstrap 95% CI)")
            print(sdf.round(4).to_string(index=False))
        print(f"\nWrote {args.out / 'strata.csv'}, strata_summary.csv and strata_differences.csv")
        return
    if args.only_vc:
        m["laf_adj"] = (m.laf - m.laf_chance) / (1 - m.laf_chance)
        vc = variance_components(m, args.n_boot)
        vc.round(5).to_csv(args.out / "variance_components.csv", index=False)
        print(vc.round(4).to_string(index=False))
        print(f"\nWrote {args.out / 'variance_components.csv'}")
        return
    pr = predictions(m)
    pr.round(4).to_csv(args.out / "rider_predictions.csv", index=False)
    a = pr[(pr["size"] == "all")].groupby("seed").agg(sens_min=("sensitivity_logit_gt0", "min"),
                                                      sens_max=("sensitivity_logit_gt0", "max"),
                                                      logit_median=("logit_median", "median"))
    L.append("RIDER PREDICTIONS (all RIDER tumours are NSCLC; sensitivity = share with logit > 0, per setting)")
    L.append(a.round(3).to_string())
    by = pr[pr["size"] != "all"].groupby("size")[["sensitivity_logit_gt0", "logit_median"]].median()
    L.append(by.round(3).to_string())
    L.append("")

    m, rep, con = laf_adj_tables(m, args.n_boot)
    rep.round(4).to_csv(args.out / "laf_adj_repeatability.csv", index=False)
    con.round(4).to_csv(args.out / "laf_adj_conditions.csv", index=False)
    L.append("CHANCE-ADJUSTED LAF (LAF - f)/(1 - f): median value, scan-rescan ICC (median over seeds, primary settings)")
    for e, g in m.groupby("explainer"):
        if rep.empty or con.empty:
            L.append(f"  {e}: median laf_adj {g.laf_adj.median():.3f} (too few pairs for agreement statistics)")
            continue
        rr = rep[(rep.explainer == e) & rep.setting.isin(PRIMARY_SETTINGS)]
        kk = con[(con.explainer == e) & (con.kind == "kernel")]
        L.append(f"  {e}: median laf_adj {g.laf_adj.median():.3f}; scan-rescan ICC {rr.icc.median():.3f} "
                 f"[{rr.icc.min():.3f}-{rr.icc.max():.3f}]; kernel ICC {kk.icc.median():.3f} "
                 f"[{kk.icc.min():.3f}-{kk.icc.max():.3f}]")
    L.append("")

    if not args.skip_controls and (args.saliency / "maps").exists():
        print("Spatial controls on the stored maps ...", flush=True)
        ctl = controls(args.saliency, m, args.shift_mm)
        ctl.round(5).to_csv(args.out / "controls.csv", index=False)
        L.append(f"SPATIAL CONTROLS (median over maps): SEG shifted {args.shift_mm or 'default'} mm; size-matched SEG of another "
                 f"patient; |attribution|")
        q = ctl.merge(m[["explainer", "seed", "PatientID", "acq", "setting", "laf", "laf_chance"]],
                      on=["explainer", "seed", "PatientID", "acq", "setting"])
        for col in ("laf_check", "laf_shift", "laf_shift_outside", "laf_mismatch", "laf_mismatch_inside_own",
                    "laf_mismatch_outside", "mismatch_overlap"):
            if col not in q:
                q[col] = np.nan
        q["own_minus_mismatch"] = q.laf - q.laf_mismatch
        L.append(q.groupby("explainer")[["laf", "laf_shift", "laf_shift_outside", "laf_mismatch",
                                         "laf_mismatch_inside_own", "laf_mismatch_outside", "mismatch_overlap",
                                         "laf_abs", "laf_chance"]].median().round(4).to_string())
        L.append("  (laf_check should equal laf: " + f"max |diff| {np.nanmax(np.abs(q.laf_check - q.laf)):.2e})")
        L.append("")

    sc = seeds_cluster(m, args.n_boot)
    sc.round(4).to_csv(args.out / "seeds_cluster_bootstrap.csv", index=False)
    if len(sc):
        L.append("BETWEEN-SEED AGREEMENT, patient-cluster bootstrap 95% CI")
        L.append(sc[sc.setting.isin(PRIMARY_SETTINGS)].groupby(["explainer", "measure"])[
            ["n_seeds", "icc", "icc_ci_lo", "icc_ci_hi", "wSD", "wSD_ci_lo", "wSD_ci_hi"]].median().round(3).to_string())
        L.append("")

    st = strata(m, args.n_boot)
    st.round(4).to_csv(args.out / "strata.csv", index=False)
    q = strata_summary(st)
    q.round(4).to_csv(args.out / "strata_summary.csv")
    sdf = strata_diff(m, args.n_boot)
    sdf.round(5).to_csv(args.out / "strata_differences.csv", index=False)
    L.append("SCAN-RESCAN LAF WITHIN STRATA (size: patient median equivalent diameter; pred: mean logit for the run)")
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        L.append(q.round(3).to_string())
    L.append("")

    print("Variance components ...", flush=True)
    vc = variance_components(m, args.n_boot)
    vc.round(5).to_csv(args.out / "variance_components.csv", index=False)
    if len(vc):
        L.append("VARIANCE COMPONENTS (REML; fixed: kernel, thickness, seed; % of random variance; bootstrap CI)")
        cols = ["explainer", "measure", "component", "sd", "sd_ci_lo", "sd_ci_hi", "pct_of_random", "pct_ci_lo",
                "pct_ci_hi", "sd_leave_one_seed_out_min", "sd_leave_one_seed_out_max", "converged", "n_boot_ok"]
        L.append(vc.reindex(columns=cols).round(4).to_string(index=False))
    (args.out / "robustness_summary.txt").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
