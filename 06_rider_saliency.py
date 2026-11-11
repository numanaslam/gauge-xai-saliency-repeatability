#!/usr/bin/env python3
"""
GAUGE-XAI step 6: explain the frozen LIDC models on every usable RIDER cell and compute the measurands.

For each RIDER cell (patient x scan x reconstruction, usable in condition_manifest.csv and qc_ok in
lesion_stats.csv) the CT and its own SEG are loaded in that scan's native space (no registration), a
128^3, 1 mm patch is sampled around the SEG centroid exactly as for LIDC, and for every seed's model:
  logit           malignancy logit (explained quantity; logits, not probabilities, avoid saturation)
  IG              Integrated Gradients on the CT channel, baseline = air (-1000 HU), --ig-steps steps
  Grad-CAM        layer4 (8^3, 16 mm cells), trilinearly upsampled to 128^3
Measurands per map (positive attribution only, inside a fixed ROI = sphere of ROI_RADIUS_MM):
  laf                 Lesion Attribution Fraction: positive attribution in the SEG / in the ROI  (PRIMARY)
  laf_sphere10        same with a fixed 10 mm-radius sphere at the centroid instead of the SEG
  laf_sphere_patient  same with a sphere of the patient's median equivalent radius (identical for both
                      scans: removes contour variability, keeps lesion size)
  com_offset_rel      distance of the positive-attribution centre of mass from the lesion centroid,
                      divided by the lesion's equivalent radius
  entropy_norm        normalised entropy of the positive attribution in the ROI (0 = one voxel, 1 = uniform)
  laf_chance          lesion voxels / ROI voxels: the LAF of a uniform map (compare laf against it)
  laf_enrichment      laf / laf_chance
  empty_map           no positive attribution in the ROI (Grad-CAM of a confidently benign case)
  ig_completeness     IG sum / (logit(x) - logit(baseline)); unstable when the two logits are close
  ig_gap              IG sum - (logit(x) - logit(baseline)), in logit units - should be close to 0
With --gradcam-layer3, Grad-CAM is also computed at layer3 (16^3, 8 mm cells) as explainer "gradcam_l3".

Writes (default ../results/saliency):
  measurands.csv                      one row per explainer x seed x cell
  maps/<explainer>/seed<k>/<cell>.npz  IG as float16 128^3; Grad-CAM at native 8^3 (upsample on use)
  masks/<cell>.npz                    lesion mask on the patch grid (uint8) + centroid, eq. radius
  cells/<cell>.csv                    per-cell rows (resume per seed: only missing seeds are computed,
                                      so re-run after training more seeds)
  determinism.txt                     the same cell explained twice: differences must be ~0

Usage:
  python 06_rider_saliency.py --patients RIDER-1129164940        # quick test on one patient
  python 06_rider_saliency.py                                    # all cells, all seeds in ../results/runs
"""
import argparse
import importlib.util
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from gauge_lib import (PATCH_SIZE, PATCH_SPACING_MM, RESULTS_DIR, extract_patch, from_uint8, load_ct, roi_radius_mm,
                       normalize_hu, pixel_to_world, to_uint8)
from gauge_model import load_checkpoint

ROI_RADIUS_MM = roi_radius_mm(PATCH_SIZE)  # fixed ROI: 60 mm sphere in the 128 mm patch (scaled for 64^3 models)
SPHERE_FIXED_MM = 10.0        # fixed-sphere control radius
EXPLAINERS = ("ig", "gradcam")


def load_qc_module():
    """02_rider_qc.py starts with a digit, so import it by path for its SEG decoder."""
    spec = importlib.util.spec_from_file_location("rider_qc", Path(__file__).with_name("02_rider_qc.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def patch_coords_mm(n=PATCH_SIZE, spacing=PATCH_SPACING_MM):
    g = (np.arange(n) - (n - 1) / 2.0) * spacing
    zz, yy, xx = np.meshgrid(g, g, g, indexing="ij", sparse=True)
    return zz, yy, xx, np.sqrt(zz ** 2 + yy ** 2 + xx ** 2)


ZZ, YY, XX, RR = patch_coords_mm()
ROI = RR <= ROI_RADIUS_MM


def set_grid(n):
    """Patch grid and ROI for models with n^3 inputs (128 default; 64 for the centre-prior ablation)."""
    global ZZ, YY, XX, RR, ROI, ROI_RADIUS_MM
    ZZ, YY, XX, RR = patch_coords_mm(n)
    ROI_RADIUS_MM = roi_radius_mm(n)
    ROI = RR <= ROI_RADIUS_MM


def measurands(attr, lesion, eq_radius_mm, patient_radius_mm):
    pos = np.clip(attr, 0, None) * ROI
    tot = float(pos.sum())
    out = dict(pos_total=tot, neg_total=float(np.clip(attr, None, 0)[ROI].sum()))
    if tot <= 0:
        return {**out, **{k: np.nan for k in ("laf", "laf_sphere10", "laf_sphere_patient",
                                               "com_offset_rel", "entropy_norm")}}
    out["laf"] = float(pos[lesion].sum() / tot)
    out["laf_sphere10"] = float(pos[RR <= SPHERE_FIXED_MM].sum() / tot)
    out["laf_sphere_patient"] = float(pos[RR <= patient_radius_mm].sum() / tot)
    com = np.array([(pos * ZZ).sum(), (pos * YY).sum(), (pos * XX).sum()]) / tot
    out["com_offset_rel"] = float(np.linalg.norm(com) / max(eq_radius_mm, 1e-6))
    p = pos[ROI] / tot
    p = p[p > 0]
    out["entropy_norm"] = float(-(p * np.log(p)).sum() / np.log(ROI.sum()))
    return out


def integrated_gradients(model, x, steps, batch, base=None):
    """x: (1, 1, D, H, W) on device. Default baseline 0 = -1000 HU (air). Riemann midpoint rule."""
    base = torch.zeros_like(x) if base is None else base
    alphas = (torch.arange(steps, device=x.device, dtype=torch.float32) + 0.5) / steps
    grad_sum = torch.zeros_like(x)
    for i in range(0, steps, batch):
        a = alphas[i:i + batch].view(-1, 1, 1, 1, 1)
        xi = (base + a * (x - base)).requires_grad_(True)
        out = model(xi)
        g, = torch.autograd.grad(out.sum(), xi)
        grad_sum += g.sum(0, keepdim=True)
    attr = (x - base) * grad_sum / steps
    with torch.no_grad():
        f_x, f_b = model(x).item(), model(base).item()
    return attr[0, 0].detach().cpu().numpy(), f_x, f_b


GRADCAM_LAYERS = {"gradcam": "layer4", "gradcam_l3": "layer3"}  # explainer name -> layer (8^3 / 16^3 cells)
IG_VARIANTS = ("ig", "ig_blur", "ig_mean")        # IG with air (default), blurred-input and mean-intensity baselines
ALL_EXPLAINERS = IG_VARIANTS + tuple(GRADCAM_LAYERS) + ("occlusion",)


def ig_baseline(x, kind, blur_mm=5.0):
    if kind == "ig":
        return torch.zeros_like(x)
    if kind == "ig_mean":
        return torch.full_like(x, float(x.mean()))
    from scipy.ndimage import gaussian_filter  # ig_blur: the input blurred with a 5 mm Gaussian

    b = gaussian_filter(x[0, 0].cpu().numpy(), sigma=blur_mm / PATCH_SPACING_MM)
    return torch.from_numpy(b.astype(np.float32))[None, None].to(x.device)


@torch.no_grad()
def occlusion(model, x, grid=8, batch=16, fill=0.0):
    """Occlusion: each of grid^3 non-overlapping cubes (16 mm at 128^3) is set to fill (air) in turn; the
    attribution of the cube is logit(x) - logit(occluded). Returns (grid^3 map, upsampled map, logit)."""
    n = x.shape[-1]
    c = n // grid
    f_x = model(x).item()
    idx = [(i, j, k) for i in range(grid) for j in range(grid) for k in range(grid)]
    drop = np.zeros((grid,) * 3, np.float32)
    for b0 in range(0, len(idx), batch):
        chunk = idx[b0:b0 + batch]
        xb = x.repeat(len(chunk), 1, 1, 1, 1)
        for t, (i, j, k) in enumerate(chunk):
            xb[t, :, i * c:(i + 1) * c, j * c:(j + 1) * c, k * c:(k + 1) * c] = fill
        out = model(xb).float().cpu().numpy()
        for t, (i, j, k) in enumerate(chunk):
            drop[i, j, k] = f_x - out[t]
    up = F.interpolate(torch.from_numpy(drop)[None, None], size=x.shape[2:], mode="trilinear",
                       align_corners=False)[0, 0].numpy()
    return drop, up, f_x


def grad_cam(model, x, layer="layer4"):
    """Grad-CAM of the malignancy logit at a layer -> (cam at the layer's grid, cam upsampled to the input)."""
    store = {}
    h = getattr(model, layer).register_forward_hook(lambda _, __, o: store.__setitem__("a", o))
    try:
        with torch.enable_grad():
            out = model(x)
            a = store["a"]
            g, = torch.autograd.grad(out.sum(), a)
    finally:
        h.remove()
    w = g.mean(dim=(2, 3, 4), keepdim=True)
    cam = F.relu((w * a).sum(1, keepdim=True))
    up = F.interpolate(cam, size=x.shape[2:], mode="trilinear", align_corners=False)
    return cam[0, 0].detach().cpu().numpy(), up[0, 0].detach().cpu().numpy(), out.item()


def cell_name(r):
    return f"{r['PatientID']}_{r['acq']}_{r['setting']}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--audit", default=RESULTS_DIR / "audit", type=Path)
    ap.add_argument("--runs", default=RESULTS_DIR / "runs", type=Path, help="folder with seed*/best.pt")
    ap.add_argument("--out", default=RESULTS_DIR / "saliency", type=Path)
    ap.add_argument("--patients", nargs="*")
    ap.add_argument("--ig-steps", type=int, default=32)
    ap.add_argument("--ig-batch", type=int, default=4, help="interpolation points per backward pass (GPU memory)")
    ap.add_argument("--no-save-maps", action="store_true", help="only measurands (saves ~4 MB per IG map)")
    ap.add_argument("--explainers", default="ig,gradcam",
                    help=f"comma-separated subset of {','.join(ALL_EXPLAINERS)} (journal: add occlusion,ig_blur,ig_mean)")
    ap.add_argument("--save-patches", action="store_true",
                    help="also store each cell's CT patch (uint8) for 14_faithfulness.py and 15_scan_registration.py")
    ap.add_argument("--centre-jitter-mm", type=float, default=0.0,
                    help="journal centring-sensitivity run: shift the patch centre by this distance in a random direction, "
                         "drawn independently per cell (simulates an imperfect lesion detector); ROI stays at the patch centre")
    ap.add_argument("--gradcam-layer3", action="store_true",
                    help="also compute Grad-CAM at layer3 (16^3 = 8 mm cells), explainer name 'gradcam_l3'")
    args = ap.parse_args()
    explainers = [e.strip() for e in args.explainers.split(",") if e.strip()]
    explainers += ["gradcam_l3"] if args.gradcam_layer3 and "gradcam_l3" not in explainers else []
    bad = [e for e in explainers if e not in ALL_EXPLAINERS]
    if bad:
        raise SystemExit(f"unknown explainer(s) {bad}; choose from {ALL_EXPLAINERS}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # deterministic kernels: repeat runs of the same input must give identical maps (determinism.txt)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)

    ckpts = sorted(args.runs.glob("seed*/best.pt"))
    if not ckpts:
        raise SystemExit(f"no seed*/best.pt under {args.runs}")
    models, sizes = {}, set()
    for c in ckpts:
        m, ck = load_checkpoint(c, device)
        models[int(c.parent.name.replace("seed", ""))] = m
        sizes.add(ck["model_cfg"].get("input_size", PATCH_SIZE))
        print(f"loaded {c} (epoch {ck['epoch']}, val AUC {ck['val_auc']:.3f}, cfg {ck['model_cfg']})")

    if len(sizes) != 1:
        raise SystemExit(f"models in {args.runs} have different input sizes {sizes}")
    n_in = sizes.pop()
    set_grid(n_in)
    print(f"input {n_in}^3 at {PATCH_SPACING_MM:g} mm, ROI radius {ROI_RADIUS_MM:g} mm"
          + ("" if n_in == PATCH_SIZE else "  (ablation grid)"))

    cond = pd.read_csv(args.audit / "condition_manifest.csv")
    cond = cond[cond["usable"] & cond["seg_file"].notna()]
    stats = pd.read_csv(args.audit / "lesion_stats.csv")
    if "qc_ok" in stats:
        bad = stats[~stats["qc_ok"]][["PatientID", "acq", "setting"]]
        cond = cond.merge(bad, how="left", indicator=True).query("_merge == 'left_only'").drop(columns="_merge")
        print(f"skipping {len(bad)} cells rejected by 02_rider_qc.py")
    if args.patients:
        cond = cond[cond["PatientID"].isin(args.patients)]
    # sphere control radius: the patient's median equivalent radius over all its qc_ok cells
    good = stats[stats["qc_ok"]] if "qc_ok" in stats else stats
    patient_radius = (good.groupby("PatientID")["eq_sphere_diam_mm"].median() / 2).to_dict()

    qc = load_qc_module()
    for d in ("cells", "masks"):
        (args.out / d).mkdir(parents=True, exist_ok=True)
    t0, done = time.time(), 0
    cells = list(cond.sort_values(["PatientID", "acq", "setting"]).to_dict("records"))
    for n, r in enumerate(cells, 1):
        name = cell_name(r)
        cell_csv = args.out / "cells" / f"{name}.csv"
        old = pd.read_csv(cell_csv) if cell_csv.exists() else pd.DataFrame(columns=["seed", "explainer"])
        have = set(zip(old["seed"].astype(int), old["explainer"]))
        todo = {s: [e for e in explainers if (s, e) not in have] for s in models}  # resume per seed and explainer
        todo = {s: es for s, es in todo.items() if es}
        need_patch = args.save_patches and not (args.out / "patches" / f"{name}.npz").exists()
        if not todo and not need_patch:
            continue
        try:
            ct = load_ct(r["ct_folder"], r["ct_series_uid"])
            mask, _ = qc.load_seg_mask(r["seg_file"], ct)
        except Exception as e:  # noqa: BLE001
            print(f"{name}: FAILED to load ({type(e).__name__}: {e})")
            continue
        kk, rr_, cc = np.nonzero(mask)
        centroid = pixel_to_world(ct, kk, rr_, cc).mean(0)
        if args.centre_jitter_mm > 0:  # deterministic per cell, independent between scans and reconstructions
            import zlib

            v = np.random.default_rng(zlib.crc32(name.encode())).normal(size=3)
            centroid = centroid + args.centre_jitter_mm * v / np.linalg.norm(v)
        vox_ml = ct["ps"][0] * ct["ps"][1] * ct["dz"] / 1000.0
        eq_radius = 0.5 * (6 * mask.sum() * vox_ml * 1000 / np.pi) ** (1 / 3)
        p_rad = patient_radius.get(r["PatientID"], eq_radius)
        patch = from_uint8(to_uint8(normalize_hu(extract_patch(ct, centroid, size=n_in))))  # as in training
        lesion = extract_patch(ct, centroid, size=n_in, volume=mask.astype(np.float32), cval=0.0) > 0.5
        np.savez_compressed(args.out / "masks" / f"{name}.npz", mask=lesion.astype(np.uint8), centroid=centroid,
                            eq_radius_mm=eq_radius, patient_radius_mm=p_rad)
        if args.save_patches:
            (args.out / "patches").mkdir(exist_ok=True)
            np.savez_compressed(args.out / "patches" / f"{name}.npz", patch=to_uint8(patch))
        del ct, mask
        if not todo:  # only the patch was missing
            continue
        x = torch.from_numpy(patch)[None, None].to(device)

        roi_lesion = int((lesion & ROI).sum())
        chance = roi_lesion / int(ROI.sum())
        rows = []
        for seed, need in todo.items():
            model = models[seed]
            maps, stored, ig_base_logit = {}, {}, {}
            with torch.no_grad():
                f_x, f_b = model(x).item(), model(torch.zeros_like(x)).item()
            for expl in need:
                if expl in IG_VARIANTS:
                    base = ig_baseline(x, expl)
                    a, _, fb = integrated_gradients(model, x, args.ig_steps, args.ig_batch, base=base)
                    maps[expl], stored[expl], ig_base_logit[expl] = a, a.astype(np.float16), fb
                elif expl in GRADCAM_LAYERS:
                    cam_native, cam, _ = grad_cam(model, x, GRADCAM_LAYERS[expl])
                    maps[expl], stored[expl] = cam, cam_native.astype(np.float32)
                elif expl == "occlusion":
                    occ_native, occ, _ = occlusion(model, x)
                    maps[expl], stored[expl] = occ, occ_native
            for expl in need:
                attr = maps[expl]
                row = dict(explainer=expl, seed=seed, PatientID=r["PatientID"], acq=r["acq"], setting=r["setting"],
                           logit=f_x, logit_baseline=f_b, eq_radius_mm=eq_radius, patient_radius_mm=p_rad,
                           lesion_voxels_in_patch=int(lesion.sum()),
                           lesion_inside_roi=float(roi_lesion / max(lesion.sum(), 1)), laf_chance=chance,
                           **measurands(attr, lesion, eq_radius, p_rad))
                row["laf_enrichment"] = row["laf"] / chance if chance > 0 else np.nan
                row["empty_map"] = bool(row["pos_total"] <= 0)
                if expl in IG_VARIANTS:
                    fb = ig_base_logit[expl]
                    row["logit_ig_baseline"] = fb
                    row["ig_completeness"] = float(attr.sum() / (f_x - fb)) if abs(f_x - fb) > 1e-6 else np.nan
                    row["ig_gap"] = float(attr.sum() - (f_x - fb))
                rows.append(row)
            if not args.no_save_maps:
                for expl, arr in stored.items():
                    d = args.out / "maps" / expl / f"seed{seed}"
                    d.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(d / f"{name}.npz", attr=arr)
            if done == 0 and seed == min(todo) and "ig" in need:
                ig2, _, _ = integrated_gradients(model, x, args.ig_steps, args.ig_batch)
                _, cam2, _ = grad_cam(model, x)
                _, cam1, _ = grad_cam(model, x)
                (args.out / "determinism.txt").write_text(
                    f"{name} seed {seed}: same input explained twice\n"
                    f"  IG max |diff| = {np.abs(maps['ig'] - ig2).max():.3e} (max |IG| = {np.abs(maps['ig']).max():.3e})\n"
                    f"  Grad-CAM max |diff| = {np.abs(cam1 - cam2).max():.3e} (max = {np.abs(cam1).max():.3e})\n")
        if len(old):  # rows from an older 06 version: same cell and mask, so the new columns can be filled
            old["laf_chance"] = chance
            old["laf_enrichment"] = old["laf"] / chance if chance > 0 else np.nan
            old["empty_map"] = old["pos_total"] <= 0
            is_ig = old["explainer"] == "ig"
            if "ig_completeness" in old:
                old.loc[is_ig, "ig_gap"] = ((old["ig_completeness"] - 1) * (old["logit"] - old["logit_baseline"]))[is_ig]
        pd.concat([old, pd.DataFrame(rows)], ignore_index=True).to_csv(cell_csv, index=False)
        done += 1
        el = (time.time() - t0) / 60
        laf_txt = ", ".join(f"{q['explainer']} LAF {q['laf']:.3f}" for q in rows if q["seed"] == min(todo))
        print(f"[{n}/{len(cells)}] {name} seeds {sorted(todo)}: logit {rows[0]['logit']:+.2f}, {laf_txt} "
              f"(chance {chance:.3f})  ({el:.1f} min, ~{el / done * (len(cells) - n):.0f} min left)")

    parts = sorted((args.out / "cells").glob("*.csv"))
    if parts:
        res = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
        res.to_csv(args.out / "measurands.csv", index=False)
        print(f"\nWrote {args.out}/measurands.csv: {len(res)} rows, {res['PatientID'].nunique()} patients")
        print(res.groupby(["explainer", "setting"])[["laf", "laf_sphere_patient", "logit"]].median().round(3))
        print(f"Rows per seed: {res.groupby('seed').size().to_dict()}")
        if "ig_gap" in res:
            g = res[res.explainer == "ig"]["ig_gap"].abs()
            print(f"IG |gap| (logit units): median {g.median():.3f}, 95th pct {g.quantile(0.95):.3f} "
                  f"(raise --ig-steps if large)")
    if (args.out / "determinism.txt").exists():
        print((args.out / "determinism.txt").read_text())


if __name__ == "__main__":
    main()
