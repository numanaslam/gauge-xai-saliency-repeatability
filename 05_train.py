#!/usr/bin/env python3
"""
GAUGE-XAI step 5: train the frozen LIDC malignancy model (one run per seed).

Task: radiologist-assessed nodule malignancy (LIDC ratings are not pathology). A nodule is
benign if the median reader rating is < 3, malignant if > 3; median = 3 is excluded. Only nodules
marked by >= --min-readers readers are used.

The train/validation split is by patient and fixed by --split-seed, so seeds differ only in
initialisation, augmentation and batch order (the "seed" factor in the gauge study). RIDER is
never touched here.

Writes (in --out/seed<N>):
  best.pt               weights with the highest validation AUC (+ config, epoch, metrics)
  last.pt               weights after the final epoch
  history.csv           per-epoch train loss, val loss, val AUC
  val_predictions.csv   validation logits of best.pt
  summary.json
and --out/split.csv (patient-level split, shared by all seeds).

Default configuration (19 Oct 2026): 128^3 at 1 mm, centre prior on, aux diameter 0.5, lr 1e-3,
100 epochs -> seed 0 val AUC 0.955 (diameter alone 0.938).

Usage (outputs go to ../results/runs):
  python 05_train.py --seed 0
  python 05_train.py --seed 1
  python 05_train.py --seed 2
"""
import argparse
import faulthandler
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from gauge_lib import HU_WINDOW, RESULTS_DIR, PATCH_SIZE, PATCH_SPACING_MM, STORE_SIZE, npy_layout, read_npy_rows
from gauge_model import build_model

MAX_SHIFT = (STORE_SIZE - PATCH_SIZE) // 2   # +-8 voxels = +-8 mm at 1 mm


def roc_auc(y, s):
    """Mann-Whitney AUC with ties counted as 1/2."""
    y = np.asarray(y).astype(bool)
    s = np.asarray(s, float)
    pos, neg = s[y], s[~y]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order))
    ranks[order] = np.arange(1, len(order) + 1)
    allv = np.concatenate([pos, neg])
    for v in np.unique(allv):          # average ranks of ties
        m = allv == v
        if m.sum() > 1:
            ranks[m] = ranks[m].mean()
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def check_alignment(path, nod, n=200, seed=0):
    """Recompute the patch-centre HU from patches.npy and compare with nodules.csv (written by 04 from the
    same patch before storing). A mismatch means patch rows and nodule rows are out of order."""
    if "centre_hu" not in nod:
        return None
    lo, hi = HU_WINDOW
    g = (np.arange(STORE_SIZE) - (STORE_SIZE - 1) / 2.0) * PATCH_SPACING_MM
    zz, yy, xx = np.meshgrid(g, g, g, indexing="ij", sparse=True)
    d2 = zz ** 2 + yy ** 2 + xx ** 2
    rows = nod.iloc[np.random.default_rng(seed).choice(len(nod), min(n, len(nod)), replace=False)]
    X = read_npy_rows(path, rows["patch_index"].to_numpy())
    got, want = [], []
    for j, r in enumerate(rows.itertuples()):
        p = X[j].astype(np.float32)
        p = p / 255.0 * (hi - lo) + lo if X.dtype == np.uint8 else p * (hi - lo) + lo
        got.append(p[d2 <= max(1.5, 0.35 * r.eq_sphere_diam_mm) ** 2].mean())
        want.append(np.clip(r.centre_hu, lo, hi))
    got, want = np.array(got), np.array(want)
    return dict(n=len(got), corr=float(np.corrcoef(got, want)[0, 1]), median_abs_diff_hu=float(np.median(np.abs(got - want))))


def check_gpu():
    if not torch.cuda.is_available():
        print("WARNING: CUDA not available - training on CPU will be very slow.")
        return torch.device("cpu")
    cap = torch.cuda.get_device_capability(0)
    archs = torch.cuda.get_arch_list()
    name = torch.cuda.get_device_name(0)
    print(f"GPU: {name}, capability {cap}, torch {torch.__version__} (CUDA {torch.version.cuda}), archs {archs}")
    if cap >= (12, 0) and f"sm_{cap[0]}{cap[1]}" not in archs:
        raise SystemExit(f"This PyTorch build has no kernels for sm_{cap[0]}{cap[1]} ({name}). Install a "
                         f"CUDA 12.8+ build, e.g. pip install torch --index-url https://download.pytorch.org/whl/cu128")
    return torch.device("cuda")


def make_split(nod, val_frac, split_seed):
    """Patient-level split, stratified on whether the patient has any malignant nodule."""
    rng = np.random.default_rng(split_seed)
    pat = nod.groupby("PatientID")["label"].max()
    val = []
    for lab in (0, 1):
        p = pat[pat == lab].index.to_numpy()
        p = p[rng.permutation(len(p))]
        val += list(p[:int(round(val_frac * len(p)))])
    return pd.DataFrame({"PatientID": pat.index, "split": ["val" if p in set(val) else "train" for p in pat.index]})


def augment(x, gen, size=PATCH_SIZE):
    """x: (B, 1, STORE^3) on device -> (B, 1, PATCH^3): random shift crop, flips, intensity jitter."""
    b = x.shape[0]
    base = (STORE_SIZE - size) // 2 - MAX_SHIFT  # 0 for the full 128^3 input
    off = (base + torch.randint(0, 2 * MAX_SHIFT + 1, (b, 3), generator=gen, device="cpu")).tolist()
    out = torch.stack([x[i, :, o[0]:o[0] + size, o[1]:o[1] + size, o[2]:o[2] + size]
                       for i, o in enumerate(off)])
    flips = torch.rand((b, 3), generator=gen) < 0.5
    for i in range(b):
        dims = [d + 1 for d in range(3) if flips[i, d]]  # out[i] is (1, z, y, x)
        if dims:
            out[i] = torch.flip(out[i], dims=dims)
    scale = 1 + 0.05 * (2 * torch.rand((b, 1, 1, 1, 1), generator=gen) - 1)
    shift = 0.03 * (2 * torch.rand((b, 1, 1, 1, 1), generator=gen) - 1)
    return out * scale.to(out.device) + shift.to(out.device)


def to_float(t):
    """Stored patches (uint8 0-255, or float in [0, 1]) -> float32 in [0, 1]."""
    return t.float() / 255.0 if t.dtype == torch.uint8 else t.float()


def center(x, size=PATCH_SIZE):
    o = (x.shape[-1] - size) // 2
    return x[:, :, o:o + size, o:o + size, o:o + size]


@torch.no_grad()
def predict(model, X, idx, device, batch, amp, size=PATCH_SIZE):
    model.eval()
    out = []
    for i in range(0, len(idx), batch):
        xb = to_float(torch.from_numpy(np.ascontiguousarray(X[idx[i:i + batch]]))[:, None].to(device))
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
            out.append(model(center(xb, size)).float().cpu())
    return torch.cat(out).numpy()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--patches", default=RESULTS_DIR / "lidc_patches", type=Path, help="output folder of 04_lidc_patches.py")
    ap.add_argument("--out", default=RESULTS_DIR / "runs", type=Path)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--min-readers", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--aux-diameter-weight", type=float, default=0.5,
                    help="weight of an auxiliary log-diameter regression head (0 = off); training aid only")
    ap.add_argument("--batch", type=int, default=16, help="16 fits 128^3 on a 12 GB GPU; lower if out of memory")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--arch", default="resnet", choices=["resnet", "densenet"], help="model family (journal: densenet)")
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--no-amp", action="store_true", help="disable bf16 autocast")
    ap.add_argument("--max-train", type=int, help="subsample the training set (smoke tests, overfit check)")
    ap.add_argument("--no-aug", action="store_true", help="centre crop only (overfit check: train AUC should reach ~1)")
    ap.add_argument("--no-centre-prior", dest="centre_prior", action="store_false",
                    help="disable the fixed radial-distance channel (on by default: without it the network cannot "
                         "tell which object in the 128 mm FOV is the target; val AUC 0.75 vs 0.955)")
    ap.add_argument("--input-size", type=int, default=PATCH_SIZE,
                    help=f"model input in voxels (default {PATCH_SIZE} = 128 mm FOV; 64 = 64 mm FOV control)")
    args = ap.parse_args()
    faulthandler.enable()  # a native crash (no Python traceback) still prints where it happened

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    device = check_gpu()
    amp = device.type == "cuda" and not args.no_amp

    nod = pd.read_csv(args.patches / "nodules.csv")
    xpath = args.patches / "patches.npy"
    xshape, _, _ = npy_layout(xpath)  # header only: patches are read row by row (no memory map)
    assert tuple(xshape[1:]) == (STORE_SIZE,) * 3, f"patches are {xshape[1:]}, expected {STORE_SIZE}^3"
    nod = nod[(nod["n_readers"] >= args.min_readers) & (nod["malignancy_median"] != 3)].copy()
    nod["label"] = (nod["malignancy_median"] > 3).astype(int)

    args.out.mkdir(parents=True, exist_ok=True)
    split_path = args.out / "split.csv"
    if split_path.exists():
        split = pd.read_csv(split_path)
        missing = set(nod["PatientID"]) - set(split["PatientID"])
        if missing:
            raise SystemExit(f"{split_path} does not cover {len(missing)} patients - delete it or use a new --out")
    else:
        split = make_split(nod, args.val_frac, args.split_seed)
        split.to_csv(split_path, index=False)
    nod = nod.merge(split, on="PatientID")
    tr = nod[nod["split"] == "train"]
    va = nod[nod["split"] == "val"]
    if args.max_train:
        tr = tr.sample(min(args.max_train, len(tr)), random_state=args.seed)
    print(f"Nodules: train {len(tr)} ({tr.label.sum()} malignant, {tr.PatientID.nunique()} patients), "
          f"val {len(va)} ({va.label.sum()} malignant, {va.PatientID.nunique()} patients)")
    if va.label.nunique() < 2 or tr.label.nunique() < 2:
        raise SystemExit("Both classes are needed in train and val.")
    print("Checking patch/row alignment ...", flush=True)
    al = check_alignment(xpath, nod)
    if al is not None:
        print(f"Patch/row alignment check: corr {al['corr']:.3f}, median |diff| {al['median_abs_diff_hu']:.0f} HU "
              f"over {al['n']} nodules")
        if al["corr"] < 0.9:
            raise SystemExit("patches.npy rows do not match nodules.csv - rerun 04_lidc_patches.py")
    diam_auc = roc_auc(va["label"], va["diameter_mm"])
    if args.input_size + 2 * MAX_SHIFT > STORE_SIZE or args.input_size % 8:
        raise SystemExit(f"--input-size must be a multiple of 8 and <= {STORE_SIZE - 2 * MAX_SHIFT}")
    print(f"Patches: {STORE_SIZE}^3 stored, {args.input_size}^3 at {PATCH_SPACING_MM:g} mm used, "
          f"centre prior {'on' if args.centre_prior else 'off'}. "
          f"Diameter-only val AUC (what the CNN should beat): {diam_auc:.3f}")

    # Train and validation patches are read once into RAM (uint8, ~3 MB each); batches go to the GPU.
    t_read = time.time()
    Xtr = torch.from_numpy(read_npy_rows(xpath, tr["patch_index"].to_numpy()))
    Xva = read_npy_rows(xpath, va["patch_index"].to_numpy())
    print(f"Read {len(tr) + len(va)} patches in {time.time() - t_read:.0f} s", flush=True)
    ytr = torch.tensor(tr["label"].to_numpy(), dtype=torch.float32)
    logd = np.log(tr["diameter_mm"].to_numpy())
    d_mu, d_sd = float(logd.mean()), float(logd.std())
    dtr = torch.tensor((logd - d_mu) / d_sd, dtype=torch.float32)
    n_aux = 1 if args.aux_diameter_weight > 0 else 0
    va_idx = np.arange(len(va))  # rows of Xva

    model = build_model(width=args.width, dropout=args.dropout, input_size=args.input_size, n_aux=n_aux,
                        centre_prior=args.centre_prior, arch=args.arch).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = int(np.ceil(len(tr) / args.batch))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, epochs=args.epochs,
                                                steps_per_epoch=steps_per_epoch, pct_start=0.1)
    pos_weight = torch.tensor((len(tr) - tr.label.sum()) / max(tr.label.sum(), 1), device=device)
    gen = torch.Generator().manual_seed(args.seed)

    run = args.out / f"seed{args.seed}"
    run.mkdir(parents=True, exist_ok=True)
    cfg = dict(vars(args), patches=str(args.patches), out=str(args.out))
    model_cfg = dict(width=args.width, dropout=args.dropout, input_size=args.input_size, n_aux=n_aux,
                     centre_prior=args.centre_prior, arch=args.arch)
    hist, best = [], (-1.0, None)
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(len(tr), generator=gen)
        tot, n = 0.0, 0
        for i in range(0, len(perm), args.batch):
            b = perm[i:i + args.batch]
            if len(b) < 2:
                continue  # BatchNorm needs > 1 sample
            xb = to_float(Xtr[b].to(device, non_blocking=True))[:, None]
            yb = ytr[b].to(device)
            xb = center(xb, args.input_size) if args.no_aug else augment(xb, gen, args.input_size)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
                out = model.forward_all(xb).float()
            loss = F.binary_cross_entropy_with_logits(out[:, 0], yb, pos_weight=pos_weight)
            if n_aux:
                loss = loss + args.aux_diameter_weight * F.smooth_l1_loss(out[:, 1], dtr[b].to(device))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item() * len(b)
            n += len(b)
        lv = predict(model, Xva, va_idx, device, args.batch, amp, args.input_size)
        yv = va["label"].to_numpy()
        vloss = float(F.binary_cross_entropy_with_logits(torch.tensor(lv), torch.tensor(yv, dtype=torch.float32),
                                                         pos_weight=pos_weight.cpu()))
        auc = roc_auc(yv, lv)
        hist.append(dict(epoch=ep, train_loss=tot / max(n, 1), val_loss=vloss, val_auc=auc,
                         lr=sched.get_last_lr()[0], minutes=(time.time() - t0) / 60))
        print(f"seed {args.seed} epoch {ep:3d}  train(+aux) {tot / max(n, 1):.4f}  val {vloss:.4f}  AUC {auc:.3f}")
        state = dict(model=model.state_dict(), model_cfg=model_cfg, config=cfg, epoch=ep, val_auc=auc, val_loss=vloss)
        if auc > best[0]:
            best = (auc, ep)
            torch.save(state, run / "best.pt")
            pd.DataFrame({"patch_index": va["patch_index"].to_numpy(), "PatientID": va["PatientID"].to_numpy(), "label": yv,
                          "logit": lv, "diameter_mm": va["diameter_mm"].to_numpy()}
                         ).to_csv(run / "val_predictions.csv", index=False)
        pd.DataFrame(hist).to_csv(run / "history.csv", index=False)
    torch.save(state, run / "last.pt")

    ck = torch.load(run / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])
    train_auc = roc_auc(tr["label"], predict(model, Xtr.numpy(), np.arange(len(tr)), device, args.batch, amp,
                                             args.input_size))
    vp = pd.read_csv(run / "val_predictions.csv")
    strata = {}
    for name, m in (("le10mm", vp["diameter_mm"] <= 10), ("gt10mm", vp["diameter_mm"] > 10)):
        strata[f"val_auc_{name}"] = roc_auc(vp.label[m], vp.logit[m])
        strata[f"diameter_only_val_auc_{name}"] = roc_auc(vp.label[m], vp.diameter_mm[m])
        strata[f"n_val_{name}"] = int(m.sum())
    summary = dict(seed=args.seed, best_epoch=best[1], best_val_auc=best[0], last_val_auc=hist[-1]["val_auc"],
                   train_auc_best_model=train_auc, diameter_only_val_auc=diam_auc,
                   **strata, patch_alignment=al,
                   n_train=len(tr), n_val=len(va), minutes=(time.time() - t0) / 60, torch=torch.__version__,
                   device=torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu")
    (run / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
