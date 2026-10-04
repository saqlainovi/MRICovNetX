"""
=============================================================================
MRICovNetX - Experiment 7: Definitive CovNet22 Evaluation (D2, D3, D4)
=============================================================================
Leakage-free re-evaluation protocol used for every CovNet22 number reported
in the revised manuscript.

Protocol
--------
* Dataset-2 (Bhuvaji, Kaggle)  : provider's canonical Training / Testing split
* Dataset-3 (Nickparvar)       : provider's canonical Training / Testing split
* Dataset-4 (Mendeley)         : fixed stratified 85:15 train/test split
                                 (random_state=42 -> 5,147 / 909 images)
* Validation                   : stratified 10% of the TRAINING partition
                                 (re-drawn per seed). Early stopping, LR
                                 scheduling and checkpoint selection use the
                                 validation set ONLY. The test set is evaluated
                                 exactly once, after training.
* Seeds                        : 42, 123, 2024
* Optimiser                    : Adam (lr = 1e-3), batch size 32, max 60 epochs
* Callbacks                    : ReduceLROnPlateau(factor 0.5, patience 5) and
                                 EarlyStopping(patience 10) on validation loss,
                                 best-validation-loss weights restored
* Regularisation               : Dropout 10% (D2, D3); 30% conv / 50% dense (D4)
* Augmentation (train only)    : random zoom (+/-20%), shear (+/-0.2 rad),
                                 horizontal flip
* Duplicate audit              : 64-bit pHash; test images whose hash exactly
                                 matches any training-partition image are
                                 flagged, and accuracy is also reported on the
                                 de-duplicated ("clean") test subset.

Hardware: NVIDIA GeForce RTX 3060 (8 GB)

Usage
-----
    python 07_definitive_covnet22_eval.py --mode main
    python 07_definitive_covnet22_eval.py --mode ablation      # Dataset-3 only
=============================================================================
"""
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

import argparse
import json
import os
import time
from pathlib import Path

import imagehash
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             precision_recall_fscore_support, roc_auc_score,
                             roc_curve)
from sklearn.model_selection import train_test_split

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEEDS = [42, 123, 2024]
MAX_EPOCHS = 60
BATCH_SIZE = 32
LR = 1e-3
IMG = 224
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

DATA_ROOT = Path(os.environ.get("MRI_DATA_ROOT", r"C:\MRIData"))
OUT_ROOT = Path(os.environ.get("MRI_OUT_ROOT", r"C:\MRIData\results_v2"))

DATASETS = {
    "Dataset-2": dict(root="Dataset_2_Kaggle", split="canonical",
                      classes=["glioma_tumor", "meningioma_tumor", "no_tumor", "pituitary_tumor"],
                      labels=["Glioma", "Meningioma", "No tumor", "Pituitary"],
                      p_conv=0.10, p_fc=0.10),
    "Dataset-2-dedup": dict(root="Dataset_2_Kaggle", split="dedup_stratified_80_20",
                      classes=["glioma_tumor", "meningioma_tumor", "no_tumor", "pituitary_tumor"],
                      labels=["Glioma", "Meningioma", "No tumor", "Pituitary"],
                      p_conv=0.10, p_fc=0.10),
    "Dataset-3": dict(root="Dataset_3_Nickparvar", split="canonical",
                      classes=["glioma", "meningioma", "notumor", "pituitary"],
                      labels=["Glioma", "Meningioma", "No tumor", "Pituitary"],
                      p_conv=0.10, p_fc=0.10),
    "Dataset-4": dict(root="Dataset_4_Mendeley/Brain_Cancer", split="stratified_85_15",
                      classes=["brain_glioma", "brain_menin", "brain_tumor"],
                      labels=["Brain_Glioma", "Brain_Menin", "Brain Tumor"],
                      p_conv=0.30, p_fc=0.50),
}


# =============================================================================
# Model
# =============================================================================
class CovNet22(nn.Module):
    """CovNet22: five 5x5 Conv2D layers (BN, ReLU, max-pooling) + two dense layers.

    ~1.45 M trainable parameters for 4 classes.
    """

    def __init__(self, num_classes, p_conv=0.1, p_fc=0.1, use_bn=True, shallow=False):
        super().__init__()
        bn2 = (lambda c: nn.BatchNorm2d(c)) if use_bn else (lambda c: nn.Identity())
        self.shallow = shallow
        self.block1 = nn.Sequential(nn.Conv2d(3, 32, 5), nn.ReLU(inplace=True),
                                    nn.MaxPool2d(4), bn2(32), nn.Dropout(p_conv))
        self.block2 = nn.Sequential(nn.Conv2d(32, 64, 5), nn.ReLU(inplace=True),
                                    nn.Conv2d(64, 64, 5), nn.ReLU(inplace=True),
                                    nn.MaxPool2d(2), bn2(64), nn.Dropout(p_conv))
        if shallow:
            # Reduced-depth variant: blocks 3-4 removed, adaptive pooling keeps
            # the dense head identical (1024 -> 256 -> C)
            self.block3 = nn.Identity()
            self.block4 = nn.Sequential(nn.AdaptiveAvgPool2d(4))
            flat = 64 * 4 * 4
        else:
            self.block3 = nn.Sequential(nn.Conv2d(64, 128, 5), nn.ReLU(inplace=True),
                                        nn.MaxPool2d(2), bn2(128), nn.Dropout(p_conv))
            self.block4 = nn.Sequential(nn.Conv2d(128, 256, 5), nn.ReLU(inplace=True),
                                        nn.MaxPool2d(2), bn2(256), nn.Dropout(p_conv))
            flat = 256 * 2 * 2
        self.fc1 = nn.Linear(flat, 256)
        self.bn_fc = nn.BatchNorm1d(256) if use_bn else nn.Identity()
        self.drop_fc = nn.Dropout(p_fc)
        self.fc2 = nn.Linear(256, num_classes)

    def features(self, x):
        x = self.block4(self.block3(self.block2(self.block1(x))))
        x = torch.flatten(x, 1)
        return self.bn_fc(F.relu(self.fc1(x)))

    def forward(self, x):
        return self.fc2(self.drop_fc(self.features(x)))


def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


# =============================================================================
# Data
# =============================================================================
def _list_class_dir(base, cls):
    """Case-insensitive lookup of a class folder."""
    for d in os.listdir(base):
        if d.lower() == cls.lower():
            return Path(base) / d
    raise FileNotFoundError(f"class folder '{cls}' not found in {base}")


def load_folder(base, classes):
    imgs, labels, hashes, files = [], [], [], []
    for ci, cls in enumerate(classes):
        d = _list_class_dir(base, cls)
        for f in sorted(os.listdir(d)):
            if not f.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            im = Image.open(d / f).convert("RGB")
            hashes.append(str(imagehash.phash(im)))
            imgs.append(np.asarray(im.resize((IMG, IMG), Image.BILINEAR), dtype=np.uint8))
            labels.append(ci)
            files.append(f"{d.name}/{f}")
    x = torch.from_numpy(np.stack(imgs)).permute(0, 3, 1, 2).contiguous()  # N,3,H,W uint8
    return x, np.array(labels), np.array(hashes), np.array(files)


def load_dataset(name):
    cfg = DATASETS[name]
    root = DATA_ROOT / cfg["root"]
    t0 = time.time()
    if cfg["split"] == "canonical":
        xtr, ytr, htr, ftr = load_folder(root / "Training", cfg["classes"])
        xte, yte, hte, fte = load_folder(root / "Testing", cfg["classes"])
    elif cfg["split"] == "dedup_stratified_80_20":
        # Pool Training+Testing, drop exact pHash duplicates (keep first
        # occurrence; drop hash groups with conflicting labels), then a fixed
        # stratified 80:20 split (random_state=42).
        xa, ya, ha, fa = load_folder(root / "Training", cfg["classes"])
        xb, yb, hb, fb = load_folder(root / "Testing", cfg["classes"])
        x = torch.cat([xa, xb]); y = np.concatenate([ya, yb])
        h = np.concatenate([ha, hb]); f = np.concatenate([np.char.add("Training/", fa.astype(str)),
                                                          np.char.add("Testing/", fb.astype(str))])
        groups = {}
        for i, hh in enumerate(h):
            groups.setdefault(hh, []).append(i)
        keep = [g[0] for g in groups.values() if len(set(y[g])) == 1]
        n_conflict = sum(1 for g in groups.values() if len(set(y[g])) > 1)
        keep = np.array(sorted(keep))
        print(f"[{name}] pooled={len(y)} unique_hashes={len(groups)} conflicting_label_groups={n_conflict} kept={len(keep)}")
        x, y, h, f = x[keep], y[keep], h[keep], f[keep]
        itr, ite = train_test_split(np.arange(len(y)), test_size=0.20, stratify=y, random_state=42)
        xtr, ytr, htr, ftr = x[itr], y[itr], h[itr], f[itr]
        xte, yte, hte, fte = x[ite], y[ite], h[ite], f[ite]
    else:
        x, y, h, f = load_folder(root, cfg["classes"])
        idx = np.arange(len(y))
        itr, ite = train_test_split(idx, test_size=0.15, stratify=y, random_state=42)
        xtr, ytr, htr, ftr = x[itr], y[itr], h[itr], f[itr]
        xte, yte, hte, fte = x[ite], y[ite], h[ite], f[ite]
    print(f"[{name}] loaded train={len(ytr)} test={len(yte)} in {time.time()-t0:.1f}s")
    return dict(xtr=xtr, ytr=ytr, htr=htr, ftr=ftr, xte=xte, yte=yte, hte=hte, fte=fte)


def to_input(xb_uint8):
    x = xb_uint8.float().div_(255.0)
    return (x - MEAN.to(x.device)) / STD.to(x.device)


def augment(x):
    """GPU augmentation: random zoom (+/-20%), shear (+/-0.2 rad), h-flip."""
    n = x.shape[0]
    dev = x.device
    s = torch.empty(n, device=dev).uniform_(0.8, 1.2)
    sh = torch.tan(torch.empty(n, device=dev).uniform_(-0.2, 0.2))
    flip = (torch.rand(n, device=dev) < 0.5).float() * -2 + 1  # -1 or 1
    theta = torch.zeros(n, 2, 3, device=dev)
    theta[:, 0, 0] = s * flip
    theta[:, 0, 1] = s * sh
    theta[:, 1, 1] = s
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)


# =============================================================================
# Train / evaluate
# =============================================================================
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@torch.no_grad()
def predict(model, x_uint8, bs=128):
    model.eval()
    probs, feats = [], []
    for i in range(0, len(x_uint8), bs):
        xb = to_input(x_uint8[i:i + bs].to(DEVICE, non_blocking=True))
        f = model.features(xb)
        probs.append(F.softmax(model.fc2(model.drop_fc(f)), 1).cpu())
        feats.append(f.cpu())
    return torch.cat(probs).numpy(), torch.cat(feats).numpy()


def train_one(model, xtr, ytr, xva, yva, use_aug=True, log_prefix=""):
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=5)
    crit = nn.CrossEntropyLoss()
    xtr_g, ytr_g = xtr.to(DEVICE), torch.as_tensor(ytr, device=DEVICE)
    xva_g, yva_g = xva.to(DEVICE), torch.as_tensor(yva, device=DEVICE)
    n = len(ytr)
    best, best_state, wait, hist = float("inf"), None, 0, []
    for ep in range(1, MAX_EPOCHS + 1):
        model.train()
        perm = torch.randperm(n, device=DEVICE)
        tl, tc = 0.0, 0
        for i in range(0, n, BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            if len(idx) < 2:
                continue
            xb = to_input(xtr_g[idx])
            if use_aug:
                xb = augment(xb)
            yb = ytr_g[idx]
            opt.zero_grad(set_to_none=True)
            out = model(xb)
            loss = crit(out, yb)
            loss.backward()
            opt.step()
            tl += loss.item() * len(idx)
            tc += (out.argmax(1) == yb).sum().item()
        model.eval()
        vl, vc = 0.0, 0
        with torch.no_grad():
            for i in range(0, len(yva), 128):
                xb = to_input(xva_g[i:i + 128])
                out = model(xb)
                vl += crit(out, yva_g[i:i + 128]).item() * len(xb)
                vc += (out.argmax(1) == yva_g[i:i + 128]).sum().item()
        tl, ta, vl, va = tl / n, tc / n * 100, vl / len(yva), vc / len(yva) * 100
        lr_now = opt.param_groups[0]["lr"]
        hist.append(dict(epoch=ep, train_loss=tl, train_acc=ta, val_loss=vl, val_acc=va, lr=lr_now))
        sched.step(vl)
        if vl < best - 1e-6:
            best, wait = vl, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            best_ep = ep
        else:
            wait += 1
        print(f"{log_prefix} ep {ep:02d} | tr_loss {tl:.4f} tr_acc {ta:.2f} | "
              f"val_loss {vl:.4f} val_acc {va:.2f} | lr {lr_now:.1e}")
        if wait >= 10:
            print(f"{log_prefix} early stop at epoch {ep} (best epoch {best_ep})")
            break
    model.load_state_dict(best_state)
    del xtr_g, ytr_g, xva_g, yva_g
    torch.cuda.empty_cache()
    return hist, best_ep


def per_class_table(y, p, labels):
    cm = confusion_matrix(y, p, labels=list(range(len(labels))))
    N = cm.sum()
    rows = []
    for i, lab in enumerate(labels):
        tp = cm[i, i]
        fn = cm[i].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = N - tp - fn - fp
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        rows.append(dict(cls=lab, TP=int(tp), TN=int(tn), FP=int(fp), FN=int(fn),
                         accuracy=(tp + tn) / N * 100, precision=prec, recall=rec,
                         FPR=fp / max(fp + tn, 1), TNR=tn / max(fp + tn, 1),
                         F1=2 * prec * rec / max(prec + rec, 1e-12)))
    return cm, rows


@torch.no_grad()
def latency_ms(model, n=200):
    model.eval()
    x = torch.randn(1, 3, IMG, IMG, device=DEVICE)
    for _ in range(20):
        model(x)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        model(x)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1000


def run(name, data, seed, variant="full", save_artifacts=True):
    cfg = DATASETS[name]
    labels = cfg["labels"]
    set_seed(seed)
    itr, iva = train_test_split(np.arange(len(data["ytr"])), test_size=0.10,
                                stratify=data["ytr"], random_state=seed)
    p_conv, p_fc = cfg["p_conv"], cfg["p_fc"]
    use_bn, shallow, use_aug = True, False, True
    if variant == "no_dropout":
        p_conv = p_fc = 0.0
    elif variant == "no_bn":
        use_bn = False
    elif variant == "no_aug":
        use_aug = False
    elif variant == "shallow":
        shallow = True
    model = CovNet22(len(labels), p_conv, p_fc, use_bn=use_bn, shallow=shallow).to(DEVICE)
    tag = f"[{name}|{variant}|seed {seed}]"
    print(f"{tag} params={count_params(model):,} train={len(itr)} val={len(iva)} test={len(data['yte'])}")
    t0 = time.time()
    hist, best_ep = train_one(model, data["xtr"][itr], data["ytr"][itr],
                              data["xtr"][iva], data["ytr"][iva], use_aug=use_aug, log_prefix=tag)
    train_time = time.time() - t0

    probs, feats = predict(model, data["xte"])
    yte, pred = data["yte"], probs.argmax(1)
    acc = accuracy_score(yte, pred) * 100
    pr, rc, f1, _ = precision_recall_fscore_support(yte, pred, average="macro", zero_division=0)
    try:
        auc = roc_auc_score(yte, probs, multi_class="ovr", average="macro")
    except ValueError:
        auc = float("nan")

    # duplicate-aware evaluation
    train_hashes = set(data["htr"].tolist())
    dup = np.array([h in train_hashes for h in data["hte"]])
    clean = ~dup
    acc_clean = accuracy_score(yte[clean], pred[clean]) * 100 if clean.any() else float("nan")
    f1_clean = precision_recall_fscore_support(yte[clean], pred[clean], average="macro",
                                               zero_division=0)[2] * 100 if clean.any() else float("nan")

    cm, rows = per_class_table(yte, pred, labels)
    res = dict(dataset=name, variant=variant, seed=seed, params=count_params(model),
               n_train=int(len(itr)), n_val=int(len(iva)), n_test=int(len(yte)),
               accuracy=acc, precision=pr * 100, recall=rc * 100, f1=f1 * 100, auc_macro=auc,
               n_test_dup_of_train=int(dup.sum()), n_test_clean=int(clean.sum()),
               accuracy_clean=acc_clean, f1_clean=f1_clean,
               best_epoch=int(best_ep), epochs_run=len(hist), train_time_s=train_time,
               latency_ms_bs1=latency_ms(model))
    print(f"{tag} TEST acc={acc:.2f} F1={f1*100:.2f} AUC={auc:.4f} | clean({clean.sum()}) acc={acc_clean:.2f} "
          f"| best_ep={best_ep} time={train_time:.0f}s")

    if save_artifacts:
        od = OUT_ROOT / name.replace("-", "") / f"{variant}_seed{seed}"
        od.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(hist).to_csv(od / "history.csv", index=False)
        pd.DataFrame(rows).to_csv(od / "per_class.csv", index=False)
        np.savetxt(od / "confusion_matrix.csv", cm, fmt="%d", delimiter=",")
        np.save(od / "test_probs.npy", probs)
        np.save(od / "test_labels.npy", yte)
        if variant == "full":
            np.save(od / "test_features.npy", feats)
            torch.save(model.state_dict(), od / "covnet22.pt")
            roc = {}
            for i, lab in enumerate(labels):
                fpr, tpr, _ = roc_curve((yte == i).astype(int), probs[:, i])
                roc[lab] = dict(fpr=fpr.tolist(), tpr=tpr.tolist(),
                                auc=float(roc_auc_score((yte == i).astype(int), probs[:, i])))
            (od / "roc.json").write_text(json.dumps(roc))
        mis = np.where(pred != yte)[0]
        pd.DataFrame(dict(file=data["fte"][mis], true=[labels[i] for i in yte[mis]],
                          pred=[labels[i] for i in pred[mis]],
                          pred_conf=probs[mis, pred[mis]] * 100,
                          true_prob=probs[mis, yte[mis]] * 100,
                          duplicate_of_train=dup[mis])).to_csv(od / "misclassified.csv", index=False)
        (od / "result.json").write_text(json.dumps(res, indent=2))
    del model
    torch.cuda.empty_cache()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["main", "ablation"], default="main")
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--seeds", nargs="*", type=int, default=SEEDS)
    args = ap.parse_args()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"Device: {DEVICE} ({torch.cuda.get_device_name(0) if DEVICE.type == 'cuda' else 'cpu'}) "
          f"| torch {torch.__version__} | mode={args.mode}")

    if args.mode == "main":
        names = args.datasets or list(DATASETS)
        variants = ["full"]
        out_csv = OUT_ROOT / "covnet22_definitive_per_seed.csv"
    else:
        names = args.datasets or ["Dataset-3"]
        variants = ["no_aug", "no_dropout", "no_bn", "shallow"]
        out_csv = OUT_ROOT / "covnet22_ablation_d3_per_seed.csv"

    all_res = []
    if out_csv.exists():
        all_res = pd.read_csv(out_csv).to_dict("records")
    done = {(r["dataset"], r["variant"], int(r["seed"])) for r in all_res}

    for name in names:
        data = None
        for variant in variants:
            for seed in args.seeds:
                if (name, variant, seed) in done:
                    print(f"skip {name} {variant} {seed} (done)")
                    continue
                if data is None:
                    data = load_dataset(name)
                    tr_h = set(data["htr"].tolist())
                    print(f"[{name}] test images with exact pHash match in training partition: "
                          f"{sum(h in tr_h for h in data['hte'])}/{len(data['hte'])}")
                all_res.append(run(name, data, seed, variant))
                pd.DataFrame(all_res).to_csv(out_csv, index=False)
        del data

    df = pd.DataFrame(all_res)
    summ = (df.groupby(["dataset", "variant"])
              .agg(acc_mean=("accuracy", "mean"), acc_sd=("accuracy", "std"),
                   f1_mean=("f1", "mean"), f1_sd=("f1", "std"),
                   prec_mean=("precision", "mean"), rec_mean=("recall", "mean"),
                   auc_mean=("auc_macro", "mean"),
                   acc_clean_mean=("accuracy_clean", "mean"), acc_clean_sd=("accuracy_clean", "std"),
                   n_test=("n_test", "first"), n_test_clean=("n_test_clean", "first"),
                   n=("seed", "count")).reset_index())
    summ["ci95_low"] = summ.acc_mean - 1.96 * summ.acc_sd / np.sqrt(summ.n)
    summ["ci95_high"] = summ.acc_mean + 1.96 * summ.acc_sd / np.sqrt(summ.n)
    summ.to_csv(out_csv.with_name(out_csv.stem.replace("per_seed", "summary") + ".csv"), index=False)
    print(summ.to_string())


if __name__ == "__main__":
    main()
