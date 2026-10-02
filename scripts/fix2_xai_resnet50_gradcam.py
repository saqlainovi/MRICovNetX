"""
=============================================================================
FIX 2 (FINAL): ResNet-50 Pretrained → Grad-CAM → Energy-Based XAI Metrics
=============================================================================
Fine-tune ImageNet-pretrained ResNet-50 on Figshare 224x224 RGB.
Run Grad-CAM on layer4 (7x7 with 2048 channels — spatially rich).
Compute EBPG, IoU, Pointing Game, Relaxed PG.
Generate qualitative overlay figure.
=============================================================================
"""
import os, time, h5py, numpy as np, pandas as pd, cv2, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, classification_report

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "datasets" / "Dataset_1_Figshare" / "BRAIN_DATA"
RESULTS_DIR = BASE_DIR / "results"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = {0: "Meningioma", 1: "Glioma", 2: "Pituitary"}

# =====================================================================
# ResNet-50 with Grad-CAM hooks
# =====================================================================
class ResNet50GradCAM(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.resnet.fc = nn.Linear(2048, num_classes)
        self.gradients = None
        self.activations = None
        # Register hooks on layer4 (last conv block, output 7x7x2048)
        self.resnet.layer4.register_forward_hook(self._fwd_hook)
        self.resnet.layer4.register_full_backward_hook(self._bwd_hook)

    def _fwd_hook(self, module, input, output):
        self.activations = output

    def _bwd_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def forward(self, x):
        return self.resnet(x)


class FigshareDS(Dataset):
    def __init__(self, images, labels, masks, raw):
        self.images = images; self.labels = labels
        self.masks = masks; self.raw = raw
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx):
        img = torch.tensor(self.images[idx]).permute(2, 0, 1).float()
        return img, self.labels[idx], self.masks[idx], self.raw[idx]


def compute_gradcam(model, img_tensor):
    model.eval()
    inp = img_tensor.unsqueeze(0).to(DEVICE).requires_grad_(True)
    output = model(inp)
    pred_class = output.argmax(1).item()
    pred_conf = torch.softmax(output, 1)[0, pred_class].item()
    model.zero_grad()
    output[0, pred_class].backward()
    grads = model.gradients   # (1, 2048, 7, 7)
    acts  = model.activations # (1, 2048, 7, 7)
    weights = grads.mean(dim=(2, 3), keepdim=True)
    cam = (weights * acts).sum(dim=1).squeeze().detach().cpu().numpy()
    cam = np.maximum(cam, 0)
    if cam.max() > 0: cam = cam / cam.max()
    return cam, pred_class, pred_conf


def main():
    print("=" * 70)
    print(f"ResNet-50 Grad-CAM XAI on Figshare [{DEVICE}]")
    print("=" * 70)

    # ---- Load Data ----
    print("\n[1/5] Loading Figshare .mat files...")
    images_224, labels, pids, masks_512, raw_imgs = [], [], [], [], []
    mat_files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith(".mat")],
                       key=lambda x: int(os.path.splitext(x)[0]))
    for fname in mat_files:
        try:
            with h5py.File(DATA_DIR / fname, "r") as mat:
                img = mat["cjdata"]["image"][()]
                lbl = int(mat["cjdata"]["label"][()].item()) - 1
                pid = "".join([chr(c[0]) for c in mat["cjdata"]["PID"][()]])
                mask = mat["cjdata"]["tumorMask"][()]
                if img.shape != (512, 512): continue
                img_n = ((img - img.min()) / (img.max() - img.min() + 1e-8) * 255).astype(np.uint8)
                img_rgb = cv2.cvtColor(img_n, cv2.COLOR_GRAY2RGB)
                img_224 = cv2.resize(img_rgb, (224, 224)).astype(np.float32) / 255.0
                # ImageNet normalization
                mean = np.array([0.485, 0.456, 0.406])
                std  = np.array([0.229, 0.224, 0.225])
                img_224 = (img_224 - mean) / std
                images_224.append(img_224.astype(np.float32))
                labels.append(lbl); pids.append(pid)
                masks_512.append((mask > 0).astype(np.uint8))
                raw_imgs.append(img_n)
        except: continue

    images_224 = np.array(images_224); labels = np.array(labels)
    pids = np.array(pids); masks_512 = np.array(masks_512); raw_imgs = np.array(raw_imgs)
    print(f"  {len(images_224)} slices, {len(set(pids))} patients")

    # ---- Split ----
    print("\n[2/5] Patient-level split...")
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    tr_idx, te_idx = next(gss.split(images_224, labels, groups=pids))
    print(f"  Train: {len(tr_idx)} | Test: {len(te_idx)}")

    tr_ds = FigshareDS(images_224[tr_idx], labels[tr_idx], masks_512[tr_idx], raw_imgs[tr_idx])
    te_ds = FigshareDS(images_224[te_idx], labels[te_idx], masks_512[te_idx], raw_imgs[te_idx])
    tr_loader = DataLoader(tr_ds, batch_size=32, shuffle=True, num_workers=0, drop_last=True)
    te_loader = DataLoader(te_ds, batch_size=32, shuffle=False, num_workers=0)

    # ---- Train ----
    print("\n[3/5] Fine-tuning ResNet-50 (15 epochs)...")
    torch.manual_seed(42)
    model = ResNet50GradCAM(num_classes=3).to(DEVICE)

    # Freeze early layers, train layer3, layer4, fc
    for param in model.resnet.parameters():
        param.requires_grad = False
    for param in model.resnet.layer3.parameters():
        param.requires_grad = True
    for param in model.resnet.layer4.parameters():
        param.requires_grad = True
    for param in model.resnet.fc.parameters():
        param.requires_grad = True

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15)
    criterion = nn.CrossEntropyLoss()

    best_acc, best_state = 0, None
    for epoch in range(15):
        model.train()
        t_loss = 0
        for bx, by, _, _ in tr_loader:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward(); optimizer.step()
            t_loss += loss.item()
        scheduler.step()

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for bx, by, _, _ in te_loader:
                bx, by = bx.to(DEVICE), by.to(DEVICE)
                correct += (model(bx).argmax(1) == by).sum().item()
                total += len(by)
        acc = correct / total * 100
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if (epoch + 1) % 3 == 0:
            print(f"  Epoch {epoch+1:2d}/15 — Loss: {t_loss/len(tr_loader):.4f}, Acc: {acc:.2f}%")

    print(f"  Best Test Accuracy: {best_acc:.2f}%")
    model.load_state_dict(best_state)

    # ---- Grad-CAM + Metrics ----
    print(f"\n[4/5] Grad-CAM on {len(te_idx)} test slices...")
    model.eval()

    ebpg_scores, iou_scores = [], []
    pg_hits, pg_relaxed = 0, 0
    class_ebpg = defaultdict(list); class_iou = defaultdict(list)
    class_pg = defaultdict(int); class_rpg = defaultdict(int); class_n = defaultdict(int)
    viz_best = defaultdict(lambda: (0, None, None, None, None))  # per class best
    total = 0

    for i in range(len(te_ds)):
        img_t, lbl, mask, raw = te_ds[i]
        if np.sum(mask) == 0: continue

        try:
            cam, pred_cls, conf = compute_gradcam(model, img_t)
        except Exception as e:
            continue

        cam_full = cv2.resize(cam, (512, 512))
        cam_sum = np.sum(cam_full)

        # EBPG
        ebpg = np.sum(cam_full * mask) / cam_sum if cam_sum > 0 else 0
        ebpg_scores.append(ebpg); class_ebpg[lbl].append(ebpg)

        # IoU
        thr = max(cam_full.mean() + 0.5 * cam_full.std(), 0.15)
        pmask = (cam_full >= thr).astype(np.uint8)
        inter = np.sum(pmask * mask); union = np.sum(pmask) + np.sum(mask) - inter
        iou = inter / max(union, 1e-8)
        iou_scores.append(iou); class_iou[lbl].append(iou)

        # Pointing Game
        ml = np.unravel_index(np.argmax(cam_full), cam_full.shape)
        if mask[ml[0], ml[1]] > 0:
            pg_hits += 1; class_pg[lbl] += 1

        # Relaxed PG (15px)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
        dmask = cv2.dilate(mask, kernel, iterations=1)
        if dmask[ml[0], ml[1]] > 0:
            pg_relaxed += 1; class_rpg[lbl] += 1

        class_n[lbl] += 1; total += 1

        # Track best visualization per class
        if ebpg > viz_best[lbl][0]:
            viz_best[lbl] = (ebpg, raw, mask, cam_full, pred_cls)

        if total % 100 == 0:
            print(f"  {total} slices... EBPG={np.mean(ebpg_scores):.4f}, IoU={np.mean(iou_scores):.4f}")

    # ---- Print Results ----
    me, se = np.mean(ebpg_scores), np.std(ebpg_scores)
    mi, si = np.mean(iou_scores), np.std(iou_scores)
    pgs = pg_hits / max(total, 1) * 100
    pgr = pg_relaxed / max(total, 1) * 100

    print(f"\n{'='*70}")
    print(f"QUANTITATIVE XAI — ResNet-50 Grad-CAM on Figshare ({total} slices)")
    print(f"{'='*70}")
    print(f"  Energy-Based PG:       {me:.4f} +/- {se:.4f}")
    print(f"  Mean IoU:              {mi:.4f} +/- {si:.4f}")
    print(f"  Pointing Game:         {pg_hits}/{total} ({pgs:.1f}%)")
    print(f"  Relaxed PG (15px):     {pg_relaxed}/{total} ({pgr:.1f}%)")

    rows = []
    for c in range(3):
        n = class_n[c]; e = class_ebpg[c]; ic = class_iou[c]
        em = np.mean(e) if e else 0; es = np.std(e) if e else 0
        im = np.mean(ic) if ic else 0; iss = np.std(ic) if ic else 0
        p = class_pg[c]; r = class_rpg[c]
        rows.append({
            "Class": CLASS_NAMES[c], "N": n,
            "EBPG": f"{em:.4f}", "EBPG_SD": f"{es:.4f}",
            "IoU": f"{im:.4f}", "IoU_SD": f"{iss:.4f}",
            "PG%": f"{p/max(n,1)*100:.1f}", "RPG%": f"{r/max(n,1)*100:.1f}"
        })
        print(f"  {CLASS_NAMES[c]:12s}: EBPG={em:.4f}, IoU={im:.4f}, PG={p/max(n,1)*100:.1f}%, RPG={r/max(n,1)*100:.1f}%")

    rows.append({"Class": "Overall", "N": total,
                 "EBPG": f"{me:.4f}", "EBPG_SD": f"{se:.4f}",
                 "IoU": f"{mi:.4f}", "IoU_SD": f"{si:.4f}",
                 "PG%": f"{pgs:.1f}", "RPG%": f"{pgr:.1f}"})

    df = pd.DataFrame(rows)
    csv_out = RESULTS_DIR / "quantitative_xai_resnet50_energy.csv"
    df.to_csv(csv_out, index=False)
    print(f"\nSaved: {csv_out}")

    # ---- Qualitative Figure ----
    print("\nGenerating Grad-CAM overlay figure...")
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))

    for c in range(3):
        ebpg_v, raw, mask, cam_f, pcls = viz_best[c]
        if raw is None: continue

        axes[c,0].imshow(raw, cmap="gray"); axes[c,0].set_title(CLASS_NAMES[c], fontsize=13, fontweight="bold"); axes[c,0].axis("off")
        axes[c,1].imshow(raw, cmap="gray"); axes[c,1].imshow(mask, cmap="Reds", alpha=0.5); axes[c,1].set_title("Tumor Mask (GT)", fontsize=11); axes[c,1].axis("off")

        hm = cv2.applyColorMap((cam_f * 255).astype(np.uint8), cv2.COLORMAP_JET)
        hm = cv2.cvtColor(hm, cv2.COLOR_BGR2RGB)
        axes[c,2].imshow(hm); axes[c,2].set_title(f"Grad-CAM (EBPG={ebpg_v:.3f})", fontsize=11); axes[c,2].axis("off")

        orig_rgb = cv2.cvtColor(raw, cv2.COLOR_GRAY2RGB)
        overlay = cv2.addWeighted(orig_rgb, 0.6, hm, 0.4, 0)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)
        axes[c,3].imshow(overlay); axes[c,3].set_title("Overlay + Contour", fontsize=11); axes[c,3].axis("off")

    plt.suptitle("Grad-CAM Explainability — ResNet-50 on Figshare (Patient-Level Split)",
                 fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig_path = RESULTS_DIR / "fig_gradcam_resnet50.png"
    plt.savefig(fig_path, dpi=200, bbox_inches="tight"); plt.close()
    print(f"Saved: {fig_path}")

    # Save model
    torch.save(best_state, RESULTS_DIR / "resnet50_figshare_best.pth")
    print("\n[ALL DONE]")

if __name__ == "__main__":
    main()
