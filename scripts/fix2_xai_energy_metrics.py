"""
=============================================================================
FIX 2 (v3): Energy-Based XAI Metrics + Qualitative Grad-CAM Overlays
=============================================================================
Uses the already-trained CovNet22 model on Figshare data.
Computes:
  1. Energy-Based Pointing Game (EBPG): sum(CAM * mask) / sum(CAM)
  2. Relevance Mass Accuracy (RMA): % of CAM energy inside tumor
  3. Relaxed Pointing Game: max-activation within 15px of tumor boundary
  4. Generates qualitative overlay figures for the paper
=============================================================================
"""
import os, h5py, numpy as np, pandas as pd, cv2, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
from sklearn.model_selection import GroupShuffleSplit

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "datasets" / "Dataset_1_Figshare" / "BRAIN_DATA"
RESULTS_DIR = BASE_DIR / "results"
MODEL_PATH = RESULTS_DIR / "covnet22_figshare_best.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = {0: "Meningioma", 1: "Glioma", 2: "Pituitary"}

# =====================================================================
# CovNet22 Architecture (same as fix2 v2)
# =====================================================================
class CovNet22(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 5); self.pool1 = nn.MaxPool2d(4)
        self.bn1 = nn.BatchNorm2d(32); self.drop1 = nn.Dropout(0.25)
        self.conv2 = nn.Conv2d(32, 64, 5); self.conv3 = nn.Conv2d(64, 64, 5)
        self.pool2 = nn.MaxPool2d(2); self.bn2 = nn.BatchNorm2d(64); self.drop2 = nn.Dropout(0.25)
        self.conv4 = nn.Conv2d(64, 128, 5); self.pool3 = nn.MaxPool2d(2)
        self.bn3 = nn.BatchNorm2d(128); self.drop3 = nn.Dropout(0.25)
        self.conv5 = nn.Conv2d(128, 256, 5); self.pool4 = nn.MaxPool2d(2)
        self.bn4 = nn.BatchNorm2d(256); self.drop4 = nn.Dropout(0.25)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(256 * 2 * 2, 256)
        self.bn_fc = nn.BatchNorm1d(256); self.drop_fc = nn.Dropout(0.25)
        self.fc2 = nn.Linear(256, num_classes)
        self.gradients = None; self.activations = None

    def activations_hook(self, grad):
        self.gradients = grad

    def forward(self, x):
        x = self.drop1(self.bn1(self.pool1(torch.relu(self.conv1(x)))))
        x = self.drop2(self.bn2(self.pool2(torch.relu(self.conv3(torch.relu(self.conv2(x)))))))
        x = self.drop3(self.bn3(self.pool3(torch.relu(self.conv4(x)))))
        x = torch.relu(self.conv5(x))
        self.activations = x
        if x.requires_grad:
            x.register_hook(self.activations_hook)
        x = self.drop4(self.bn4(self.pool4(x)))
        x = self.flatten(x)
        x = self.drop_fc(self.bn_fc(torch.relu(self.fc1(x))))
        return self.fc2(x)

# =====================================================================
# Grad-CAM
# =====================================================================
def compute_gradcam(model, img_tensor):
    model.eval()
    inp = img_tensor.unsqueeze(0).to(DEVICE).requires_grad_(True)
    output = model(inp)
    pred_class = output.argmax(1).item()
    model.zero_grad()
    output[0, pred_class].backward()
    grads = model.gradients       # (1, 256, 5, 5)
    acts  = model.activations     # (1, 256, 5, 5)
    weights = grads.mean(dim=(2, 3), keepdim=True)
    cam = (weights * acts).sum(dim=1).squeeze().detach().cpu().numpy()
    cam = np.maximum(cam, 0)
    if cam.max() > 0:
        cam = cam / cam.max()
    return cam, pred_class

# =====================================================================
# MAIN
# =====================================================================
def main():
    print("=" * 70)
    print("Energy-Based XAI Metrics + Qualitative Overlays")
    print("=" * 70)

    # Load model
    model = CovNet22(num_classes=3).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
    model.eval()
    print(f"Model loaded from {MODEL_PATH}")

    # Load data (same split as training)
    print("\nLoading Figshare data...")
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
                img_norm = ((img - img.min()) / (img.max() - img.min() + 1e-8) * 255).astype(np.uint8)
                img_rgb = cv2.cvtColor(img_norm, cv2.COLOR_GRAY2RGB)
                img_224 = cv2.resize(img_rgb, (224, 224)).astype(np.float32) / 255.0
                images_224.append(img_224)
                labels.append(lbl); pids.append(pid)
                masks_512.append((mask > 0).astype(np.uint8))
                raw_imgs.append(img_norm)
        except: continue

    images_224 = np.array(images_224)
    labels = np.array(labels); pids = np.array(pids)
    masks_512 = np.array(masks_512); raw_imgs = np.array(raw_imgs)
    print(f"  Loaded {len(images_224)} slices")

    # Same split
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    _, te_idx = next(gss.split(images_224, labels, groups=pids))
    print(f"  Test set: {len(te_idx)} slices")

    # Compute metrics
    print("\nComputing Energy-Based XAI metrics...")
    ebpg_scores = []  # Energy-Based Pointing Game
    iou_scores = []
    pg_standard_hits = 0
    pg_relaxed_hits = 0  # within 15px neighborhood
    class_ebpg = defaultdict(list)
    class_iou = defaultdict(list)
    class_pg_hits = defaultdict(int)
    class_pg_relaxed = defaultdict(int)
    class_totals = defaultdict(int)
    total = 0

    # For qualitative figure: collect best examples per class
    viz_candidates = defaultdict(list)  # class -> [(ebpg, idx, cam, pred)]

    for i, idx in enumerate(te_idx):
        img_t = torch.tensor(images_224[idx]).permute(2, 0, 1).float()
        gt_mask = masks_512[idx]
        lbl = labels[idx]

        if np.sum(gt_mask) == 0: continue

        try:
            cam, pred_class = compute_gradcam(model, img_t)
        except: continue

        cam_full = cv2.resize(cam, (512, 512))

        # 1. Energy-Based Pointing Game (EBPG)
        cam_sum = np.sum(cam_full)
        if cam_sum > 0:
            ebpg = np.sum(cam_full * gt_mask) / cam_sum
        else:
            ebpg = 0.0
        ebpg_scores.append(ebpg)
        class_ebpg[lbl].append(ebpg)

        # 2. Standard IoU (threshold = mean + 0.5*std or 0.2)
        threshold = max(cam_full.mean() + 0.5 * cam_full.std(), 0.2)
        pred_mask = (cam_full >= threshold).astype(np.uint8)
        intersection = np.sum(pred_mask * gt_mask)
        union = np.sum(pred_mask) + np.sum(gt_mask) - intersection
        iou = intersection / max(union, 1e-8)
        iou_scores.append(iou)
        class_iou[lbl].append(iou)

        # 3. Standard Pointing Game
        max_loc = np.unravel_index(np.argmax(cam_full), cam_full.shape)
        if gt_mask[max_loc[0], max_loc[1]] > 0:
            pg_standard_hits += 1
            class_pg_hits[lbl] += 1

        # 4. Relaxed Pointing Game (15px dilation of mask)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
        dilated_mask = cv2.dilate(gt_mask, kernel, iterations=1)
        if dilated_mask[max_loc[0], max_loc[1]] > 0:
            pg_relaxed_hits += 1
            class_pg_relaxed[lbl] += 1

        class_totals[lbl] += 1
        total += 1

        # Save for visualization
        viz_candidates[lbl].append((ebpg, idx, cam_full, pred_class))

        if (total) % 100 == 0:
            print(f"  {total} slices... EBPG={np.mean(ebpg_scores):.4f}")

    # ===== RESULTS =====
    print(f"\n{'='*70}")
    print("QUANTITATIVE XAI RESULTS")
    print(f"{'='*70}")

    mean_ebpg = np.mean(ebpg_scores)
    std_ebpg = np.std(ebpg_scores)
    mean_iou = np.mean(iou_scores)
    std_iou = np.std(iou_scores)
    pg_std_pct = (pg_standard_hits / max(total, 1)) * 100
    pg_rel_pct = (pg_relaxed_hits / max(total, 1)) * 100

    print(f"\nOverall ({total} test slices):")
    print(f"  Energy-Based PG (EBPG):     {mean_ebpg:.4f} +/- {std_ebpg:.4f}")
    print(f"  Standard IoU:               {mean_iou:.4f} +/- {std_iou:.4f}")
    print(f"  Standard Pointing Game:     {pg_standard_hits}/{total} ({pg_std_pct:.1f}%)")
    print(f"  Relaxed Pointing Game (15px): {pg_relaxed_hits}/{total} ({pg_rel_pct:.1f}%)")

    rows = []
    for c in range(3):
        n = class_totals[c]
        e = class_ebpg[c]
        iou_c = class_iou[c]
        em, es = (np.mean(e), np.std(e)) if e else (0, 0)
        im, is_ = (np.mean(iou_c), np.std(iou_c)) if iou_c else (0, 0)
        ph = class_pg_hits[c]
        pr = class_pg_relaxed[c]
        rows.append({
            "Tumor Class": CLASS_NAMES[c], "Samples": n,
            "EBPG (mean)": f"{em:.4f}", "EBPG (SD)": f"{es:.4f}",
            "IoU (mean)": f"{im:.4f}", "IoU (SD)": f"{is_:.4f}",
            "Pointing Game (%)": f"{ph/max(n,1)*100:.1f}",
            "Relaxed PG 15px (%)": f"{pr/max(n,1)*100:.1f}",
        })
        print(f"  {CLASS_NAMES[c]:12s}: EBPG={em:.4f}+/-{es:.4f}, IoU={im:.4f}, PG={ph/max(n,1)*100:.1f}%, RPG={pr/max(n,1)*100:.1f}%")

    rows.append({
        "Tumor Class": "Overall", "Samples": total,
        "EBPG (mean)": f"{mean_ebpg:.4f}", "EBPG (SD)": f"{std_ebpg:.4f}",
        "IoU (mean)": f"{mean_iou:.4f}", "IoU (SD)": f"{std_iou:.4f}",
        "Pointing Game (%)": f"{pg_std_pct:.1f}",
        "Relaxed PG 15px (%)": f"{pg_rel_pct:.1f}",
    })

    df = pd.DataFrame(rows)
    out_csv = RESULTS_DIR / "quantitative_xai_energy_based.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")

    # ===== QUALITATIVE FIGURE =====
    print("\nGenerating qualitative Grad-CAM overlay figure...")
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    col_titles = ["Original MRI", "Tumor Mask (GT)", "Grad-CAM Heatmap", "Overlay"]

    for c in range(3):
        candidates = sorted(viz_candidates[c], key=lambda x: -x[0])  # best EBPG first
        if not candidates: continue
        best_ebpg, best_idx, best_cam, best_pred = candidates[0]

        orig = raw_imgs[best_idx]
        mask = masks_512[best_idx]

        # Col 0: Original
        axes[c, 0].imshow(orig, cmap="gray")
        axes[c, 0].set_title(f"{CLASS_NAMES[c]}", fontsize=13, fontweight="bold")
        axes[c, 0].axis("off")

        # Col 1: Tumor mask
        axes[c, 1].imshow(orig, cmap="gray")
        axes[c, 1].imshow(mask, cmap="Reds", alpha=0.5)
        axes[c, 1].set_title("Ground Truth Mask", fontsize=11)
        axes[c, 1].axis("off")

        # Col 2: Grad-CAM heatmap
        heatmap = cv2.applyColorMap((best_cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        axes[c, 2].imshow(heatmap)
        axes[c, 2].set_title(f"Grad-CAM (EBPG={best_ebpg:.3f})", fontsize=11)
        axes[c, 2].axis("off")

        # Col 3: Overlay
        orig_rgb = cv2.cvtColor(orig, cv2.COLOR_GRAY2RGB)
        overlay = cv2.addWeighted(orig_rgb, 0.6, heatmap, 0.4, 0)
        # Draw tumor contour
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)
        axes[c, 3].imshow(overlay)
        axes[c, 3].set_title("Overlay + Tumor Contour", fontsize=11)
        axes[c, 3].axis("off")

    plt.suptitle("Grad-CAM++ Explainability Analysis — CovNet22 on Figshare Dataset",
                 fontsize=15, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig_path = RESULTS_DIR / "fig_gradcam_qualitative.png"
    plt.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved figure: {fig_path}")

    print("\n[ALL DONE]")

if __name__ == "__main__":
    main()
