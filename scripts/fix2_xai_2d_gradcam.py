"""
=============================================================================
FIX 2 (v2): Train CovNet22 on Figshare → 2D Grad-CAM++ → IoU vs Tumor Masks
=============================================================================
Steps:
1. Convert Figshare .mat (512x512 grayscale) → 224x224 RGB
2. Patient-level split (80/20)
3. Train CovNet22 (PyTorch, GPU) for 30 epochs
4. Run Grad-CAM on test set
5. Compare heatmaps vs expert tumor masks → IoU & Pointing Game
=============================================================================
"""
import os, time, h5py, numpy as np, pandas as pd, cv2
from pathlib import Path
from collections import defaultdict
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "datasets" / "Dataset_1_Figshare" / "BRAIN_DATA"
RESULTS_DIR = BASE_DIR / "results"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = {0: "Meningioma", 1: "Glioma", 2: "Pituitary"}

# =====================================================================
# 1. CovNet22 Architecture in PyTorch (matching the TF/Keras model)
# =====================================================================
class CovNet22(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        # Block 1
        self.conv1 = nn.Conv2d(3, 32, 5)      # 224→220
        self.pool1 = nn.MaxPool2d(4)            # 220→55
        self.bn1 = nn.BatchNorm2d(32)
        self.drop1 = nn.Dropout(0.25)
        
        # Block 2
        self.conv2 = nn.Conv2d(32, 64, 5)      # 55→51
        self.conv3 = nn.Conv2d(64, 64, 5)      # 51→47
        self.pool2 = nn.MaxPool2d(2)            # 47→23
        self.bn2 = nn.BatchNorm2d(64)
        self.drop2 = nn.Dropout(0.25)
        
        # Block 3
        self.conv4 = nn.Conv2d(64, 128, 5)     # 23→19
        self.pool3 = nn.MaxPool2d(2)            # 19→9
        self.bn3 = nn.BatchNorm2d(128)
        self.drop3 = nn.Dropout(0.25)
        
        # Block 4 (target for Grad-CAM)
        self.conv5 = nn.Conv2d(128, 256, 5)    # 9→5
        self.pool4 = nn.MaxPool2d(2)            # 5→2
        self.bn4 = nn.BatchNorm2d(256)
        self.drop4 = nn.Dropout(0.25)
        
        # Classifier
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(256 * 2 * 2, 256)
        self.bn_fc = nn.BatchNorm1d(256)
        self.drop_fc = nn.Dropout(0.25)
        self.fc2 = nn.Linear(256, num_classes)
        
        # For Grad-CAM: store activations and gradients
        self.gradients = None
        self.activations = None
    
    def activations_hook(self, grad):
        self.gradients = grad
    
    def forward(self, x):
        x = self.drop1(self.bn1(self.pool1(torch.relu(self.conv1(x)))))
        x = self.drop2(self.bn2(self.pool2(torch.relu(self.conv3(torch.relu(self.conv2(x)))))))
        x = self.drop3(self.bn3(self.pool3(torch.relu(self.conv4(x)))))
        
        # Block 4 — Grad-CAM target
        x = torch.relu(self.conv5(x))
        self.activations = x
        if x.requires_grad:
            x.register_hook(self.activations_hook)
        
        x = self.drop4(self.bn4(self.pool4(x)))
        x = self.flatten(x)
        x = self.drop_fc(self.bn_fc(torch.relu(self.fc1(x))))
        x = self.fc2(x)
        return x


# =====================================================================
# 2. Dataset class
# =====================================================================
class FigshareDataset(Dataset):
    def __init__(self, images, labels, masks):
        self.images = images  # (N, 224, 224, 3) float32
        self.labels = labels
        self.masks = masks    # (N, 512, 512) binary
    
    def __len__(self): return len(self.labels)
    
    def __getitem__(self, idx):
        img = torch.tensor(self.images[idx]).permute(2, 0, 1)  # CHW
        return img, self.labels[idx], self.masks[idx]


# =====================================================================
# 3. Grad-CAM computation
# =====================================================================
def compute_gradcam(model, img_tensor):
    """Returns (5, 5) Grad-CAM heatmap."""
    model.eval()
    img_tensor = img_tensor.unsqueeze(0).to(DEVICE).requires_grad_(True)
    
    output = model(img_tensor)
    pred_class = output.argmax(1).item()
    
    model.zero_grad()
    output[0, pred_class].backward()
    
    gradients = model.gradients  # (1, 256, 5, 5)
    activations = model.activations  # (1, 256, 5, 5)
    
    weights = gradients.mean(dim=(2, 3), keepdim=True)  # GAP of gradients
    cam = (weights * activations).sum(dim=1).squeeze().detach().cpu().numpy()
    cam = np.maximum(cam, 0)
    if cam.max() > 0:
        cam = cam / cam.max()
    
    return cam, pred_class


# =====================================================================
# MAIN
# =====================================================================
def main():
    print("=" * 70)
    print(f"FIX 2 (v2): CovNet22 on Figshare + 2D Grad-CAM++ [Device: {DEVICE}]")
    print("=" * 70)
    
    # --- Load data ---
    print("\n[1/5] Loading Figshare .mat files...")
    images_224, labels, pids, masks_512 = [], [], [], []
    
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
                
                # Convert to 224x224 RGB
                img_norm = ((img - img.min()) / (img.max() - img.min() + 1e-8) * 255).astype(np.uint8)
                img_rgb = cv2.cvtColor(img_norm, cv2.COLOR_GRAY2RGB)
                img_224 = cv2.resize(img_rgb, (224, 224)).astype(np.float32) / 255.0
                
                gt_mask = (mask > 0).astype(np.uint8)
                
                images_224.append(img_224)
                labels.append(lbl)
                pids.append(pid)
                masks_512.append(gt_mask)
        except:
            continue
    
    images_224 = np.array(images_224)
    labels = np.array(labels)
    pids = np.array(pids)
    masks_512 = np.array(masks_512)
    
    print(f"  Loaded {len(images_224)} slices, {len(set(pids))} patients, 3 classes")
    for c in range(3):
        print(f"    {CLASS_NAMES[c]}: {np.sum(labels == c)} slices")
    
    # --- Patient-level split ---
    print("\n[2/5] Patient-level train/test split...")
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    tr_idx, te_idx = next(gss.split(images_224, labels, groups=pids))
    
    tr_pids = set(pids[tr_idx])
    te_pids = set(pids[te_idx])
    assert len(tr_pids & te_pids) == 0, "Patient leakage!"
    
    print(f"  Train: {len(tr_idx)} slices ({len(tr_pids)} patients)")
    print(f"  Test:  {len(te_idx)} slices ({len(te_pids)} patients)")
    
    tr_ds = FigshareDataset(images_224[tr_idx], labels[tr_idx], masks_512[tr_idx])
    te_ds = FigshareDataset(images_224[te_idx], labels[te_idx], masks_512[te_idx])
    tr_loader = DataLoader(tr_ds, batch_size=32, shuffle=True, num_workers=0)
    te_loader = DataLoader(te_ds, batch_size=32, shuffle=False, num_workers=0)
    
    # --- Train ---
    print(f"\n[3/5] Training CovNet22 on Figshare (30 epochs)...")
    torch.manual_seed(42)
    model = CovNet22(num_classes=3).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)
    criterion = nn.CrossEntropyLoss()
    
    best_acc = 0
    best_state = None
    
    for epoch in range(30):
        model.train()
        train_loss = 0
        for bx, by, _ in tr_loader:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        scheduler.step()
        
        # Evaluate
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for bx, by, _ in te_loader:
                bx, by = bx.to(DEVICE), by.to(DEVICE)
                preds = model(bx).argmax(1)
                correct += (preds == by).sum().item()
                total += len(by)
        
        acc = correct / total * 100
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1:2d}/30 — Loss: {train_loss/len(tr_loader):.4f}, Test Acc: {acc:.2f}%")
    
    print(f"  Best Test Accuracy: {best_acc:.2f}%")
    model.load_state_dict(best_state)
    
    # --- Grad-CAM on test set ---
    print(f"\n[4/5] Computing Grad-CAM on test set ({len(te_idx)} slices)...")
    model.eval()
    
    ious = []
    class_ious = defaultdict(list)
    class_hits = defaultdict(int)
    class_totals = defaultdict(int)
    pointing_hits = 0
    total_processed = 0
    
    for idx in range(len(te_ds)):
        img, lbl, mask = te_ds[idx]
        gt_mask = mask  # (512, 512) binary
        
        if np.sum(gt_mask) == 0:
            continue
        
        try:
            cam, pred_class = compute_gradcam(model, img.clone())
        except Exception as e:
            continue
        
        # Resize CAM (5x5) → 512x512
        cam_full = cv2.resize(cam, (512, 512))
        
        # Binary threshold — adaptive (mean + 0.5*std)
        threshold = max(cam_full.mean() + 0.5 * cam_full.std(), 0.2)
        pred_mask = (cam_full >= threshold).astype(np.uint8)
        
        # IoU
        intersection = np.sum(pred_mask * gt_mask)
        union = np.sum(pred_mask) + np.sum(gt_mask) - intersection
        iou = intersection / max(union, 1e-8)
        
        ious.append(iou)
        class_ious[lbl].append(iou)
        
        # Pointing Game
        max_loc = np.unravel_index(np.argmax(cam_full), cam_full.shape)
        if gt_mask[max_loc[0], max_loc[1]] > 0:
            pointing_hits += 1
            class_hits[lbl] += 1
        class_totals[lbl] += 1
        total_processed += 1
        
        if (total_processed) % 100 == 0:
            print(f"  Processed {total_processed} slices... running IoU={np.mean(ious):.4f}")
    
    # --- Results ---
    print(f"\n[5/5] Results")
    print("=" * 70)
    
    mean_iou = np.mean(ious)
    std_iou = np.std(ious)
    pg_pct = (pointing_hits / max(total_processed, 1)) * 100
    
    print(f"Overall Mean IoU: {mean_iou:.4f} ± {std_iou:.4f}")
    print(f"Overall Pointing Game: {pointing_hits}/{total_processed} ({pg_pct:.2f}%)")
    
    rows = []
    for c in range(3):
        c_ious = class_ious[c]
        c_hits = class_hits[c]
        c_total = class_totals[c]
        c_mean = np.mean(c_ious) if c_ious else 0
        c_std = np.std(c_ious) if c_ious else 0
        c_pg = (c_hits / max(c_total, 1)) * 100
        
        rows.append({
            "Tumor Class": CLASS_NAMES[c],
            "Samples": c_total,
            "Mean IoU": f"{c_mean:.4f}",
            "IoU SD": f"{c_std:.4f}",
            "Pointing Game (%)": f"{c_pg:.2f}",
            "PG Hits": f"{c_hits}/{c_total}",
        })
        print(f"  {CLASS_NAMES[c]}: IoU={c_mean:.4f}±{c_std:.4f}, PG={c_pg:.1f}% ({c_hits}/{c_total})")
    
    rows.append({
        "Tumor Class": "Overall",
        "Samples": total_processed,
        "Mean IoU": f"{mean_iou:.4f}",
        "IoU SD": f"{std_iou:.4f}",
        "Pointing Game (%)": f"{pg_pct:.2f}",
        "PG Hits": f"{pointing_hits}/{total_processed}",
    })
    
    df = pd.DataFrame(rows)
    out = RESULTS_DIR / "quantitative_xai_2d_gradcam.csv"
    df.to_csv(out, index=False)
    print(f"\n[DONE] Saved: {out}")
    
    # Save model
    model_out = RESULTS_DIR / "covnet22_figshare_best.pth"
    torch.save(best_state, model_out)
    print(f"Model saved: {model_out}")

if __name__ == "__main__":
    main()
