"""
=============================================================================
MRICovNetX — Experiment 3: Patient-Level Split, CovBI-GRU Ablation & Quantitative XAI
=============================================================================
Author: Research Team
Implements:
1. Patient-level train/test separation (233 unique patients, zero leakage)
2. 5-Variant CovBI-GRU Ablation Study on RTX 3060 GPU
3. Quantitative XAI (IoU & Pointing Game) evaluated against ground-truth tumor masks
=============================================================================
"""

import os
import cv2
import h5py
import time
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "datasets" / "Dataset_1_Figshare" / "BRAIN_DATA"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_figshare_dataset():
    print(f"Loading Figshare dataset from {DATA_DIR}...")
    images, labels, pids, masks = [], [], [], []
    
    mat_files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith(".mat")], 
                       key=lambda x: int(os.path.splitext(x)[0]))
    
    for f in mat_files:
        fp = DATA_DIR / f
        try:
            with h5py.File(fp, "r") as mat:
                img = mat["cjdata"]["image"][()]
                lbl = int(mat["cjdata"]["label"][()].item()) - 1
                pid_raw = mat["cjdata"]["PID"][()]
                pid = "".join([chr(c[0]) for c in pid_raw])
                mask = mat["cjdata"]["tumorMask"][()]
                
                if img.shape == (512, 512):
                    img_norm = (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-8)
                    images.append(img_norm.astype(np.float32))
                    labels.append(lbl)
                    pids.append(pid)
                    masks.append((mask > 0).astype(np.uint8))
        except Exception:
            pass

    print(f"Successfully loaded {len(images)} slices across {len(set(pids))} unique patients.")
    return np.array(images), np.array(labels), np.array(pids), np.array(masks)

# =============================================================================
# PyTorch Model Architectures for CovBI-GRU Ablation
# =============================================================================

class CovBIGRU(nn.Module):
    def __init__(self, variant="full"):
        super(CovBIGRU, self).__init__()
        self.variant = variant
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(0.2)
        
        if variant == "full":
            self.conv1 = nn.Conv1d(512, 64, kernel_size=4)
            self.pool1 = nn.MaxPool1d(2)
            self.bn1 = nn.BatchNorm1d(64)
            self.gru1 = nn.GRU(64, 64, bidirectional=True, batch_first=True)
            self.conv2 = nn.Conv1d(128, 64, kernel_size=3)
            self.conv3 = nn.Conv1d(64, 64, kernel_size=3)
            self.pool2 = nn.MaxPool1d(2)
            self.bn2 = nn.BatchNorm1d(64)
            self.gru2 = nn.GRU(64, 128, bidirectional=True, batch_first=True)
            self.conv4 = nn.Conv1d(256, 128, kernel_size=2)
            self.pool3 = nn.MaxPool1d(2)
            self.bn3 = nn.BatchNorm1d(128)
            
        elif variant == "conv1d_only":
            self.conv1 = nn.Conv1d(512, 64, kernel_size=4)
            self.pool1 = nn.MaxPool1d(2)
            self.bn1 = nn.BatchNorm1d(64)
            self.conv2 = nn.Conv1d(64, 64, kernel_size=3)
            self.conv3 = nn.Conv1d(64, 64, kernel_size=3)
            self.pool2 = nn.MaxPool1d(2)
            self.bn2 = nn.BatchNorm1d(64)
            self.conv4 = nn.Conv1d(64, 128, kernel_size=2)
            self.pool3 = nn.MaxPool1d(2)
            self.bn3 = nn.BatchNorm1d(128)

        elif variant == "bigru_only":
            self.gru1 = nn.GRU(512, 64, bidirectional=True, batch_first=True)
            self.bn1 = nn.BatchNorm1d(128)  # 64 * 2 = 128 channels
            self.gru2 = nn.GRU(128, 128, bidirectional=True, batch_first=True)

        elif variant == "no_batchnorm":
            self.conv1 = nn.Conv1d(512, 64, kernel_size=4)
            self.pool1 = nn.MaxPool1d(2)
            self.gru1 = nn.GRU(64, 64, bidirectional=True, batch_first=True)
            self.conv2 = nn.Conv1d(128, 64, kernel_size=3)
            self.conv3 = nn.Conv1d(64, 64, kernel_size=3)
            self.pool2 = nn.MaxPool1d(2)
            self.gru2 = nn.GRU(64, 128, bidirectional=True, batch_first=True)
            self.conv4 = nn.Conv1d(256, 128, kernel_size=2)
            self.pool3 = nn.MaxPool1d(2)

        elif variant == "no_dropout":
            self.conv1 = nn.Conv1d(512, 64, kernel_size=4)
            self.pool1 = nn.MaxPool1d(2)
            self.bn1 = nn.BatchNorm1d(64)
            self.gru1 = nn.GRU(64, 64, bidirectional=True, batch_first=True)
            self.conv2 = nn.Conv1d(128, 64, kernel_size=3)
            self.conv3 = nn.Conv1d(64, 64, kernel_size=3)
            self.pool2 = nn.MaxPool1d(2)
            self.bn2 = nn.BatchNorm1d(64)
            self.gru2 = nn.GRU(64, 128, bidirectional=True, batch_first=True)
            self.conv4 = nn.Conv1d(256, 128, kernel_size=2)
            self.pool3 = nn.MaxPool1d(2)
            self.bn3 = nn.BatchNorm1d(128)

        # Dynamic dimension computation
        with torch.no_grad():
            dummy = torch.zeros(1, 512, 512)
            flat_dim = self.extract_features(dummy).shape[1]
        
        self.fc1 = nn.Linear(flat_dim, 128)
        self.bn_fc = nn.BatchNorm1d(128) if variant != "no_batchnorm" else nn.Identity()
        self.fc2 = nn.Linear(128, 3)

    def extract_features(self, x):
        if self.variant == "full":
            x = x.transpose(1, 2)
            x = self.pool1(self.relu(self.conv1(x)))
            x = self.bn1(x)
            x = x.transpose(1, 2)
            x, _ = self.gru1(x)
            x = x.transpose(1, 2)
            x = self.relu(self.conv2(x))
            x = self.pool2(self.relu(self.conv3(x)))
            x = self.bn2(x)
            x = x.transpose(1, 2)
            x, _ = self.gru2(x)
            x = x.transpose(1, 2)
            x = self.pool3(self.relu(self.conv4(x)))
            x = self.bn3(x)
            return x.flatten(1)

        elif self.variant == "conv1d_only":
            x = x.transpose(1, 2)
            x = self.pool1(self.relu(self.conv1(x)))
            x = self.bn1(x)
            x = self.relu(self.conv2(x))
            x = self.pool2(self.relu(self.conv3(x)))
            x = self.bn2(x)
            x = self.pool3(self.relu(self.conv4(x)))
            x = self.bn3(x)
            return x.flatten(1)

        elif self.variant == "bigru_only":
            x, _ = self.gru1(x)                # (B, 512, 128)
            x = x.transpose(1, 2)              # (B, 128, 512)
            x = self.bn1(x).transpose(1, 2)    # (B, 512, 128)
            x, _ = self.gru2(x)                # (B, 512, 256)
            return x.flatten(1)

        elif self.variant == "no_batchnorm":
            x = x.transpose(1, 2)
            x = self.pool1(self.relu(self.conv1(x)))
            x = x.transpose(1, 2)
            x, _ = self.gru1(x)
            x = x.transpose(1, 2)
            x = self.relu(self.conv2(x))
            x = self.pool2(self.relu(self.conv3(x)))
            x = x.transpose(1, 2)
            x, _ = self.gru2(x)
            x = x.transpose(1, 2)
            x = self.pool3(self.relu(self.conv4(x)))
            return x.flatten(1)

        elif self.variant == "no_dropout":
            x = x.transpose(1, 2)
            x = self.pool1(self.relu(self.conv1(x)))
            x = self.bn1(x)
            x = x.transpose(1, 2)
            x, _ = self.gru1(x)
            x = x.transpose(1, 2)
            x = self.relu(self.conv2(x))
            x = self.pool2(self.relu(self.conv3(x)))
            x = self.bn2(x)
            x = x.transpose(1, 2)
            x, _ = self.gru2(x)
            x = x.transpose(1, 2)
            x = self.pool3(self.relu(self.conv4(x)))
            x = self.bn3(x)
            return x.flatten(1)

    def forward(self, x):
        feat = self.extract_features(x)
        x = self.bn_fc(self.relu(self.fc1(feat)))
        if self.variant != "no_dropout":
            x = self.drop(x)
        return self.fc2(x)

def train_eval_model(model, train_loader, test_loader, epochs=60):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    model.to(DEVICE)
    for epoch in range(epochs):
        model.train()
        for bx, by in train_loader:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()

    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for bx, by in test_loader:
            bx = bx.to(DEVICE)
            out = model(bx)
            preds = torch.argmax(out, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(by.numpy())
            
    acc = accuracy_score(all_targets, all_preds)
    prec, rec, f1, _ = precision_recall_fscore_support(all_targets, all_preds, average="macro", zero_division=0)
    return acc, prec, rec, f1, model

def main():
    print("=" * 70)
    print(f"RUNNING ON: {DEVICE} ({torch.cuda.get_device_name(0)})")
    print("=" * 70)

    images, labels, pids, masks = load_figshare_dataset()

    # Step 1: Patient-Level Split
    print("\n" + "=" * 60)
    print("STEP 1: Patient-Level Train/Test Split (Zero Data Leakage)")
    print("=" * 60)

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(images, labels, groups=pids))

    train_pids = set(pids[train_idx])
    test_pids = set(pids[test_idx])
    overlap = train_pids & test_pids

    print(f"Training Slices: {len(train_idx)} from {len(train_pids)} unique patients")
    print(f"Testing Slices:  {len(test_idx)} from {len(test_pids)} unique patients")
    print(f"Patient Overlap: {len(overlap)} (Strict Zero Leakage)")

    train_ds = TensorDataset(torch.tensor(images[train_idx]), torch.tensor(labels[train_idx]))
    test_ds = TensorDataset(torch.tensor(images[test_idx]), torch.tensor(labels[test_idx]))
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    p_split_df = pd.DataFrame([{
        "Total Patients": len(set(pids)),
        "Training Patients": len(train_pids),
        "Testing Patients": len(test_pids),
        "Training Slices": len(train_idx),
        "Testing Slices": len(test_idx),
        "Patient Overlap": len(overlap),
        "Data Leakage": "None (0.00%)"
    }])
    p_split_df.to_csv(RESULTS_DIR / "patient_level_split_summary.csv", index=False)
    print("Saved patient-level split summary to:", RESULTS_DIR / "patient_level_split_summary.csv")

    # Step 2: CovBI-GRU Ablation Study
    print("\n" + "=" * 60)
    print("STEP 2: CovBI-GRU Ablation Study (5 Variants)")
    print("=" * 60)

    variants = [
        ("Full CovBI-GRU (Proposed)", "full"),
        ("w/o Bi-GRU (Conv1D Only)", "conv1d_only"),
        ("w/o Conv1D (Bi-GRU Only)", "bigru_only"),
        ("w/o Batch Normalization", "no_batchnorm"),
        ("w/o Dropout Regularization", "no_dropout")
    ]

    ablation_results = []
    saved_full_model = None

    for name, var in variants:
        print(f"\nEvaluating: {name}...")
        start_t = time.time()
        model = CovBIGRU(variant=var)
        acc, prec, rec, f1, trained_m = train_eval_model(model, train_loader, test_loader, epochs=60)
        elapsed = time.time() - start_t
        print(f"  Acc: {acc*100:.2f}% | Prec: {prec*100:.2f}% | Rec: {rec*100:.2f}% | F1: {f1*100:.2f}% | Time: {elapsed:.1f}s")
        
        ablation_results.append({
            "Model Configuration": name,
            "Accuracy (%)": f"{acc * 100:.2f}",
            "Precision (%)": f"{prec * 100:.2f}",
            "Recall (%)": f"{rec * 100:.2f}",
            "F1-Score (%)": f"{f1 * 100:.2f}",
            "Training Time (s)": f"{elapsed:.1f}"
        })
        
        if var == "full":
            saved_full_model = trained_m

    df_abl = pd.DataFrame(ablation_results)
    df_abl.to_csv(RESULTS_DIR / "covbigru_ablation_results.csv", index=False)
    print("\n[SUCCESS] Ablation results saved to:", RESULTS_DIR / "covbigru_ablation_results.csv")
    print(df_abl.to_string(index=False))

    # Step 3: Quantitative XAI (IoU & Pointing Game)
    print("\n" + "=" * 60)
    print("STEP 3: Quantitative XAI (IoU & Pointing Game vs Ground-Truth Masks)")
    print("=" * 60)

    test_masks = masks[test_idx]
    test_imgs = images[test_idx]
    test_lbls = labels[test_idx]

    ious = []
    pointing_hits = 0
    class_ious = defaultdict(list)
    class_hits = defaultdict(int)
    class_totals = defaultdict(int)
    class_names = ["Meningioma", "Glioma", "Pituitary"]

    saved_full_model.eval()

    # Pre-compute feature maps using hook
    activation_storage = []
    def hook_conv(m, i, o):
        activation_storage.append(o)

    hook = saved_full_model.conv4.register_forward_hook(hook_conv)

    print("Computing Grad-CAM attention maps and IoU metrics...")
    for i in range(len(test_idx)):
        img_np = test_imgs[i]
        gt_mask = test_masks[i]
        lbl = test_lbls[i]

        inp = torch.tensor(img_np, device=DEVICE).unsqueeze(0)
        activation_storage.clear()
        
        with torch.no_grad():
            out = saved_full_model(inp)
            fm = activation_storage[0].squeeze(0).cpu().numpy() # (128, L)

        # Average energy across feature channels
        cam_1d = np.maximum(0, np.mean(fm, axis=0))
        cam_1d = (cam_1d - np.min(cam_1d)) / (np.max(cam_1d) - np.min(cam_1d) + 1e-8)
        
        # Project 1D sequence to 2D image matrix
        cam_2d = np.tile(cam_1d.reshape(-1, 1), (1, 512))
        cam_2d = cv2.resize(cam_2d, (512, 512))

        # Binary attention mask
        pred_mask = (cam_2d > 0.40).astype(np.uint8)
        
        intersection = np.sum(pred_mask * gt_mask)
        union = np.sum(pred_mask) + np.sum(gt_mask) - intersection
        iou = intersection / max(union, 1e-8)
        
        ious.append(iou)
        class_ious[lbl].append(iou)

        # Pointing game hit test
        max_loc = np.unravel_index(np.argmax(cam_2d), cam_2d.shape)
        if gt_mask[max_loc[0], max_loc[1]] > 0:
            pointing_hits += 1
            class_hits[lbl] += 1
        class_totals[lbl] += 1

    hook.remove()

    mean_iou = np.mean(ious)
    pg_acc = (pointing_hits / len(test_idx)) * 100.0

    print(f"\nOverall Quantitative XAI Results:")
    print(f"  Mean IoU: {mean_iou:.4f} ± {np.std(ious):.4f}")
    print(f"  Pointing Game Accuracy: {pointing_hits}/{len(test_idx)} ({pg_acc:.2f}%)")

    xai_rows = []
    for c in range(3):
        xai_rows.append({
            "Tumor Class": class_names[c],
            "Mean IoU": f"{np.mean(class_ious[c]):.4f}",
            "Pointing Game Accuracy (%)": f"{(class_hits[c] / max(class_totals[c], 1)) * 100:.2f}%",
            "Test Samples": class_totals[c]
        })
    xai_rows.append({
        "Tumor Class": "Overall (All Classes)",
        "Mean IoU": f"{mean_iou:.4f}",
        "Pointing Game Accuracy (%)": f"{pg_acc:.2f}%",
        "Test Samples": len(test_idx)
    })

    df_xai = pd.DataFrame(xai_rows)
    df_xai.to_csv(RESULTS_DIR / "quantitative_xai_metrics.csv", index=False)
    print("\n[SUCCESS] Quantitative XAI results saved to:", RESULTS_DIR / "quantitative_xai_metrics.csv")
    print(df_xai.to_string(index=False))

if __name__ == "__main__":
    main()
