"""
=============================================================================
MRICovNetX — Experiment 5: CovNet22 Multi-Seed Runs on D2, D3, D4
=============================================================================
Reviewer R3.5 asked for mean +/- SD on ALL datasets. This script trains CovNet22
on D2, D3, D4 with 3 random seeds each (9 total runs), producing a CSV with
mean+/-SD accuracy, precision, recall, F1 for each dataset.

Hardware: NVIDIA GeForce RTX 3060 (8 GB VRAM)
=============================================================================
"""

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import os
import time
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEEDS = [42, 123, 2024]
EPOCHS = 12
BATCH_SIZE = 32
LR = 0.001

# =====================================================================
# CovNet22 Architecture
# =====================================================================
class CovNet22(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 5)
        self.pool1 = nn.MaxPool2d(4)
        self.bn1 = nn.BatchNorm2d(32)
        self.drop1 = nn.Dropout(0.25)

        self.conv2 = nn.Conv2d(32, 64, 5)
        self.conv3 = nn.Conv2d(64, 64, 5)
        self.pool2 = nn.MaxPool2d(2)
        self.bn2 = nn.BatchNorm2d(64)
        self.drop2 = nn.Dropout(0.25)

        self.conv4 = nn.Conv2d(64, 128, 5)
        self.pool3 = nn.MaxPool2d(2)
        self.bn3 = nn.BatchNorm2d(128)
        self.drop3 = nn.Dropout(0.25)

        self.conv5 = nn.Conv2d(128, 256, 5)
        self.pool4 = nn.MaxPool2d(2)
        self.bn4 = nn.BatchNorm2d(256)
        self.drop4 = nn.Dropout(0.25)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(256 * 2 * 2, 256)
        self.bn_fc = nn.BatchNorm1d(256)
        self.drop_fc = nn.Dropout(0.25)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.drop1(self.bn1(self.pool1(torch.relu(self.conv1(x)))))
        x = self.drop2(self.bn2(self.pool2(torch.relu(self.conv3(torch.relu(self.conv2(x)))))))
        x = self.drop3(self.bn3(self.pool3(torch.relu(self.conv4(x)))))
        x = self.drop4(self.bn4(self.pool4(torch.relu(self.conv5(x)))))
        x = self.flatten(x)
        x = self.drop_fc(self.bn_fc(torch.relu(self.fc1(x))))
        x = self.fc2(x)
        return x


# =====================================================================
# Dataset Loaders
# =====================================================================
class ImageFolderDataset(Dataset):
    """Generic image folder dataset: root/class_name/*.jpg"""
    def __init__(self, folder, class_map, transform=None):
        self.samples = []
        self.transform = transform
        for cls_name, cls_idx in class_map.items():
            dir_p = Path(folder) / cls_name
            if dir_p.exists():
                for f in os.listdir(dir_p):
                    if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                        self.samples.append((dir_p / f, cls_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label


class TransformedSubset(Dataset):
    """Wrapper to apply specific transform to a subset."""
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        path, label = self.subset.dataset.samples[self.subset.indices[idx]]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label


# Dataset configurations
DATASETS = {
    "Dataset-2": {
        "path": BASE_DIR / "datasets" / "Dataset_2_Kaggle",
        "has_train_test": True,
        "classes": {
            'glioma_tumor': 0, 'meningioma_tumor': 1,
            'no_tumor': 2, 'pituitary_tumor': 3
        },
        "num_classes": 4
    },
    "Dataset-3": {
        "path": BASE_DIR / "datasets" / "Dataset_3_Nickparvar",
        "has_train_test": True,
        "classes": {
            'glioma': 0, 'meningioma': 1,
            'notumor': 2, 'pituitary': 3
        },
        "num_classes": 4
    },
    "Dataset-4": {
        "path": BASE_DIR / "datasets" / "Dataset_4_Mendeley",
        "has_train_test": False,
        "classes": {
            'brain_glioma': 0, 'brain_menin': 1, 'brain_tumor': 2
        },
        "num_classes": 3,
        "root_subfolder": "Brain_Cancer"
    },
}


def set_seed(seed):
    """Set all random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_transforms():
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return transform_train, transform_test


def train_and_evaluate(model, train_loader, test_loader, epochs=EPOCHS):
    """Train CovNet22 and return metrics."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float('inf')
    best_state = None
    patience = 5
    patience_counter = 0

    for ep in range(epochs):
        model.train()
        running_loss = 0.0
        for bx, by in train_loader:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        scheduler.step()

        # Validate
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for bx, by in test_loader:
                bx, by = bx.to(DEVICE), by.to(DEVICE)
                out = model(bx)
                val_loss += criterion(out, by).item()
                preds = torch.argmax(out, dim=1)
                correct += (preds == by).sum().item()
                total += len(by)

        avg_val_loss = val_loss / len(test_loader)
        val_acc = correct / max(total, 1) * 100.0

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        print(f"    Epoch {ep+1:02d}/{epochs:02d} | Train Loss: {running_loss/len(train_loader):.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2f}%")

        if patience_counter >= patience:
            print(f"    Early stopping triggered at epoch {ep+1}")
            break

    # Load best model
    if best_state is not None:
        model.load_state_dict(best_state)

    # Evaluate
    model.eval()
    all_preds, all_targets = [], []
    latencies = []

    with torch.no_grad():
        for bx, by in test_loader:
            bx = bx.to(DEVICE)
            t0 = time.time()
            out = model(bx)
            latencies.append((time.time() - t0) / len(bx))
            preds = torch.argmax(out, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(by.numpy())

    acc = accuracy_score(all_targets, all_preds) * 100.0
    prec, rec, f1, _ = precision_recall_fscore_support(
        all_targets, all_preds, average='macro', zero_division=0)
    avg_latency_ms = np.mean(latencies) * 1000.0

    return {
        'accuracy': acc,
        'precision': prec * 100.0,
        'recall': rec * 100.0,
        'f1': f1 * 100.0,
        'latency_ms': avg_latency_ms
    }


def main():
    print("=" * 70)
    print("EXPERIMENT 5: CovNet22 Multi-Seed Runs (D2, D3, D4)")
    print(f"Device: {DEVICE} | Seeds: {SEEDS} | Epochs: {EPOCHS} | LR: {LR}")
    print("=" * 70)

    transform_train, transform_test = get_transforms()
    all_results = []
    summary_rows = []

    for ds_name, ds_cfg in DATASETS.items():
        print(f"\n{'='*60}")
        print(f"  {ds_name}")
        print(f"{'='*60}")

        seed_results = []

        for seed in SEEDS:
            set_seed(seed)
            print(f"\n  Seed {seed}:")

            num_classes = ds_cfg["num_classes"]

            if ds_cfg["has_train_test"]:
                train_ds = ImageFolderDataset(
                    ds_cfg["path"] / "Training", ds_cfg["classes"],
                    transform=transform_train)
                test_ds = ImageFolderDataset(
                    ds_cfg["path"] / "Testing", ds_cfg["classes"],
                    transform=transform_test)
            else:
                # D4: single folder, split 80:20
                raw_ds = ImageFolderDataset(
                    ds_cfg["path"] / ds_cfg["root_subfolder"],
                    ds_cfg["classes"], transform=None)
                n_total = len(raw_ds)
                n_train = int(0.8 * n_total)
                n_test = n_total - n_train
                sub_tr, sub_te = random_split(
                    raw_ds, [n_train, n_test],
                    generator=torch.Generator().manual_seed(seed))
                train_ds = TransformedSubset(sub_tr, transform=transform_train)
                test_ds = TransformedSubset(sub_te, transform=transform_test)

            print(f"    Train: {len(train_ds)} | Test: {len(test_ds)}")

            train_loader = DataLoader(
                train_ds, batch_size=BATCH_SIZE, shuffle=True,
                num_workers=0, pin_memory=True)
            test_loader = DataLoader(
                test_ds, batch_size=BATCH_SIZE, shuffle=False,
                num_workers=0, pin_memory=True)

            model = CovNet22(num_classes=num_classes).to(DEVICE)
            t0 = time.time()
            metrics = train_and_evaluate(model, train_loader, test_loader)
            train_time = time.time() - t0

            metrics['seed'] = seed
            metrics['dataset'] = ds_name
            metrics['train_time_s'] = train_time
            seed_results.append(metrics)
            all_results.append(metrics)

            print(f"    [Result] Acc: {metrics['accuracy']:.2f}% | "
                  f"F1: {metrics['f1']:.2f}% | "
                  f"Time: {train_time:.1f}s")

        # Compute mean +/- SD for this dataset
        accs = [r['accuracy'] for r in seed_results]
        f1s = [r['f1'] for r in seed_results]
        precs = [r['precision'] for r in seed_results]
        recs = [r['recall'] for r in seed_results]

        summary = {
            'Dataset': ds_name,
            'Acc_mean': np.mean(accs),
            'Acc_std': np.std(accs, ddof=1),
            'F1_mean': np.mean(f1s),
            'F1_std': np.std(f1s, ddof=1),
            'Prec_mean': np.mean(precs),
            'Rec_mean': np.mean(recs),
            '95_CI_lower': np.mean(accs) - 1.96 * np.std(accs, ddof=1) / np.sqrt(len(accs)),
            '95_CI_upper': np.mean(accs) + 1.96 * np.std(accs, ddof=1) / np.sqrt(len(accs)),
        }
        summary_rows.append(summary)

        print(f"\n  * {ds_name} Summary: "
              f"{summary['Acc_mean']:.2f} +/- {summary['Acc_std']:.2f}% "
              f"(95% CI: [{summary['95_CI_lower']:.2f}, {summary['95_CI_upper']:.2f}])")

    # Save detailed per-seed results
    df_all = pd.DataFrame(all_results)
    out_all = RESULTS_DIR / "covnet22_3seed_all_datasets.csv"
    df_all.to_csv(out_all, index=False)
    print(f"\n[SAVED] Per-seed results: {out_all}")

    # Save summary
    df_summary = pd.DataFrame(summary_rows)
    out_summary = RESULTS_DIR / "covnet22_3seed_summary.csv"
    df_summary.to_csv(out_summary, index=False)
    print(f"[SAVED] Summary: {out_summary}")

    # Print final table
    print("\n" + "=" * 70)
    print("FINAL SUMMARY — CovNet22 Mean +/- SD (3 Seeds)")
    print("=" * 70)
    for s in summary_rows:
        print(f"  {s['Dataset']:12s}: {s['Acc_mean']:.2f} +/- {s['Acc_std']:.2f}%  "
              f"F1: {s['F1_mean']:.2f} +/- {s['F1_std']:.2f}%  "
              f"95% CI: [{s['95_CI_lower']:.2f}, {s['95_CI_upper']:.2f}]")
    print("=" * 70)
    print("\n[SUCCESS] All done! Copy these numbers into your LaTeX paper.")


if __name__ == "__main__":
    main()
