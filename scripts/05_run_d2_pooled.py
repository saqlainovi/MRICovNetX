"""
=============================================================================
MRICovNetX — Run CovNet22 3-Seed on Dataset-2 (Pooled 3,264 images, 80/20 split)
=============================================================================
In the original paper and notebook (Copy_of_CovNet_(Dataset_2).ipynb), Dataset-2
was evaluated using all 3,264 images (2,870 Training + 394 Testing pooled) with
an 80/20 train/test split.

This script runs 3 seeds on this canonical setup, then updates:
- covnet22_3seed_all_datasets.csv
- covnet22_3seed_summary.csv
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

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEEDS = [42, 123, 2024]
EPOCHS = 12
BATCH_SIZE = 32
LR = 0.001


class CovNet22(nn.Module):
    def __init__(self, num_classes=4):
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


class ImageFolderDataset(Dataset):
    def __init__(self, folders, class_map):
        self.samples = []
        for folder in folders:
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
        return img, label


class TransformedSubset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        img, label = self.subset.dataset[self.subset.indices[idx]]
        if self.transform:
            img = self.transform(img)
        return img, label


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


def train_and_eval(model, train_loader, test_loader, epochs=EPOCHS):
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
        correct, total = 0, 0
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

    if best_state is not None:
        model.load_state_dict(best_state)

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
    print("EXPERIMENT 5b: CovNet22 Dataset-2 (Pooled 3,264 Images, 80/20 Split)")
    print(f"Device: {DEVICE} | Seeds: {SEEDS} | Epochs: {EPOCHS} | LR: {LR}")
    print("=" * 70)

    classes = {'glioma_tumor': 0, 'meningioma_tumor': 1, 'no_tumor': 2, 'pituitary_tumor': 3}
    d2_folders = [
        BASE_DIR / "datasets" / "Dataset_2_Kaggle" / "Training",
        BASE_DIR / "datasets" / "Dataset_2_Kaggle" / "Testing"
    ]

    raw_ds = ImageFolderDataset(d2_folders, classes)
    print(f"Dataset-2 pooled total images: {len(raw_ds)}")

    t_tr, t_te = get_transforms()
    d2_results = []

    for seed in SEEDS:
        print(f"\n--- Seed {seed} ---")
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)

        n_tot = len(raw_ds)
        n_tr = int(0.8 * n_tot)
        n_te = n_tot - n_tr
        sub_tr, sub_te = random_split(
            raw_ds, [n_tr, n_te],
            generator=torch.Generator().manual_seed(seed))

        print(f"  Train: {len(sub_tr)} | Test: {len(sub_te)}")
        tr_loader = DataLoader(TransformedSubset(sub_tr, t_tr), batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
        te_loader = DataLoader(TransformedSubset(sub_te, t_te), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

        model = CovNet22(num_classes=4).to(DEVICE)
        t0 = time.time()
        res = train_and_eval(model, tr_loader, te_loader)
        res['seed'] = seed
        res['dataset'] = "Dataset-2"
        res['train_time_s'] = time.time() - t0
        d2_results.append(res)
        print(f"  [Result] Acc: {res['accuracy']:.2f}% | F1: {res['f1']:.2f}% | Time: {res['train_time_s']:.1f}s")

    # Update covnet22_3seed_all_datasets.csv
    csv_all_path = RESULTS_DIR / "covnet22_3seed_all_datasets.csv"
    if csv_all_path.exists():
        existing_df = pd.read_csv(csv_all_path)
        existing_other = existing_df[existing_df['dataset'] != 'Dataset-2']
        updated_all = pd.concat([pd.DataFrame(d2_results), existing_other], ignore_index=True)
    else:
        updated_all = pd.DataFrame(d2_results)

    updated_all.to_csv(csv_all_path, index=False)
    print(f"\n[UPDATED] {csv_all_path}")

    # Recompute summary
    summary_rows = []
    for ds_name in ['Dataset-2', 'Dataset-3', 'Dataset-4']:
        sub = updated_all[updated_all['dataset'] == ds_name]
        if len(sub) > 0:
            accs = sub['accuracy'].values
            f1s = sub['f1'].values
            precs = sub['precision'].values
            recs = sub['recall'].values
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

    df_summary = pd.DataFrame(summary_rows)
    out_summary = RESULTS_DIR / "covnet22_3seed_summary.csv"
    df_summary.to_csv(out_summary, index=False)
    print(f"[UPDATED] {out_summary}")

    print("\n" + "=" * 70)
    print("FINAL SUMMARY — ALL DATASETS (3 Seeds Mean +/- SD)")
    print("=" * 70)
    for s in summary_rows:
        print(f"  {s['Dataset']:12s}: {s['Acc_mean']:.2f} +/- {s['Acc_std']:.2f}%  "
              f"F1: {s['F1_mean']:.2f} +/- {s['F1_std']:.2f}%  "
              f"95% CI: [{s['95_CI_lower']:.2f}, {s['95_CI_upper']:.2f}]")
    print("=" * 70)


if __name__ == "__main__":
    main()
