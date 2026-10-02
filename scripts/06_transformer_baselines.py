"""
=============================================================================
MRICovNetX — Experiment 6: ViT & Swin Transformer Baselines
=============================================================================
Reviewer R3.7 asked for lightweight ViT/DeiT comparison.
Runs ViT-B/16 (fine-tuned) and Swin-Tiny on Dataset-3 under identical
protocol as Experiment 4 (04_run_baseline_comparison.py).

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
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "datasets" / "Dataset_3_Nickparvar"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES = {'glioma': 0, 'meningioma': 1, 'notumor': 2, 'pituitary': 3}
NUM_CLASSES = 4
SEEDS = [42, 123, 2024]
EPOCHS = 8
BATCH_SIZE = 16  # Fits in 8GB VRAM
LR = 0.0001


class MRIDataset(Dataset):
    def __init__(self, folder, transform=None):
        self.samples = []
        self.transform = transform
        for cls_name, cls_idx in CLASSES.items():
            dir_p = folder / cls_name
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


def get_transformer_model(name, num_classes=NUM_CLASSES):
    """Create a transformer baseline model."""
    if name == "ViT-B/16":
        m = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
        m.heads.head = nn.Linear(m.heads.head.in_features, num_classes)
        total_params = sum(p.numel() for p in m.parameters())
    elif name == "Swin-Tiny":
        m = models.swin_t(weights=models.Swin_T_Weights.DEFAULT)
        m.head = nn.Linear(m.head.in_features, num_classes)
        total_params = sum(p.numel() for p in m.parameters())
    return m, total_params


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_eval(model, train_loader, test_loader, model_name, epochs=EPOCHS):
    """Train and evaluate a model, return metrics dict."""
    model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    start_train = time.time()
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

        if (ep + 1) % 2 == 0:
            print(f"    Epoch {ep+1}/{epochs} - loss: {running_loss/len(train_loader):.4f}")

    train_time = time.time() - start_train

    # Test & Latency
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

    avg_latency_ms = np.mean(latencies) * 1000.0
    acc = accuracy_score(all_targets, all_preds) * 100.0
    prec, rec, f1, _ = precision_recall_fscore_support(
        all_targets, all_preds, average='macro', zero_division=0)

    return {
        'accuracy': acc,
        'precision': prec * 100.0,
        'recall': rec * 100.0,
        'f1': f1 * 100.0,
        'latency_ms': avg_latency_ms,
        'train_time_s': train_time
    }


def main():
    print("=" * 70)
    print("EXPERIMENT 6: Transformer Baselines on Dataset-3")
    print(f"Device: {DEVICE} | Seeds: {SEEDS} | Epochs: {EPOCHS}")
    print("=" * 70)

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

    train_dir = DATA_DIR / "Training"
    test_dir = DATA_DIR / "Testing"

    train_ds = MRIDataset(train_dir, transform=transform_train)
    test_ds = MRIDataset(test_dir, transform=transform_test)

    print(f"Dataset-3: {len(train_ds)} train, {len(test_ds)} test\n")

    model_names = ["Swin-Tiny", "ViT-B/16"]
    all_results = []

    for model_name in model_names:
        print(f"\n{'='*50}")
        print(f"  {model_name}")
        print(f"{'='*50}")

        seed_results = []

        for seed in SEEDS:
            set_seed(seed)
            print(f"\n  Seed {seed}:")

            g = torch.Generator()
            g.manual_seed(seed)
            train_loader = DataLoader(
                train_ds, batch_size=BATCH_SIZE, shuffle=True,
                num_workers=0, pin_memory=True, generator=g)
            test_loader = DataLoader(
                test_ds, batch_size=BATCH_SIZE, shuffle=False,
                num_workers=0, pin_memory=True)

            model, total_params = get_transformer_model(model_name)
            metrics = train_eval(model, train_loader, test_loader, model_name)
            metrics['seed'] = seed
            metrics['model'] = model_name
            metrics['params_m'] = total_params / 1e6
            seed_results.append(metrics)
            all_results.append(metrics)

            print(f"    [Result] Acc: {metrics['accuracy']:.2f}% | "
                  f"F1: {metrics['f1']:.2f}% | "
                  f"Params: {metrics['params_m']:.1f}M | "
                  f"Time: {metrics['train_time_s']:.1f}s")

            del model
            torch.cuda.empty_cache()

        accs = [r['accuracy'] for r in seed_results]
        f1s = [r['f1'] for r in seed_results]
        print(f"\n  * {model_name}: {np.mean(accs):.2f} +/- {np.std(accs, ddof=1):.2f}% | "
              f"F1: {np.mean(f1s):.2f} +/- {np.std(f1s, ddof=1):.2f}%")

    # Save all results
    df = pd.DataFrame(all_results)
    out_csv = RESULTS_DIR / "transformer_baselines_3seed.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n[SAVED] {out_csv}")

    # Print summary table for LaTeX
    print("\n" + "=" * 70)
    print("SUMMARY FOR LATEX TABLE 8")
    print("=" * 70)
    for model_name in model_names:
        rows = [r for r in all_results if r['model'] == model_name]
        accs = [r['accuracy'] for r in rows]
        f1s = [r['f1'] for r in rows]
        precs = [r['precision'] for r in rows]
        lats = [r['latency_ms'] for r in rows]
        trains = [r['train_time_s'] for r in rows]
        params = rows[0]['params_m']
        print(f"  {model_name:12s} | {params:.1f}M | "
              f"Acc: {np.mean(accs):.2f}+/-{np.std(accs,ddof=1):.2f}% | "
              f"F1: {np.mean(f1s):.2f}+/-{np.std(f1s,ddof=1):.2f}% | "
              f"Prec: {np.mean(precs):.2f}% | "
              f"Lat: {np.mean(lats):.1f}ms | "
              f"Train: {np.mean(trains):.1f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
