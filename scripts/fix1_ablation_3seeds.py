"""
=============================================================================
FIX 1: CovBI-GRU Ablation — 3 Seeds for Mean ± SD + 95% CI
=============================================================================
"""
import os, time, numpy as np, pandas as pd
from pathlib import Path
from collections import defaultdict
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import h5py, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "datasets" / "Dataset_1_Figshare" / "BRAIN_DATA"
RESULTS_DIR = BASE_DIR / "results"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_figshare():
    images, labels, pids = [], [], []
    for f in sorted(os.listdir(DATA_DIR)):
        if not f.endswith(".mat"): continue
        try:
            with h5py.File(DATA_DIR / f, "r") as mat:
                img = mat["cjdata"]["image"][()]
                lbl = int(mat["cjdata"]["label"][()].item()) - 1
                pid = "".join([chr(c[0]) for c in mat["cjdata"]["PID"][()]])
                if img.shape == (512, 512):
                    img_n = (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-8)
                    images.append(img_n.astype(np.float32))
                    labels.append(lbl); pids.append(pid)
        except: pass
    return np.array(images), np.array(labels), np.array(pids)

class CovBIGRU(nn.Module):
    def __init__(self, variant="full"):
        super().__init__()
        self.variant = variant
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(0.2)
        
        if variant in ("full", "no_batchnorm", "no_dropout"):
            self.conv1 = nn.Conv1d(512, 64, 4); self.pool1 = nn.MaxPool1d(2)
            self.gru1 = nn.GRU(64, 64, bidirectional=True, batch_first=True)
            self.conv2 = nn.Conv1d(128, 64, 3); self.conv3 = nn.Conv1d(64, 64, 3)
            self.pool2 = nn.MaxPool1d(2)
            self.gru2 = nn.GRU(64, 128, bidirectional=True, batch_first=True)
            self.conv4 = nn.Conv1d(256, 128, 2); self.pool3 = nn.MaxPool1d(2)
            if variant != "no_batchnorm":
                self.bn1 = nn.BatchNorm1d(64); self.bn2 = nn.BatchNorm1d(64)
                self.bn3 = nn.BatchNorm1d(128)
        elif variant == "conv1d_only":
            self.conv1 = nn.Conv1d(512, 64, 4); self.pool1 = nn.MaxPool1d(2)
            self.bn1 = nn.BatchNorm1d(64)
            self.conv2 = nn.Conv1d(64, 64, 3); self.conv3 = nn.Conv1d(64, 64, 3)
            self.pool2 = nn.MaxPool1d(2); self.bn2 = nn.BatchNorm1d(64)
            self.conv4 = nn.Conv1d(64, 128, 2); self.pool3 = nn.MaxPool1d(2)
            self.bn3 = nn.BatchNorm1d(128)
        elif variant == "bigru_only":
            self.gru1 = nn.GRU(512, 64, bidirectional=True, batch_first=True)
            self.bn1 = nn.BatchNorm1d(128)
            self.gru2 = nn.GRU(128, 128, bidirectional=True, batch_first=True)

        with torch.no_grad():
            flat_dim = self._features(torch.zeros(1, 512, 512)).shape[1]
        self.fc1 = nn.Linear(flat_dim, 128)
        self.bn_fc = nn.BatchNorm1d(128) if variant != "no_batchnorm" else nn.Identity()
        self.fc2 = nn.Linear(128, 3)

    def _features(self, x):
        v = self.variant
        if v in ("full", "no_batchnorm", "no_dropout"):
            x = x.transpose(1,2)
            x = self.pool1(self.relu(self.conv1(x)))
            if v != "no_batchnorm": x = self.bn1(x)
            x = x.transpose(1,2); x, _ = self.gru1(x); x = x.transpose(1,2)
            x = self.relu(self.conv2(x)); x = self.pool2(self.relu(self.conv3(x)))
            if v != "no_batchnorm": x = self.bn2(x)
            x = x.transpose(1,2); x, _ = self.gru2(x); x = x.transpose(1,2)
            x = self.pool3(self.relu(self.conv4(x)))
            if v != "no_batchnorm": x = self.bn3(x)
        elif v == "conv1d_only":
            x = x.transpose(1,2)
            x = self.bn1(self.pool1(self.relu(self.conv1(x))))
            x = self.relu(self.conv2(x)); x = self.bn2(self.pool2(self.relu(self.conv3(x))))
            x = self.bn3(self.pool3(self.relu(self.conv4(x))))
        elif v == "bigru_only":
            x, _ = self.gru1(x)
            x = self.bn1(x.transpose(1,2)).transpose(1,2)
            x, _ = self.gru2(x)
        return x.flatten(1)

    def forward(self, x):
        f = self._features(x)
        x = self.bn_fc(self.relu(self.fc1(f)))
        if self.variant != "no_dropout": x = self.drop(x)
        return self.fc2(x)

def train_eval(variant, train_loader, test_loader, seed, epochs=60, lr=0.001):
    torch.manual_seed(seed); np.random.seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    
    model = CovBIGRU(variant).to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    
    # Use LR scheduler for better convergence
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    
    for ep in range(epochs):
        model.train()
        for bx, by in train_loader:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            opt.zero_grad(); loss = crit(model(bx), by); loss.backward(); opt.step()
        scheduler.step()
    
    model.eval(); preds, tgts = [], []
    with torch.no_grad():
        for bx, by in test_loader:
            preds.extend(torch.argmax(model(bx.to(DEVICE)), 1).cpu().numpy())
            tgts.extend(by.numpy())
    acc = accuracy_score(tgts, preds)
    p, r, f1, _ = precision_recall_fscore_support(tgts, preds, average="macro", zero_division=0)
    return acc, p, r, f1

def main():
    print("=" * 70)
    print(f"CovBI-GRU ABLATION — 3 Seeds × 5 Variants on {DEVICE}")
    print("=" * 70)

    images, labels, pids = load_figshare()
    print(f"Loaded {len(images)} slices, {len(set(pids))} patients")

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    tr_idx, te_idx = next(gss.split(images, labels, groups=pids))
    
    tr_ds = TensorDataset(torch.tensor(images[tr_idx]), torch.tensor(labels[tr_idx]))
    te_ds = TensorDataset(torch.tensor(images[te_idx]), torch.tensor(labels[te_idx]))
    tr_loader = DataLoader(tr_ds, batch_size=32, shuffle=True)
    te_loader = DataLoader(te_ds, batch_size=32, shuffle=False)

    SEEDS = [42, 123, 2024]
    variants = [
        ("Full CovBI-GRU (Proposed)", "full"),
        ("w/o Bi-GRU (Conv1D Only)", "conv1d_only"),
        ("w/o Conv1D (Bi-GRU Only)", "bigru_only"),
        ("w/o Batch Normalization", "no_batchnorm"),
        ("w/o Dropout Regularization", "no_dropout"),
    ]

    rows = []
    for name, var in variants:
        accs, precs, recs, f1s = [], [], [], []
        print(f"\n--- {name} ---")
        for s in SEEDS:
            t0 = time.time()
            a, p, r, f = train_eval(var, tr_loader, te_loader, seed=s, epochs=80, lr=0.001)
            dt = time.time() - t0
            accs.append(a*100); precs.append(p*100); recs.append(r*100); f1s.append(f*100)
            print(f"  Seed {s}: Acc={a*100:.2f}%, F1={f*100:.2f}% ({dt:.0f}s)")

        acc_m, acc_s = np.mean(accs), np.std(accs)
        f1_m, f1_s = np.mean(f1s), np.std(f1s)
        # 95% CI = mean ± 1.96 * SD/sqrt(n)
        ci_lo = acc_m - 1.96 * acc_s / np.sqrt(len(SEEDS))
        ci_hi = acc_m + 1.96 * acc_s / np.sqrt(len(SEEDS))

        rows.append({
            "Model Configuration": name,
            "Accuracy (mean +/- SD)": f"{acc_m:.2f} +/- {acc_s:.2f}",
            "95% CI": f"[{ci_lo:.2f}, {ci_hi:.2f}]",
            "Precision (mean)": f"{np.mean(precs):.2f}",
            "Recall (mean)": f"{np.mean(recs):.2f}",
            "F1-Score (mean +/- SD)": f"{f1_m:.2f} +/- {f1_s:.2f}",
        })
        print(f"  => Acc: {acc_m:.2f} +/- {acc_s:.2f}, 95% CI [{ci_lo:.2f}, {ci_hi:.2f}]")

    df = pd.DataFrame(rows)
    out = RESULTS_DIR / "covbigru_ablation_3seeds.csv"
    df.to_csv(out, index=False)
    print(f"\n[DONE] Saved to {out}")
    print(df.to_string(index=False))

if __name__ == "__main__":
    main()
