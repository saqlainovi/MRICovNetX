"""
=============================================================================
MRICovNetX — Experiment 1: Dataset Deduplication & Overlap Check
=============================================================================
Author: Research Team
Runs perceptual hashing (pHash) across all datasets to check for duplicate
or leaking images between Dataset 1, Dataset 2, Dataset 3, and Dataset 4.
=============================================================================
"""

import os
import h5py
import numpy as np
import pandas as pd
from PIL import Image
import imagehash
from pathlib import Path
from collections import defaultdict

BASE_DIR = Path(__file__).resolve().parent.parent
DATASETS_DIR = BASE_DIR / "datasets"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

D1_DIR = DATASETS_DIR / "Dataset_1_Figshare" / "BRAIN_DATA"
D2_DIR = DATASETS_DIR / "Dataset_2_Kaggle"
D3_DIR = DATASETS_DIR / "Dataset_3_Nickparvar"
D4_DIR = DATASETS_DIR / "Dataset_4_Mendeley"

def get_hash_from_mat(filepath):
    try:
        with h5py.File(filepath, "r") as f:
            img = f["cjdata"]["image"][()]
            img_norm = ((img - np.min(img)) / (np.max(img) - np.min(img) + 1e-8) * 255.0).astype(np.uint8)
            pil_img = Image.fromarray(img_norm)
            return str(imagehash.phash(pil_img))
    except Exception:
        return None

def get_hash_from_img(filepath):
    try:
        pil_img = Image.open(filepath).convert("L")
        return str(imagehash.phash(pil_img))
    except Exception:
        return None

def main():
    print("=" * 60)
    print("EXPERIMENT 1: Dataset Deduplication & Contamination Check")
    print("=" * 60)
    
    dataset_hashes = {}
    
    # 1. Dataset 1 (.mat)
    print("\n[1/4] Hashing Dataset-1 (Figshare .mat files)...")
    d1_hashes = {}
    if D1_DIR.exists():
        for fname in os.listdir(D1_DIR):
            if fname.endswith(".mat"):
                fp = D1_DIR / fname
                h = get_hash_from_mat(fp)
                if h:
                    d1_hashes[str(fp)] = h
        print(f"  --> Hashed {len(d1_hashes)} images from Dataset 1")
    else:
        print(f"  ⚠️ Warning: {D1_DIR} not found")
    dataset_hashes["Dataset-1 (Figshare)"] = d1_hashes

    # 2. Dataset 2 (.jpg)
    print("\n[2/4] Hashing Dataset-2 (Kaggle Standard)...")
    d2_hashes = {}
    if D2_DIR.exists():
        for root, _, files in os.walk(D2_DIR):
            for fname in files:
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    fp = Path(root) / fname
                    h = get_hash_from_img(fp)
                    if h:
                        d2_hashes[str(fp)] = h
        print(f"  --> Hashed {len(d2_hashes)} images from Dataset 2")
    else:
        print(f"  ⚠️ Warning: {D2_DIR} not found")
    dataset_hashes["Dataset-2 (Kaggle)"] = d2_hashes

    # 3. Dataset 3 (.jpg)
    print("\n[3/4] Hashing Dataset-3 (Nickparvar Benchmark)...")
    d3_hashes = {}
    if D3_DIR.exists():
        for root, _, files in os.walk(D3_DIR):
            for fname in files:
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    fp = Path(root) / fname
                    h = get_hash_from_img(fp)
                    if h:
                        d3_hashes[str(fp)] = h
        print(f"  --> Hashed {len(d3_hashes)} images from Dataset 3")
    else:
        print(f"  ⚠️ Warning: {D3_DIR} not found")
    dataset_hashes["Dataset-3 (Nickparvar)"] = d3_hashes

    # 4. Dataset 4 (.jpg)
    print("\n[4/4] Hashing Dataset-4 (Mendeley Data)...")
    d4_hashes = {}
    if D4_DIR.exists():
        for root, _, files in os.walk(D4_DIR):
            for fname in files:
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    fp = Path(root) / fname
                    h = get_hash_from_img(fp)
                    if h:
                        d4_hashes[str(fp)] = h
        print(f"  --> Hashed {len(d4_hashes)} images from Dataset 4")
    else:
        print(f"  ⚠️ Warning: {D4_DIR} not found")
    dataset_hashes["Dataset-4 (Mendeley)"] = d4_hashes

    # Pairwise Comparison
    print("\n" + "=" * 60)
    print("CROSS-DATASET OVERLAP REPORT:")
    print("=" * 60)
    
    names = list(dataset_hashes.keys())
    results = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            d_a, d_b = names[i], names[j]
            hashes_a = set(dataset_hashes[d_a].values())
            hashes_b = set(dataset_hashes[d_b].values())
            
            overlap = hashes_a & hashes_b
            overlap_pct = (len(overlap) / max(min(len(hashes_a), len(hashes_b)), 1)) * 100.0
            
            print(f"  {d_a}  vs  {d_b}:")
            print(f"    - Count A: {len(hashes_a)}, Count B: {len(hashes_b)}")
            print(f"    - Exact Hash Matches: {len(overlap)} ({overlap_pct:.2f}%)")
            
            results.append({
                "Dataset A": d_a,
                "Dataset B": d_b,
                "Unique Images A": len(hashes_a),
                "Unique Images B": len(hashes_b),
                "Exact Hash Overlap": len(overlap),
                "Overlap Percentage (%)": f"{overlap_pct:.2f}%"
            })
            
    df = pd.DataFrame(results)
    out_csv = RESULTS_DIR / "dataset_dedup_report.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n[SUCCESS] Deduplication report saved to: {out_csv}")
    print(df.to_string(index=False))

if __name__ == "__main__":
    main()
