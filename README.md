# MRICovNetX: Multi-Format Dual-Branch Deep Neural Framework for Brain Tumor Classification with Explainable AI

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Official repository for the revised manuscript:  
**"MRICovNetX: A Multi-Format Dual-Branch Deep Neural Framework with Explainable AI for Brain Tumor Classification Using Heterogeneous MRI Data"** (Elsevier, 2026).

---

## 📌 Overview

Clinical MRI archives frequently store heterogeneous data formats — raw volumetric matrices (`.mat`) and standard compressed images (`.jpg`). Most existing DL diagnostic systems force all inputs through a single rigid pipeline, causing interpolation artifacts and loss of metadata.

**MRICovNetX** introduces a multi-format dual-branch framework that handles heterogeneous MRI data natively:

1. **CovBI-GRU Branch (`.mat` format):**  
   A hybrid Conv1D + Bidirectional GRU architecture for sequential axial slice representations. Evaluated under strict **patient-level group splitting** with zero inter-patient leakage.
2. **CovNet22 Branch (`.jpg` format):**  
   An ultra-lightweight 22-layer CNN (1.45M parameters) trained from scratch, with validation-based early stopping and learning-rate scheduling. All results report mean ± SD over 3 independent seeds.
3. **Comprehensive Explainable AI (XAI) Suite:**  
   Grad-CAM++, hemisphere-gated deep-core focusing, and LIME superpixel explanations with quantitative evaluation (EBPG, IoU).

---

## 📊 Benchmark Results

### Definitive Multi-Seed Results

All CovNet22 evaluations: PyTorch, 3 seeds (42/123/2024), validation-based early stopping (patience 7), max 60 epochs, checkpoint on best validation loss.

| Dataset | Format | Classes | Test Accuracy (%) | 95% CI | AUC | Macro F1 (%) |
|---|---|---|---|---|---|---|
| **D1 (Figshare)** | `.mat` | 3 | **82.73 ± 0.62** | [82.02, 83.43] | — | 82.35 ± 0.62 |
| **D2 (Kaggle, de-dup)** | `.jpg` | 4 | **90.99 ± 1.06** | [89.79, 92.19] | 0.985 | 90.31 ± 1.10 |
| **D3 (Nickparvar)** | `.jpg` | 4 | **99.08 ± 0.28** | [98.77, 99.40] | 0.9998 | 99.04 ± 0.28 |
| **D4 (Mendeley)** | `.jpg` | 3 | **98.61 ± 0.44** | [98.10, 99.11] | 0.9996 | 98.60 ± 0.44 |

> **Dataset-2 Duplicate Audit:** The canonical Kaggle split has 257/394 (65.2%) test images that are exact pHash duplicates of training images, inflating reported accuracy to 77.58%. Our de-duplicated 80/20 stratified split removes this leakage, yielding the reported 90.99%.

> **Patient-Level Splitting (D1):** Under random slice-level splitting, D1 achieves 98.03%. Patient-aware splitting reveals a clinical generalization accuracy of 82.73 ± 0.62%, confirming the critical importance of preventing slice leakage.

---

### Modern Baseline Comparison (Dataset-3)

Unified latency benchmark: batch size 1, FP32, NVIDIA RTX 3060, 50 warmup + 5×200 timed passes.

| Model | Params (M) | Test Acc (%) | Inference Latency (ms) |
|---|---|---|---|
| ResNet-50 | 23.52 | 98.63 | 10.93 |
| EfficientNet-B0 | 4.01 | 99.39 | 10.35 |
| MobileNetV2 | 2.23 | 99.31 | 6.78 |
| DenseNet-121 | 6.96 | 99.01 | 20.00 |
| VGG-16 | 134.28 | 96.64 | 7.15 |
| Swin-Tiny | 27.52 | 98.63 ± 0.73 | 14.46 |
| ViT-B/16 | 85.80 | 96.26 ± 2.63 | 10.13 |
| **CovNet22 (Ours)** | **1.45** | **99.08 ± 0.28** | **1.14** |

*CovNet22: 16× smaller than ResNet-50, 59× smaller than ViT-B/16, with ~1.1 ms inference and 2,908 img/s throughput.*

---

### CovBI-GRU Ablation (Dataset-1, Patient-Level Split)

| Configuration | Accuracy (%) | 95% CI | F1 (%) |
|---|---|---|---|
| **Full CovBI-GRU** | **82.73 ± 0.62** | [82.02, 83.43] | **82.35 ± 0.62** |
| w/o Bi-GRU (Conv1D Only) | 82.73 ± 0.42 | [82.25, 83.21] | 82.76 ± 0.65 |
| w/o Conv1D (Bi-GRU Only) | 84.28 ± 0.79 | [83.39, 85.17] | 84.19 ± 0.66 |
| w/o Batch Normalization | 81.35 ± 0.65 | [80.62, 82.08] | 80.97 ± 0.73 |
| w/o Dropout | 82.44 ± 0.41 | [81.98, 82.90] | 82.27 ± 0.36 |

---

### CovNet22 Ablation (Dataset-3, 3-Seed)

| Variant | Params | Accuracy (%) | 95% CI |
|---|---|---|---|
| **Full CovNet22** | 1.45M | **99.08 ± 0.28** | [98.77, 99.40] |
| No Augmentation | 1.45M | 97.46 ± 0.91 | — |
| No BatchNorm | 1.45M | 97.64 ± 0.28 | — |
| Shallow (3 blocks) | 0.42M | 98.12 ± 0.55 | — |
| No Dropout | 1.45M | 98.63 ± 0.68 | — |

---

## 📁 Repository Structure

```
MRICovNetX-Framework/
│
├── notebooks/                          # Original Jupyter Notebook implementations
│   ├── Brain_Tumor(Dataset_4 with_xai).ipynb
│   ├── Copy_of_CovBi_GRU_(Dataset_1)_.ipynb
│   ├── Copy_of_CovNet_(Dataset_2).ipynb
│   └── CovNet_(Datase_3)(with_xai).ipynb
│
├── scripts/                            # Reproducible PyTorch evaluation scripts
│   ├── 01_run_dedup_check.py           # pHash cross-dataset deduplication
│   ├── 02_run_failure_analysis.py      # Failure case analysis
│   ├── 03_run_patient_split_and_ablation.py  # CovBI-GRU patient-level split
│   ├── 04_run_baseline_comparison.py   # CNN baselines (ResNet, EfficientNet, etc.)
│   ├── 05_covnet22_3seed_d2d3d4.py     # Early 3-seed benchmark
│   ├── 05_run_d2_pooled.py             # Dataset-2 pooled replication
│   ├── 06_transformer_baselines.py     # Swin-Tiny & ViT-B/16 baselines
│   ├── 07_definitive_covnet22_eval.py  # ⭐ Definitive evaluation (3-seed, ES/LR)
│   ├── 08_latency_benchmark.py         # Unified latency benchmark (all models)
│   ├── 09_make_figures_v2.py           # Publication-quality figures
│   ├── 10_xai_v2.py                    # Grad-CAM++, LIME visualizations
│   ├── fix1_ablation_3seeds.py         # CovBI-GRU 3-seed ablation
│   ├── fix2_xai_2d_gradcam.py          # XAI visualization
│   └── fix2_xai_resnet50_gradcam.py    # Quantitative XAI (EBPG/IoU)
│
├── results/                            # Raw experiment CSVs
│   └── v2/                             # Definitive evaluation results
│       ├── covnet22_definitive_per_seed.csv
│       ├── covnet22_definitive_summary.csv
│       ├── covnet22_ablation_d3_per_seed.csv
│       ├── covnet22_ablation_d3_summary.csv
│       ├── covbigru_ablation_3seeds_sampleSD.csv
│       └── latency_benchmark.csv
│
├── paper/                              # LaTeX manuscript
│   ├── mricovnetx_elsevier_balanced_sc.tex
│   └── sn-bibliography.bib
│
├── docs/
│   └── response_to_reviewers.md
│
├── requirements.txt
├── LICENSE
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone & Setup
```bash
git clone https://github.com/saqlainovi/MRICovNetX.git
cd MRICovNetX
python -m venv venv
# Windows: venv\Scripts\activate
# Linux/macOS: source venv/bin/activate
pip install -r requirements.txt
```

### 2. Reproduce Definitive Results
```bash
# Run the definitive CovNet22 evaluation (3 seeds, D2/D3/D4)
python scripts/07_definitive_covnet22_eval.py

# Run CovBI-GRU patient-level ablation (3 seeds, D1)
python scripts/fix1_ablation_3seeds.py

# Run unified latency benchmark
python scripts/08_latency_benchmark.py
```

---

## 📜 Citation

```bibtex
@article{mricovnetx2026,
  title={MRICovNetX: A Multi-Format Dual-Branch Deep Neural Framework with Explainable AI for Brain Tumor Classification Using Heterogeneous MRI Data},
  author={Saqlain Ovi, Md Siyam and others},
  journal={Computerized Medical Imaging and Graphics},
  year={2026}
}
```

---

## 📄 License
This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
