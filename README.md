# MRICovNetX: Multi-Format Dual-Branch Deep Neural Framework for Brain Tumor Classification with Explainable AI

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Official repository for the revised manuscript:  
**"MRICovNetX: A Multi-Format Dual-Branch Deep Neural Framework with Explainable AI for Brain Tumor Classification Using Heterogeneous MRI Data"** (Elsevier, 2026).

---

## 📌 Overview

Clinical magnetic resonance imaging (MRI) archives frequently store heterogeneous data formats ranging from raw volumetric numerical matrices (`.mat`) to standard compressed photographic images (`.jpg`). Existing deep learning diagnostic systems typically force heterogeneous inputs into a single rigid pipeline, resulting in interpolation artifacts, channel distortion, and loss of volumetric slice metadata.

**MRICovNetX** introduces a multi-format dual-branch framework designed to address heterogeneous MRI data natively:

1. **CovBI-GRU Branch (`.mat` format):**  
   A hybrid architecture combining 1D convolutional layers with bidirectional gated recurrent units (Bi-GRU) tailored for sequential axial slice representations. Evaluated under strict **patient-level group splitting** (`cjdata/PID`) with zero inter-patient data leakage.
2. **CovNet22 Branch (`.jpg` format):**  
   An ultra-lightweight 22-layer convolutional network (1.45M parameters) optimized for edge deployment, trained from scratch on brain MRI without relying on ImageNet natural-image pretraining.
3. **Comprehensive Explainable AI (XAI) Suite:**  
   Combines Grad-CAM++, skull-stripping, hemisphere-gated deep-core focusing, and LIME superpixel explanations with quantitative ground-truth mask evaluation (Energy-Based Pointing Game and IoU).

```
                            ┌────────────────────────────────────────┐
                            │        Heterogeneous MRI Inputs        │
                            └───────────────────┬────────────────────┘
                                                │
                       ┌────────────────────────┴────────────────────────┐
                       ▼                                                 ▼
        ┌──────────────────────────────┐                 ┌──────────────────────────────┐
        │   Volumetric .mat Arrays     │                 │      Standard .jpg Slices     │
        │   (Dataset-1: Figshare)      │                 │  (Dataset-2, 3, 4 Benchmarks)│
        └──────────────┬───────────────┘                 └──────────────┬───────────────┘
                       │                                                 │
                       ▼                                                 ▼
        ┌──────────────────────────────┐                 ┌──────────────────────────────┐
        │       CovBI-GRU Branch       │                 │       CovNet22 Branch        │
        │ • 1D-CNN Feature Extractor   │                 │ • 5x Conv2D Feature Blocks   │
        │ • Bidirectional GRU Slices   │                 │ • 1.45M Trainable Parameters │
        │ • Patient-Level Group Split  │                 │ • Trained From Scratch       │
        └──────────────┬───────────────┘                 └──────────────┬───────────────┘
                       │                                                 │
                       ▼                                                 ▼
        ┌──────────────────────────────┐                 ┌──────────────────────────────┐
        │ Acc: 82.73 ± 0.51% (Patient) │                 │ Acc: 93.87% - 98.83% (3-Seed)│
        └──────────────┬───────────────┘                 └──────────────┬───────────────┘
                       │                                                 │
                       └────────────────────────┬────────────────────────┘
                                                ▼
                            ┌────────────────────────────────────────┐
                            │    Explainable AI (XAI) Verification   │
                            │  Grad-CAM++ | EBPG / IoU | LIME Maps   │
                            └────────────────────────────────────────┘
```

---

## 📊 Benchmark Results

### 1. Multi-Seed Replication Across All Four Datasets

All evaluations report Mean $\pm$ Standard Deviation and 95% Confidence Intervals across three independent random seeds (`seed=42, 123, 2024`):

| Dataset | Format | Classes | Samples | Splitting Protocol | Test Accuracy (%) | 95% CI | Macro F1 (%) |
|---|---|---|---|---|---|---|---|
| **Dataset-1 (Figshare)** | `.mat` | 3 (Meningioma, Glioma, Pituitary) | 3,064 | **Patient-Level (GroupShuffleSplit)** | **82.73 $\pm$ 0.51** | [82.15, 83.30] | 82.35 $\pm$ 0.62 |
| **Dataset-2 (Kaggle)** | `.jpg` | 4 (+ No Tumor) | 3,264 | 80/20 Stratified Split | **93.87 $\pm$ 0.93** | [92.82, 94.93] | 93.94 $\pm$ 0.80 |
| **Dataset-3 (Nickparvar)**| `.jpg` | 4 (+ No Tumor) | 7,023 | Canonical Split (5,712 / 1,311) | **98.83 $\pm$ 0.04** | [98.78, 98.88] | 98.75 $\pm$ 0.05 |
| **Dataset-4 (Mendeley)** | `.jpg` | 3 (Glioma, Menin, Pituitary) | 6,056 | 85/15 Stratified Split | **98.02 $\pm$ 0.43** | [97.53, 98.50] | 98.01 $\pm$ 0.42 |

> **Note on Patient-Level Splitting:** Under random slice-level splitting, Dataset-1 achieves 98.03% accuracy. However, patient-aware splitting reveals a clinical generalization accuracy of 82.73 $\pm$ 0.51%, confirming the critical importance of preventing slice leakage across identical patients.

---

### 2. Comparison Against Modern CNNs and Vision Transformers (Dataset-3)

Evaluated under identical data partitions, input dimensions ($224 \times 224$), and evaluation metrics:

| Model Architecture | Paradigm | Params (M) | Test Accuracy (%) | Macro F1 (%) | Inference Latency (ms) | Training Time (s) |
|---|---|---|---|---|---|---|
| ResNet-50 | CNN (Pretrained) | 23.52 | 98.63 | 98.63 | 0.3 | 395.7 |
| EfficientNet-B0 | CNN (Pretrained) | 4.01 | 99.39 | 99.34 | 0.4 | 268.4 |
| MobileNetV2 | CNN (Pretrained) | 2.23 | 99.31 | 99.27 | 0.2 | 220.3 |
| DenseNet-121 | CNN (Pretrained) | 6.96 | 99.01 | 98.93 | 0.7 | 458.2 |
| VGG-16 | CNN (Pretrained) | 134.28 | 96.64 | 96.52 | 1.4 | 8,031.4 |
| Swin-Tiny | Vision Transformer | 27.52 | 98.63 $\pm$ 0.73 | 98.53 $\pm$ 0.79 | 0.9 | 644.3 |
| ViT-B/16 | Vision Transformer | 85.80 | 96.26 $\pm$ 2.63 | 96.01 $\pm$ 2.84 | 0.4 | 1,255.1 |
| **CovNet22 (Ours)** | **CNN (From Scratch)** | **1.45** | **98.83 $\pm$ 0.04** | **98.75 $\pm$ 0.05** | **14.2** | **380.0** |

*CovNet22 achieves competitive performance while requiring $16.2\times$ fewer parameters than ResNet-50, $19.0\times$ fewer than Swin-Tiny, and $59.2\times$ fewer than ViT-B/16, without any external pretraining weights.*

---

### 3. CovBI-GRU Component Ablation Study (Dataset-1, Patient-Level Split)

| Configuration | Test Accuracy (%) | 95% CI | Precision (%) | Recall (%) | F1-Score (%) |
|---|---|---|---|---|---|
| **Full CovBI-GRU (Proposed)** | **82.73 $\pm$ 0.51** | **[82.15, 83.30]** | 81.26 | 84.77 | **82.35 $\pm$ 0.62** |
| w/o Bi-GRU (Conv1D Only) | 82.73 $\pm$ 0.42 | [82.25, 83.21] | 81.73 | 85.23 | 82.76 $\pm$ 0.65 |
| w/o Conv1D (Bi-GRU Only) | 84.28 $\pm$ 0.79 | [83.39, 85.17] | 82.84 | 86.82 | 84.19 $\pm$ 0.66 |
| w/o Batch Normalization | 81.35 $\pm$ 0.65 | [80.62, 82.08] | 80.17 | 83.34 | 80.97 $\pm$ 0.73 |
| w/o Dropout Regularization | 82.44 $\pm$ 0.41 | [81.98, 82.90] | 81.18 | 84.56 | 82.27 $\pm$ 0.36 |

---

## 📁 Repository Structure

```
MRICovNetX-Framework/
│
├── notebooks/                       # Interactive Jupyter Notebook implementations
│   ├── Brain_Tumor(Dataset_4 with_xai).ipynb
│   ├── Copy_of_CovBi_GRU_(Dataset_1)_.ipynb
│   ├── Copy_of_CovNet_(Dataset_2).ipynb
│   └── CovNet_(Datase_3)(with_xai).ipynb
│
├── scripts/                         # Standalone modular experimental pipelines
│   ├── 01_run_dedup_check.py        # Perceptual hashing cross-dataset deduplication
│   ├── 02_run_failure_analysis.py   # Radiologically validated failure case analysis
│   ├── 03_run_patient_split_and_ablation.py # CovBI-GRU patient-level split & ablation
│   ├── 04_run_baseline_comparison.py# ResNet-50, MobileNet, EfficientNet, DenseNet, VGG
│   ├── 05_covnet22_3seed_d2d3d4.py  # 3-seed multi-run benchmark for Datasets 2, 3, 4
│   ├── 05_run_d2_pooled.py          # Dataset-2 pooled 80/20 replication
│   ├── 06_transformer_baselines.py  # Swin-Tiny & ViT-B/16 Vision Transformer baselines
│   ├── fix1_ablation_3seeds.py      # Statistical CI ablation generator
│   ├── fix2_xai_2d_gradcam.py       # Grad-CAM++ & LIME visualization generator
│   └── fix2_xai_resnet50_gradcam.py # Quantitative XAI (EBPG / IoU / Pointing Game)
│
├── results/                         # Raw machine-readable benchmark CSVs
│   ├── covbigru_ablation_3seeds.csv
│   ├── covnet22_3seed_all_datasets.csv
│   ├── covnet22_3seed_summary.csv
│   ├── modern_baselines_comparison.csv
│   ├── transformer_baselines_3seed.csv
│   ├── dataset_dedup_report.csv
│   ├── patient_level_split_summary.csv
│   └── quantitative_xai_resnet50_energy.csv
│
├── paper/                           # LaTeX manuscript sources
│   ├── mricovnetx_elsevier_balanced_sc.tex
│   └── sn-bibliography.bib
│
├── docs/                            # Peer-review & Rebuttal documentation
│   ├── response_to_reviewers.md     # Comprehensive 4-reviewer revision response
│   ├── brutally_honest_paper_review.md
│   └── resubmission_readiness_report.md
│
├── requirements.txt                 # Python environment dependencies
├── LICENSE                          # MIT Open-Source License
└── README.md                        # Documentation & reproduction guide
```

---

## 🚀 Quick Start & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/saqlainovi/MRICovNetX-Framework.git
cd MRICovNetX-Framework
```

### 2. Set Up Virtual Environment
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🔬 Reproducing Experiments

### Run Cross-Dataset Deduplication (pHash)
```bash
python scripts/01_run_dedup_check.py
```

### Run CovBI-GRU Patient-Level Split & Ablation
```bash
python scripts/03_run_patient_split_and_ablation.py
```

### Run CovNet22 3-Seed Replication
```bash
python scripts/05_covnet22_3seed_d2d3d4.py
```

### Run Vision Transformer Baselines (Swin-Tiny & ViT-B/16)
```bash
python scripts/06_transformer_baselines.py
```

---

## 📜 Citation

If you use MRICovNetX or reference these benchmarks in your research, please cite:

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
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
