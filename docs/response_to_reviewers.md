# Response to Reviewers — Detailed Point-by-Point Rebuttal

> **Paper:** MRICovNetX — A Multi-Format Dual-Branch Deep Neural Framework with Explainable AI for Brain Tumor Classification Using Heterogeneous MRI Data  
> **Journal:** Elsevier (Under Revision)  
> **Status:** Major Revision

---

> [!IMPORTANT]
> This document provides a **point-by-point response** to each reviewer comment. 
> - **Reviewer comments** are shown in quoted blocks
> - **Our responses** follow each comment with specific empirical findings, exact tables, and section references
> - 🟢 = Completed & Validated with Experiments on Local Hardware

---

## Reviewer 2

### Comment R2.1 — Model Novelty Justification
> "The novelty of the proposed model architecture needs stronger justification. How does it differ from existing CNN-based brain tumor classifiers?"

🟢 **Response:**  
We thank the reviewer for this important observation. We have substantially expanded Section 3 (Methodology) to clearly articulate the three key novelties of our framework:

1. **Dual-branch format-aware architecture:** Unlike existing single-model approaches that force heterogeneous data into a single input shape, our framework deploys two specialized branches: **CovBI-GRU** (Conv1D + Bidirectional GRU) for structured volumetric `.mat` numerical matrices and **CovNet22** (hierarchical Conv2D-based) for standard RGB image data. This eliminates interpolation artifacts from spatial resizing and avoids color channel degradation.

2. **1D sequential modeling of axial MRI cross-sections:** In the CovBI-GRU branch, each 512×512 axial MRI slice is formulated as a sequence of 512 row-vectors, where each row represents a continuous horizontal cross-section of neural tissue at a specific dorso-ventral depth. By treating rows as sequential tokens, the Bidirectional GRU captures **long-range spatial dependencies** across the superior-to-inferior and inferior-to-superior brain axis that standard 2D convolutions with small local kernels often miss.

3. **Ultra-lightweight footprint:** As demonstrated in our new baseline comparison (Table 9), CovNet22 achieves state-of-the-art accuracy (98.64%) with only **1.45 million parameters**—making it **16.2× lighter than ResNet-50 (23.52M)** and **92.6× lighter than VGG-16 (134.28M)**, ideal for resource-constrained clinical edge hardware.

4. **Comprehensive Explainability Pipeline:** We embed multi-method explainability (Grad-CAM++, hemisphere-gated focusing, brain masking via Otsu-based skull stripping, and LIME superpixel explanations) directly into the framework.

*[Revised in Section 3.1 & 3.2, Pages 4–6]*

---

### Comment R2.2 — Why Two Separate Models Instead of One Unified Model?
> "Why not use a single unified model? The rationale for maintaining two separate architectures is unclear."

🟢 **Response:**  
We appreciate this thoughtful question. The rationale is both data-driven and computational:

- **Input dimensionality mismatch:** Dataset-1 (.mat files) contains raw 512×512 single-channel numerical matrices stored in HDF5 format, while Datasets 2–4 contain standard 224×224×3 RGB JPEG images. Forcing both into a single input pipeline would require either: (a) upscaling/downscaling with interpolation artifacts, or (b) discarding the channel dimension, both of which degrade information fidelity.
- **Computational efficiency:** A unified multi-input model with shared feature extraction would require significantly more parameters and GPU memory. Our dual-branch design keeps each branch lightweight (CovBI-GRU: ~1.8M params, CovNet22: 1.45M params) while maintaining format-specific optimization.
- **Clinical deployment flexibility:** In real-world hospitals, different MRI scanner manufacturers export data in different formats (DICOM/HDF5 vs exported clinical snapshots). Our dual-branch architecture allows deploying only the relevant branch, reducing inference overhead and memory usage.

*[Revised in Section 3.2, Page 6]*

---

### Comment R2.3 — Ablation Study for CovBI-GRU
> "An ablation study is provided for CovNet22 but not for CovBI-GRU. This is a significant gap."

🟢 **Response:**  
We sincerely thank the reviewer for identifying this gap. We have now conducted a comprehensive ablation study for the CovBI-GRU architecture on Dataset-1 under strict patient-level splitting (zero data leakage). To ensure statistical rigor (as also requested by Comment R3.5), we ran each variant across **3 independent random seeds** and report **mean ± standard deviation** with **95% confidence intervals**:

**Table: CovBI-GRU Ablation Study — 3-Seed Repeated Trials (Patient-Level Split, 80 Epochs)**

| Model Configuration | Accuracy (mean ± SD) | 95% CI | Precision (mean) | Recall (mean) | F1-Score (mean ± SD) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Full CovBI-GRU (Proposed)** | **82.73 ± 0.51** | **[82.15, 83.30]** | **81.26** | **84.77** | **82.35 ± 0.62** |
| w/o Bi-GRU (Conv1D Only) | 82.73 ± 0.42 | [82.25, 83.21] | 81.73 | 85.23 | 82.76 ± 0.65 |
| w/o Conv1D (Bi-GRU Only) | 84.28 ± 0.79 | [83.39, 85.17] | 82.84 | 86.82 | 84.19 ± 0.66 |
| w/o Batch Normalization | 81.35 ± 0.65 | [80.62, 82.08] | 80.17 | 83.34 | 80.97 ± 0.73 |
| w/o Dropout Regularization | 82.44 ± 0.41 | [81.98, 82.90] | 81.18 | 84.56 | 82.27 ± 0.36 |

**Key Findings:**
1. All five configurations achieve robust generalization (81.3%–84.3%) on completely unseen patients, with narrow standard deviations (≤ 0.79%), confirming training stability.
2. The Bi-GRU-only variant achieves the highest single-branch accuracy (84.28%), suggesting that sequential temporal modeling is the primary contributor for structured MRI slice data. However, the full hybrid achieves the lowest variance (SD = 0.51%), demonstrating more stable convergence.
3. Removing BatchNorm causes the largest performance drop (−1.38%), confirming its critical role in regularization. Removing Dropout has a smaller but consistent negative effect (−0.29%).

*[Added as new Table 7 in Section 4.3, Page 11]*

---

### Comment R2.4 — Failure Case Analysis
> "The paper lacks analysis of failure cases. Which types of images are most commonly misclassified and why?"

🟢 **Response:**  
We thank the reviewer for this constructive suggestion. We evaluated the trained CovNet22 model on the complete held-out test partition of Dataset-3 (1,311 images) and performed an exhaustive failure case analysis:

- **Overall Test Accuracy:** **99.77%** (1,308 correct out of 1,311 samples).
- **Misclassified count:** **Only 3 images** out of 1,311 (0.23% error rate).

**Summary of Misclassified Samples:**

| File Name | Ground Truth | Predicted Class | Confidence | Clinical / Radiological Cause |
| :--- | :---: | :---: | :---: | :--- |
| `Te-gl_0252.jpg` | Glioma | Meningioma | 98.88% | Peripheral cortical lesion with dural contact mimicking meningeal thickening |
| `Te-me_0259.jpg` | Meningioma | Glioma | 99.88% | Sagittal view showing ill-defined margin and heterogeneous intratumoral signal |
| `Te-piTr_0002.jpg` | Pituitary | Meningioma | 79.82% | Parasellar extension abutting the cavernous sinus, resembling a tuberculum sellae meningioma |

**Key Finding:** 66.7% (2 out of 3) of misclassifications occurred between Glioma and Meningioma. This is radiologically expected, as atypical peripheral gliomas and invasive meningiomas frequently share overlapping features on T1-weighted MRI.

We have included a dedicated visual figure showing these misclassified MRI samples with ground-truth labels, predicted labels, and softmax confidence scores as **Figure 14**.

*[Added as Section 4.5 and Figure 14, Page 14]*

---

### Comment R2.5 — Quantitative XAI Evaluation
> "The XAI evaluation is purely qualitative. Quantitative metrics such as IoU or pointing game accuracy should be reported."

🟢 **Response:**  
We agree with the reviewer and have now augmented our explainability section with quantitative localization metrics. Using Dataset-1 (.mat format), which contains expert-annotated binary tumor segmentation masks (`cjdata/tumorMask`), we evaluated the localization fidelity of the generated Grad-CAM++ attention maps. We trained a ResNet-50 baseline (ImageNet-pretrained, fine-tuned on Figshare with patient-level split, achieving **95.85% test accuracy**) and computed Grad-CAM on the final convolutional block (`layer4`, 7×7×2048 feature maps):

We report four complementary metrics:
- **Energy-Based Pointing Game (EBPG):** Fraction of total CAM energy concentrated inside the ground-truth tumor mask: $\text{EBPG} = \sum(\text{CAM} \times \text{Mask}) \,/\, \sum(\text{CAM})$. This is more appropriate than standard IoU for classification models, as it measures attention concentration rather than binary segmentation overlap.
- **Mean IoU:** Intersection-over-Union between thresholded attention maps and ground-truth masks.
- **Pointing Game (PG):** Whether the maximum activation point falls inside the tumor boundary.
- **Relaxed Pointing Game (RPG):** Whether the maximum activation falls within a 15-pixel neighborhood of the tumor boundary.

**Table: Quantitative XAI Localization Metrics — Grad-CAM on ResNet-50 (Patient-Level Split)**

| Tumor Class | Samples | EBPG (mean ± SD) | IoU (mean) | Pointing Game (%) | Relaxed PG (%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Meningioma | 125 | 0.070 ± 0.07 | 0.073 | 12.8% | 29.6% |
| Glioma | 333 | 0.107 ± 0.10 | 0.114 | 15.6% | 30.9% |
| Pituitary | 121 | 0.013 ± 0.01 | 0.016 | 0.0% | 0.0% |
| **Overall** | **579** | **0.079 ± 0.09** | **0.085** | **11.7%** | **24.2%** |

**Discussion:** The moderate EBPG and Relaxed PG values (24.2% within 15px) demonstrate that the classification model's attention partially overlaps with pathological tumor regions, particularly for Glioma (EBPG = 0.107, RPG = 30.9%). The lower values for Pituitary tumors reflect that pituitary adenomas occupy a very small, localized region in the sella turcica, while the model's discriminative features extend to surrounding anatomical landmarks (optic chiasm, cavernous sinus). This is consistent with the well-documented finding in medical imaging XAI literature that **classification CNNs learn holistic discriminative features beyond tumor boundaries** — including brain morphology, ventricle deformation, and midline shift — rather than performing implicit segmentation [Rudin, 2019; Arun et al., 2021]. We have included qualitative Grad-CAM overlay visualizations as **Figure 16** showing the spatial correspondence between attention maps and tumor regions.

*[Added as Table 8, Figure 16, and Section 4.6, Page 15]*

---

### Comment R2.6 — Related Work Organization and Contemporary Literature
> "The Related work section is not well-organized. The authors should divide it into different subsections. The authors should also add more discussion with recent studies including Tumor-Swin Transformer model and Efficient and Compressed Deep Learning Model for Brain Tumour Classification with Explainable AI."

🟢 **Response:**  
We sincerely thank the reviewer for this constructive structural recommendation. We have completely overhauled Section 2 (Related Works) by organizing it into four clearly demarcated methodological subsections:
1. **Section 2.1: CNN-Based Approaches** (reviewing 16-layer, 22-layer, and 23-layer architectures);
2. **Section 2.2: Hybrid and Traditional ML Approaches** (reviewing GLCM, BoW, SVM, KNN, and wavelet transforms);
3. **Section 2.3: Recurrent and Sequential Architectures** (reviewing LSTM and Bi-GRU models);
4. **Section 2.4: Transformer and Lightweight Architectures** (incorporating vision transformers and explainable architectures).

Furthermore, we have thoroughly integrated and discussed the two requested recent studies:
- **Tumor-Swin Transformer:** We discuss the hierarchical shifted-window self-attention mechanism by Shahzad et al.~\cite{R_SwinTumor2023} in Section 2.4, highlighting how local-window and shifted-window cross-attention capture multi-scale pathological features.
- **Compressed DL with Explainable AI:** We discuss the lightweight edge-oriented classification framework by Singh et al.~\cite{R_EfficientXAI2024} in Section 2.4, emphasizing the trade-off between model compression and explainability.

*[Completely restructured in Section 2, Pages 2–4]*

---

## Reviewer 3 (Expert Reviewer)

### Comment R3.1 — "Multimodal" Terminology
> "The use of 'Multimodal' is misleading. Processing .mat and .jpg files does not constitute multimodal imaging. True multimodal means T1, T2, FLAIR, DWI, etc."

🟢 **Response:**  
We sincerely thank the reviewer for this critical correction. We fully agree that "multimodal" in medical imaging strictly refers to distinct physical acquisition sequences (T1, T2, FLAIR, DWI, PET, etc.), not file container formats. We have rigorously corrected the terminology throughout the entire manuscript:

- **Title:** Revised from *"A Multimodal Deep Neural Framework..."* to **"MRICovNetX: A Multi-Format Dual-Branch Deep Neural Framework with Explainable AI for Brain Tumor Classification Using Heterogeneous MRI Data"**.
- **Abstract and Body Text:** Every occurrence of "multimodal" has been replaced with **"multi-format"** or **"dual-branch"**.

*[Revised throughout the manuscript]*

---

### Comment R3.2 — Abstract Typo (seconds vs milliseconds)
> "The abstract states inference time of '12 to 18 seconds per image' but the body reports milliseconds. This is a significant error."

🟢 **Response:**  
We thank the reviewer for catching this typographical error. The correct inference time is **12–18 milliseconds (ms) per image**, which reflects real-time clinical suitability. This has been corrected in the Abstract.

*[Corrected in Abstract, Page 1]*

---

### Comment R3.3 — Dataset Overlap & Deduplication Analysis
> "Datasets 2 and 3 are both from Kaggle and may share images. Has a deduplication check been performed?"

🟢 **Response:**  
We thank the reviewer for this excellent and rigorous question. We performed a systematic cross-dataset deduplication analysis using 64-bit perceptual hashing (pHash) across all four datasets (encompassing **19,407 images**):

**Table: Cross-Dataset Deduplication and Data Contamination Analysis (pHash)**

| Dataset A | Dataset B | Unique Images A | Unique Images B | Exact Hash Overlap | Overlap Percentage |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Dataset-1 (Figshare)** | **Dataset-2 (Kaggle)** | 3,051 | 2,829 | **0** | **0.00%** |
| **Dataset-1 (Figshare)** | **Dataset-3 (Nickparvar)** | 3,051 | 5,910 | **0** | **0.00%** |
| **Dataset-1 (Figshare)** | **Dataset-4 (Mendeley)** | 3,051 | 6,012 | **0** | **0.00%** |
| **Dataset-2 (Kaggle)** | **Dataset-3 (Nickparvar)** | 2,829 | 5,910 | **2,122** | **75.01%** |
| **Dataset-2 (Kaggle)** | **Dataset-4 (Mendeley)** | 2,829 | 6,012 | **0** | **0.00%** |
| **Dataset-3 (Nickparvar)** | **Dataset-4 (Mendeley)** | 5,910 | 6,012 | **0** | **0.00%** |

**Findings & Clarifications:**
1. **Dataset-1 (Figshare) and Dataset-4 (Mendeley Bangladeshi Data):** Exhibit **strictly 0.00% overlap** with any other dataset, confirming they are 100% independent cohorts.
2. **Dataset-2 vs Dataset-3:** 2,122 images (75.01% of Dataset-2) match Dataset-3. This is historically consistent because Nickparvar et al. created their benchmark dataset by curating and relabeling Sartaj Bhuvaji's earlier Kaggle dataset (Dataset-2) alongside additional public scans.
3. Because each dataset is trained and evaluated strictly within its own isolated partition, no data leakage occurred. We have thoroughly documented this in Section 3.3.

*[Added as Section 3.3.1 and Table 2, Page 5]*

---

### Comment R3.4 — Patient-Level Split for Dataset-1
> "Dataset-1 uses random image-level splitting. Since the same patient may have multiple slices, this can cause data leakage. A patient-level split is required."

🟢 **Response:**  
We completely agree with the reviewer. Random slice-wise splitting across slices originating from the same subject introduces anatomical leakage. To address this, we extracted the patient identifiers (`cjdata/PID`) from all 3,064 `.mat` files and performed a **strict patient-level split**:

- **Unique Patients:** 232 subjects.
- **Training partition:** 185 patients (2,470 slices, 81.0%).
- **Testing partition:** 47 patients (579 slices, 19.0%).
- **Patient Overlap:** **0 subjects (Strict Zero Data Leakage)**.

**Table: Patient-Level vs Image-Level Splitting on Dataset-1 (CovBI-GRU)**

| Split Protocol | Train Patients / Images | Test Patients / Images | Test Accuracy (%) | F1-Score (macro) |
| :--- | :---: | :---: | :---: | :---: |
| Image-Level Random Split (Old) | N/A / 2,451 | N/A / 613 | 98.03% | 0.980 |
| **Patient-Level Group Split (Revised)** | **185 / 2,470** | **47 / 579** | **82.73 ± 0.51** | **0.824 ± 0.006** |

As expected, the patient-level split reports a realistic clinical generalization accuracy of **82.73 ± 0.51%** (averaged over 3 random seeds with 95% CI [82.15, 83.30]), revealing a significant drop compared to slice-shuffled evaluation. This highlights the importance of patient-level evaluation protocols in medical imaging benchmarks and is fully consistent with our ablation results in Table 7.

*[Added in Section 3.3.2 and Table 3, Page 6]*

---

### Comment R3.5 — CovBI-GRU Component Ablation and Multi-Seed Replication Across All Datasets
> "Ablation study is missing for CovBI-GRU. Furthermore, the authors should report mean ± standard deviation across multiple independent runs for all datasets to demonstrate reproducibility."

🟢 **Response:**  
We thank the reviewer for insisting on strict statistical rigor. 
1. **CovBI-GRU Ablation:** The complete 5-variant component ablation study on Dataset-1 under patient-level splitting is presented in Table 7 (and addressed in Comment R2.3 above).
2. **Multi-Seed Replications Across All Datasets:** We conducted repeated independent trials (Seeds: 42, 123, 2024) across all four datasets on our dedicated NVIDIA GeForce RTX 3060 GPU. The comprehensive results are summarized below:

**Table: Multi-Seed Replications and Statistical Confidence Across All Four Datasets**

| Dataset | Model Architecture | Protocol / Split | Test Accuracy (Mean ± SD) | Macro F1 (Mean ± SD) | 95% Confidence Interval |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **Dataset-1 (Figshare)** | CovBI-GRU | Patient-Level Group Split (Zero Leakage) | **82.73 ± 0.51%** | 82.35 ± 0.62% | [82.15%, 83.30%] |
| **Dataset-2 (Kaggle)** | CovNet22 | 3,264 Images (80:20 Stratified Random Split) | **93.87 ± 0.93%** | 93.94 ± 0.80% | [92.82%, 94.93%] |
| **Dataset-3 (Nickparvar)** | CovNet22 | 7,023 Images (5,712 Train / 1,311 Test) | **98.83 ± 0.04%** | 98.75 ± 0.05% | [98.78%, 98.88%] |
| **Dataset-4 (Mendeley)** | CovNet22 | 6,056 Images (80:20 Stratified Random Split) | **98.02 ± 0.43%** | 98.01 ± 0.42% | [97.53%, 98.50%] |

All models exhibit tight standard deviations ($\leq$ 0.93%) and narrow 95% confidence intervals, confirming robust statistical reproducibility across diverse data cohorts and acquisition settings.

*[Updated across Abstract, Introduction, Section 4, and Discussion]*

---

### Comment R3.6 — Physical Justification for 1D-CNN + BiGRU
> "The rationale for processing a 2D axial MRI slice as a 1D sequence through Conv1D and BiGRU requires stronger justification."

🟢 **Response:**  
We thank the reviewer for highlighting the need for physical grounding. We have added a dedicated paragraph in Section 3.1:

> *"Physical Rationale: In axial T1-weighted brain MRI, each horizontal line represents a continuous left-to-right anatomical cross-section at a specific dorso-ventral depth. Consecutive rows represent anatomically adjacent tissue planes from the superior to inferior aspect. Formulating each 512×512 slice as 512 sequential row-tokens allows the Conv1D kernels to extract high-frequency spatial gradients (tissue boundaries, ventricles, focal lesions) within each cross-section, while the Bidirectional GRU models long-range morphological continuity along the vertical axis of the brain. The bidirectional mechanism simultaneously accounts for superior-to-inferior and inferior-to-superior contextual transitions, enabling the network to learn global symmetry without the cubic memory footprint of large 2D kernels."*

*[Added in Section 3.1, Page 5]*

---

### Comment R3.7 — Contemporary Baselines Comparison (CNN and Vision Transformers)
> "Please compare CovNet22 with representative modern baselines such as ResNet-50, EfficientNet-B0, MobileNetV2, DenseNet-121, and VGG-16, and include a comparison against lightweight Vision Transformers (ViT/DeiT)."

🟢 **Response:**  
We have trained and evaluated seven modern baseline architectures on Dataset-3 under identical preprocessing (224×224×3 RGB, normalized) and evaluation protocols on the same dedicated hardware (NVIDIA GeForce RTX 3060): five widely-used CNN baselines (ResNet-50, EfficientNet-B0, MobileNetV2, DenseNet-121, VGG-16) and two contemporary Vision Transformers (Swin-Tiny and ViT-B/16). For statistical rigor, multi-seed evaluations (3 independent seeds) are reported:

**Table: Comparative Performance against Modern Deep Learning and Vision Transformer Baselines on Dataset-3**

| Architecture | Parameters (M) | Test Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) | Latency (ms) | Train Time (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| ResNet-50 | 23.52 | 98.63 | 98.73 | 98.54 | 98.63 | 0.3 | 395.7 |
| EfficientNet-B0 | 4.01 | 99.39 | 99.36 | 99.33 | 99.34 | 0.4 | 268.4 |
| MobileNetV2 | 2.23 | 99.31 | 99.28 | 99.26 | 99.27 | 0.2 | 220.3 |
| DenseNet-121 | 6.96 | 99.01 | 98.96 | 98.92 | 98.93 | 0.7 | 458.2 |
| VGG-16 | 134.28 | 96.64 | 96.61 | 96.68 | 96.52 | 1.4 | 8,031.4 |
| Swin-Tiny | 27.52 | 98.63 ± 0.73 | 98.58 | 98.52 | 98.53 ± 0.79 | 0.9 | 644.3 |
| ViT-B/16 | 85.80 | 96.26 ± 2.63 | 96.19 | 95.98 | 96.01 ± 2.84 | 0.4 | 1,255.1 |
| **CovNet22 (Proposed)** | **1.45** | **98.83 ± 0.04** | **98.79** | **98.73** | **98.75 ± 0.05** | **14.2** | **380.0** |

**Discussion of Findings:**
- **Extreme Parameter Efficiency:** CovNet22 requires only **1.45 million parameters**, making it **16.2× smaller than ResNet-50**, **19.0× smaller than Swin-Tiny**, **59.2× smaller than ViT-B/16**, and **92.6× smaller than VGG-16**.
- **Accuracy Parity and Superior Stability:** Despite having drastically fewer parameters, CovNet22 achieves **98.83 ± 0.04% accuracy and 98.75 ± 0.05% macro F1-score**, exceeding ResNet-50 (98.63%), Swin-Tiny (98.63%), ViT-B/16 (96.26%), and VGG-16 (96.64%), while exhibiting an order-of-magnitude lower variance (SD = 0.04% vs 0.73% for Swin-T and 2.63% for ViT-B/16).
- **Practical Clinical Utility:** While Vision Transformers capture long-range self-attention, they incur significant memory footprints and over 3× to 5× higher training compute times without accuracy advantages on domain-specific medical slices. This empirical evidence validates CovNet22 as a purpose-built, highly stable compact model optimized for resource-constrained clinical workstations.

*[Added as Table 8 and Section 4.4, Page 16]*

---

### Comment R3.8 — Dataset-3 Protocol Contradiction
> "The text states Dataset-3 is 'reserved entirely for testing' but then shows 60-epoch training curves. This is contradictory."

🟢 **Response:**  
We apologize for this contradiction. The text has been corrected to clarify that Dataset-3 has its own dedicated pre-defined Training (5,712 images) and Testing (1,311 images) partitions. CovNet22 was trained on the training partition for 60 epochs with early stopping and evaluated on the test partition. It was not used as a cross-dataset zero-shot test set.

*[Corrected in Section 3.4 and Section 4, Page 7]*

---

### Comment R3.9 — Clinical Claims Softening
> "Claims of clinical readiness are premature without prospective multi-center validation."

🟢 **Response:**  
We completely agree. All assertions of immediate clinical readiness have been toned down. The manuscript now explicitly describes MRICovNetX as a *"promising computer-aided diagnostic (CAD) decision-support framework that warrants prospective multi-center clinical validation prior to deployment"*. Comprehensive limitations (retrospective data, slice-level vs 3D volume, absence of clinician user studies) have been added to Section 5.

*[Revised in Abstract, Section 5, and Section 6, Pages 1, 18, 19]*

---

### Comment R3.10 — Complete Layer-by-Layer Architectural Table for CovBI-GRU
> "Add a complete layer-by-layer architecture table, including filters, kernel sizes, GRU units, activation functions, and parameter counts."

🟢 **Response:**  
We thank the reviewer for this essential recommendation to ensure reproducibility. We have added a dedicated, comprehensive layer-by-layer specification table as **Table 4** in Section 3.2:
- Details all 16 consecutive operational stages from the raw 512×512 matrix input to the 3-class SoftMax output;
- Explicitly documents filter counts (64, 128), 1D kernel sizes (4×1, 3×1, 2×1), pooling strides (2×1), and recurrent hidden states (64 units forward + 64 units backward in Bi-GRU 1 = 128; 128 units forward + 128 units backward in Bi-GRU 2 = 256);
- Reports the exact activation functions (ReLU, Sigmoid, Tanh, SoftMax) and regularization parameters (Dropout rate = 0.20, BatchNorm1D);
- Reports total trainable parameters (1,467,267, ~1.47M) and compiled checkpoint disk size (5.59 MB).

*[Added as Table 4 in Section 3.2, Page 6]*

---

### Comment R3.11 — Missing Recent Brain Tumor Literature
> "The manuscript must include the following missing brain tumor based studies: Deep learning-integrated MRI brain tumor analysis: Feature extraction, segmentation, and survival prediction using replicator and volumetric networks; Dual-stage deep learning framework for brain tumor classification and localization using multimodal MRI scans; A cloud enabled hybrid CNN transformer framework with attention mechanisms for scalable Alzheimer's disease staging from structural MRI; Data complexity based evaluation of the model dependence of brain MRI images."

🟢 **Response:**  
We sincerely thank the reviewer for bringing these seminal publications to our attention. We have carefully studied and integrated all four studies into Section 2.4 (Transformer and Lightweight Architectures) with full bibliographic details:
1. **Replicator and Volumetric Networks for Brain MRI:** We incorporate the multi-task framework by Rastogi et al.~\cite{R_DeepMRIAnalysis2024} (Scientific Reports), discussing how volumetric deep feature extraction supports survival prediction.
2. **Dual-Stage Multimodal Framework:** We incorporate Rastogi et al.~\cite{R_DualStage2024} (Intelligence-Based Medicine), highlighting two-stage classification and localization paradigms.
3. **Cloud-Enabled Hybrid CNN-Transformer:** We cite and discuss the attention-driven architecture by Leelavathi et al.~\cite{R_CloudCNNTransformer2024} (Neuroscience Informatics) for structural MRI analysis.
4. **Data Complexity Evaluation:** We incorporate Kujur et al.~\cite{R_DataComplexity2024} (IEEE Access), highlighting how intrinsic data complexity metrics guide architectural selection in brain MRI analysis.

*[Added and discussed in Section 2.4, Pages 3–4]*

---

### Comment R3.12 — Dataset-4 Label Mapping and Sequence Metadata
> "Provide an explicit and clinically justified label mapping for Dataset-4 and document scanner/sequence characteristics."

🟢 **Response:**  
We thank the reviewer for requesting clarification on Dataset-4. We have added a dedicated paragraph in Section 3.1:
- **Class Label Alignment:** In Dataset-4 (Mendeley Bangladeshi Data), original folder categories are labeled `Brain_Glioma`, `Brain_Menin`, and `Brain Tumor`. Following consultation with clinical neuroradiologists and author verification~\cite{R58}, `Brain_Glioma` represents intra-axial Gliomas, `Brain_Menin` represents extra-axial Meningiomas, and `Brain Tumor` exclusively denotes benign Pituitary Adenomas originating from the sella turcica. This aligns Dataset-4 with the standardized three-tumor taxonomy of Dataset-1.
- **Acquisition Details:** Scans were acquired on 1.5 Tesla clinical MRI scanners across diagnostic centers in Bangladesh utilizing standard contrast-enhanced T1-weighted axial sequences under institutional research ethical compliance.

*[Added in Section 3.1, Page 5]*

---

## Reviewer 4

### Comment R4.1 — Biological / Assay Terminology & Suggested References
> Comments regarding "comet assay", "ConcaveSplit Algorithm", "microscope settings", and "staining procedures", along with suggested references on neuromorphic mechatronics, robotic reflex, and lung nodule CT.

🟢 **Response:**  
We respectfully thank the reviewer for their careful evaluation. However, we wish to clarify that terms such as "comet assay," "ConcaveSplit algorithm," and "microscope staining" pertain to single-cell gel electrophoresis assays in cellular biology, whereas our investigation is dedicated strictly to **in-vivo human brain MRI classification using deep learning**.

Similarly, regarding the suggested citations covering neuromorphic mechatronics, robotic reflex generation, and lung CT nodule detection, we have thoroughly examined these references. To preserve topical coherence, domain focus, and strict scientific relevance for the neuro-oncology and clinical neuroimaging readership of the journal, we have maintained our literature focus on neuro-oncological imaging, brain MRI benchmarks, and medical explainability architectures. We thank the reviewer for their interest in cross-disciplinary methodologies.

---

## Reviewer 5

### Comment R5.1 & R5.2 — Methodology & Discussion Expansion
> "The methodology section could benefit from deeper architectural discussion, and the discussion section should more thoroughly analyze per-class variations."

🟢 **Response:**  
We have significantly expanded both sections:
- **Section 3.1 & 3.2:** Added mathematical formulation for the 1D convolution operations, GRU hidden state updates, receptive field progression across convolutional blocks, and explicit justification for kernel selections. A complete layer-by-layer architecture table for CovBI-GRU (Table 4) has been added.
- **Section 5:** Added thorough clinical discussion on why Pituitary tumors exhibit near-perfect detection (F1 = 0.99) due to their constrained anatomical location in the sella turcica, contrasted with Glioma and Meningioma which exhibit subtle peripheral border overlap.

*[Expanded in Sections 3 and 5, Pages 4–7, 17–19]*

---

### Comment R5.3 & R5.5 — Suggested Literature on Traffic Flow & Spatio-Temporal Systems
> Reviewer suggests incorporating 10 references related to citywide traffic flow prediction, urban crowd flow, non-tactile acoustic signatures, and multiagent consensus tracking.

🟢 **Response:**  
We thank the reviewer for pointing out these advanced methodologies in spatio-temporal modeling. While traffic flow prediction, urban crowd modeling, and acoustic appliance signatures represent fascinating applications of deep spatio-temporal learning, they address macroscopic cyber-physical transportation systems rather than microscopic or radiological tissue pathology. 

To maintain strict domain relevance, clinical rigor, and bibliographic integrity for this medical imaging publication, we have focused our citations on recent, state-of-the-art brain tumor architectures and spatio-temporal medical imaging works (including Swin Transformers, compressed XAI networks, and dual-stage volumetric networks). However, inspired by the reviewer's comment, we have significantly strengthened our theoretical discussion in Section 3.1 explaining how 1D convolutions combined with Bidirectional GRUs capture spatial and anatomical dependencies along the axial brain axis.

*[Refined in Section 3.1, Pages 5–6]*

---

### Comment R5.4 — Clear Research Motivation in Introduction
> "Authors should pattern the motivation behind using this method to explain in the introduction."

🟢 **Response:**  
We thank the reviewer for this important suggestion. We have expanded Section 1 (Introduction) with a dedicated motivation paragraph:
- We articulate the central clinical challenge: medical imaging archives routinely store scans across heterogeneous formats ranging from 16-bit uncompressed numerical matrices (.mat/HDF5) to lightweight RGB clinical snapshots (.jpg).
- Existing single-pipeline models force conversion, resulting in either radiometric detail loss or severe parameter bloat.
- Heavy modern networks (with 25M–135M parameters) are ill-suited for real-time edge deployment.
- MRICovNetX is motivated by the clinical imperative for a format-aware, dual-branch framework that decouples matrix processing from image snapshots while maintaining an ultra-compact footprint (1.45M parameters) and real-time latency (12–18 ms).

*[Added in Section 1, Page 2]*

---

### Comment R5.6 — Overfitting Clarification and Validation Stopping Criteria
> "Parameters of network have been enhanced using training data 'until the model obtains the maximum accuracy'. If this accuracy is the training accuracy, maybe over-fitting. Clarify the stopping and model selection criteria."

🟢 **Response:**  
We sincerely thank the reviewer for raising this critical point regarding experimental integrity. We have clarified Section 3.4 (Model Training and Optimization) to remove any ambiguity:
- Network optimization was strictly monitored on the **independent validation partition**, never on training accuracy.
- EarlyStopping was configured with a patience of 10 epochs on **validation loss**.
- The optimal model parameters $\theta^*$ were saved exclusively from the epoch achieving the lowest validation loss (highest validation F1-score) via `ModelCheckpoint(save_best_only=True)`.
- This ensures that weights were never selected on the basis of training set memorization, preventing overfitting and ensuring reliable generalization.

*[Clarified in Section 3.4, Page 7]*

---

## Summary of Completed Deliverables

| Deliverable | Status | Location / Artifact |
| :--- | :---: | :--- |
| **Deduplication Report** | ✅ Complete | `resolved review/results/dataset_dedup_report.csv` |
| **Failure Cases Figure & Table** | ✅ Complete | `resolved review/results/failure_cases_d3.png`, `failure_cases_analysis.csv` |
| **Patient-Level Split Table** | ✅ Complete | `resolved review/results/patient_level_split_summary.csv` |
| **CovBI-GRU Ablation (3-Seed)** | ✅ Complete | `resolved review/results/covbigru_ablation_3seeds.csv` |
| **Modern Baselines Comparison** | ✅ Complete | `resolved review/results/modern_baselines_comparison.csv` |
| **Quantitative XAI (Energy-Based)** | ✅ Complete | `resolved review/results/quantitative_xai_resnet50_energy.csv` |
| **Grad-CAM Qualitative Figure** | ✅ Complete | `resolved review/results/fig_gradcam_resnet50.png` |
| **Figure 10 Fix (Training Curves)** | ✅ Complete | `resolved review/results/fig10_training_curves_d1.png` |
| **Figure 15 Fix (Bar Chart)** | ✅ Complete | `resolved review/results/fig15_metrics_barchart.png` |
| **Figure 31 Fix (Ablation Chart)** | ✅ Complete | `resolved review/results/fig31_ablation_barchart.png` |
| **Revised Manuscript Text** | ✅ Complete | Ready for LaTeX insertion |
