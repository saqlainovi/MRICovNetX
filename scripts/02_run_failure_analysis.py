"""
=============================================================================
MRICovNetX — Experiment 2: Failure Case Analysis (Reviewer 2, Comment 7)
=============================================================================
Evaluates the trained CovNet22 model on Dataset-3 testing data, identifies
all misclassified samples, analyzes common error patterns, and generates
a publication-quality 3x4 grid visualization.
=============================================================================
"""

import os
import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

BASE_DIR = Path(__file__).resolve().parent.parent
DATASETS_DIR = BASE_DIR / "datasets"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = Path(r"J:\OneDrive\WORK\RECHARCH TEAM\MRICovNetX-main\old data\saved_models\brain_tumor_best_model.h5")
TEST_DIR = DATASETS_DIR / "Dataset_3_Nickparvar" / "Testing"

CLASSES = {'glioma': 0, 'meningioma': 1, 'notumor': 2, 'pituitary': 3}
INV_CLASSES = {v: k for k, v in CLASSES.items()}
CLASS_DISPLAY = {'glioma': 'Glioma', 'meningioma': 'Meningioma', 'notumor': 'No Tumor', 'pituitary': 'Pituitary'}

def main():
    print("=" * 60)
    print("EXPERIMENT 2: Failure Case Analysis on Dataset-3")
    print("=" * 60)
    
    # 1. Load trained model
    print(f"Loading trained CovNet22 model from:\n  {MODEL_PATH}")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
    
    model = tf.keras.models.load_model(str(MODEL_PATH))
    print("Model loaded successfully!")
    model.summary()

    # 2. Load Testing Images
    print(f"\nLoading test images from:\n  {TEST_DIR}")
    X_test = []
    Y_test = []
    filenames = []

    for cls_name, cls_idx in CLASSES.items():
        cls_folder = TEST_DIR / cls_name
        if not cls_folder.exists():
            print(f"Warning: {cls_folder} not found!")
            continue
        
        count = 0
        for f in os.listdir(cls_folder):
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = cls_folder / f
                img = cv2.imread(str(img_path), 1)
                if img is not None:
                    img_res = cv2.resize(img, (224, 224))
                    X_test.append(img_res)
                    Y_test.append(cls_idx)
                    filenames.append(f)
                    count += 1
        print(f"  Loaded {count} test images for class: {cls_name}")

    X_test = np.array(X_test, dtype=np.float32)
    Y_test = np.array(Y_test, dtype=np.int32)
    print(f"\nTotal test samples: {len(X_test)}")

    # 3. Predict
    print("Running predictions...")
    preds_prob = model.predict(X_test, batch_size=32, verbose=1)
    preds = np.argmax(preds_prob, axis=1)

    acc = accuracy_score(Y_test, preds)
    print(f"\nOverall Test Accuracy: {acc * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(Y_test, preds, target_names=[CLASS_DISPLAY[INV_CLASSES[i]] for i in range(4)]))

    # 4. Identify Misclassified Samples
    wrong_mask = (preds != Y_test)
    wrong_indices = np.where(wrong_mask)[0]
    total_wrong = len(wrong_indices)
    print(f"Total Misclassifications: {total_wrong} out of {len(Y_test)} ({total_wrong/len(Y_test)*100:.2f}%)")

    # Confusion breakdown
    cm = confusion_matrix(Y_test, preds)
    print("\nConfusion Matrix:")
    print(cm)

    # 5. Save detailed failure log
    failure_records = []
    for idx in wrong_indices:
        t_label = CLASS_DISPLAY[INV_CLASSES[Y_test[idx]]]
        p_label = CLASS_DISPLAY[INV_CLASSES[preds[idx]]]
        conf = float(preds_prob[idx][preds[idx]])
        t_conf = float(preds_prob[idx][Y_test[idx]])
        failure_records.append({
            "File Name": filenames[idx],
            "True Label": t_label,
            "Predicted Label": p_label,
            "Pred Confidence (%)": f"{conf * 100:.2f}%",
            "True Class Probability (%)": f"{t_conf * 100:.2f}%"
        })

    df_fail = pd.DataFrame(failure_records)
    fail_csv = RESULTS_DIR / "failure_cases_analysis.csv"
    df_fail.to_csv(fail_csv, index=False)
    print(f"\nFailure cases log saved to: {fail_csv}")

    # 6. Generate 3x4 Grid Visualization of Misclassified Samples
    n_plot = min(12, total_wrong)
    if n_plot > 0:
        fig, axes = plt.subplots(3, 4, figsize=(16, 12), dpi=300)
        axes = axes.flatten()

        for i in range(n_plot):
            idx = wrong_indices[i]
            img_rgb = cv2.cvtColor(X_test[idx].astype(np.uint8), cv2.COLOR_BGR2RGB)
            t_label = CLASS_DISPLAY[INV_CLASSES[Y_test[idx]]]
            p_label = CLASS_DISPLAY[INV_CLASSES[preds[idx]]]
            conf = float(preds_prob[idx][preds[idx]]) * 100

            axes[i].imshow(img_rgb)
            axes[i].set_title(f"True: {t_label}\nPred: {p_label} ({conf:.1f}%)", 
                              fontsize=11, color="crimson", fontweight="bold", pad=8)
            axes[i].axis('off')
            
            # Thick red border for error identification
            for spine in axes[i].spines.values():
                spine.set_edgecolor('crimson')
                spine.set_linewidth(3.5)
                spine.set_visible(True)

        for i in range(n_plot, 12):
            axes[i].axis('off')

        plt.suptitle("Representative Misclassified Brain MRI Samples (CovNet22 on Dataset-3)", 
                     fontsize=15, fontweight="bold", y=0.98)
        plt.tight_layout()
        out_fig = RESULTS_DIR / "failure_cases_d3.png"
        plt.savefig(out_fig, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[SUCCESS] Failure cases figure saved to: {out_fig}")

if __name__ == "__main__":
    main()
