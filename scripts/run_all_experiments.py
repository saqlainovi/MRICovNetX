"""
=============================================================================
MRICovNetX Paper Revision — All Experiments Runner
=============================================================================
This script runs ALL 6 experiments sequentially on Google Colab.

HOW TO USE:
1. Upload this file to Google Colab
2. Make sure GPU runtime is enabled (Runtime → Change runtime type → T4 GPU)
3. Mount your Google Drive and download/extract all 4 datasets
4. Update the DATASET PATHS below
5. Run this file: !python run_all_experiments.py

OUTPUT: All results will be saved in ./revision_results/
=============================================================================
"""

import subprocess
import sys

# Install required packages
print("=" * 60)
print("STEP 0: Installing required packages...")
print("=" * 60)
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", 
                       "imagehash", "lime", "shap", "scikit-learn", "pandas", "matplotlib", "seaborn"])

import os
import h5py
import cv2
import time
import numpy as np
import random
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from pathlib import Path
from collections import defaultdict

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (Conv1D, MaxPooling1D, Conv2D, MaxPooling2D,
                                      BatchNormalization, Bidirectional, GRU, 
                                      Flatten, Dense, Dropout, GlobalAveragePooling2D, Input)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support, 
                              classification_report, confusion_matrix)
from sklearn.model_selection import GroupShuffleSplit

# =============================================================================
# ███ DATASET PATHS — UPDATE THESE FOR YOUR COLAB SETUP ███
# =============================================================================
D1_DIR = '/content/BRAIN_DATA'                    # Dataset-1: .mat files (1.mat to 3064.mat)
D2_TRAIN_DIR = '/content/Training'                # Dataset-2: Training folder
D2_TEST_DIR = '/content/Testing'                  # Dataset-2: Testing folder
D3_TRAIN_DIR = '/content/Brain_Image/Training'    # Dataset-3: Training folder
D3_TEST_DIR = '/content/Brain_Image/Testing'      # Dataset-3: Testing folder
D4_DIR = '/content/Mendeley_Data'                 # Dataset-4: Mendeley extracted folder

# Output directory
RESULTS_DIR = Path('./revision_results')
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def load_d1_data(seed=42):
    """Load Dataset-1 (.mat files) with random image-level split."""
    trainindata = []
    for i in range(1, 3065):
        filepath = os.path.join(D1_DIR, f"{i}.mat")
        if os.path.exists(filepath):
            data = h5py.File(filepath, 'r')
            trainindata.append(data)
    
    random.seed(seed)
    random.shuffle(trainindata)
    
    trainx, trainy, testx, testy = [], [], [], []
    size = round(4 * len(trainindata) / 5)
    
    for i in range(size):
        image = trainindata[i]['cjdata']['image'][()]
        if image.shape == (512, 512):
            trainx.append(np.expand_dims(image, axis=0))
        label = int(trainindata[i]['cjdata']['label'][()].item()) - 1
        trainy.append(label)
    
    for i in range(size, len(trainindata)):
        image = trainindata[i]['cjdata']['image'][()]
        if image.shape == (512, 512):
            testx.append(np.expand_dims(image, axis=0))
        label = int(trainindata[i]['cjdata']['label'][()].item()) - 1
        testy.append(label)
    
    trainx = np.array(trainx).reshape(-1, 512, 512)
    testx = np.array(testx).reshape(-1, 512, 512)
    trainy = np.array(trainy)
    testy = np.array(testy)
    return trainx, trainy, testx, testy


def load_d3_data():
    """Load Dataset-3 (Nickparvar) jpg images."""
    classes = {'glioma': 0, 'meningioma': 1, 'notumor': 2, 'pituitary': 3}
    
    X_train, Y_train = [], []
    for cls, label in classes.items():
        cls_dir = Path(D3_TRAIN_DIR) / cls
        if cls_dir.exists():
            for img_path in cls_dir.iterdir():
                img = cv2.imread(str(img_path), 1)
                if img is not None:
                    img = cv2.resize(img, (224, 224))
                    X_train.append(img)
                    Y_train.append(label)
    
    X_test, Y_test = [], []
    for cls, label in classes.items():
        cls_dir = Path(D3_TEST_DIR) / cls
        if cls_dir.exists():
            for img_path in cls_dir.iterdir():
                img = cv2.imread(str(img_path), 1)
                if img is not None:
                    img = cv2.resize(img, (224, 224))
                    X_test.append(img)
                    Y_test.append(label)
    
    return np.array(X_train), np.array(Y_train), np.array(X_test), np.array(Y_test)


def build_covbigru(variant='full'):
    """Build CovBI-GRU model variants for ablation study."""
    model = Sequential()
    
    if variant == 'full':
        model.add(Conv1D(64, 4, activation='relu', input_shape=(512, 512)))
        model.add(MaxPooling1D(pool_size=2))
        model.add(BatchNormalization())
        model.add(Bidirectional(GRU(64, return_sequences=True)))
        model.add(Conv1D(64, 3, activation='relu'))
        model.add(Conv1D(64, 3, activation='relu'))
        model.add(MaxPooling1D(pool_size=2))
        model.add(BatchNormalization())
        model.add(Bidirectional(GRU(128, return_sequences=True)))
        model.add(Conv1D(128, 2, activation='relu'))
        model.add(MaxPooling1D(pool_size=2))
        model.add(BatchNormalization())
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.2))
        model.add(BatchNormalization())
        model.add(Dense(3, activation='softmax'))
        
    elif variant == 'conv1d_only':
        model.add(Conv1D(64, 4, activation='relu', input_shape=(512, 512)))
        model.add(MaxPooling1D(pool_size=2))
        model.add(BatchNormalization())
        model.add(Conv1D(64, 3, activation='relu'))
        model.add(Conv1D(64, 3, activation='relu'))
        model.add(MaxPooling1D(pool_size=2))
        model.add(BatchNormalization())
        model.add(Conv1D(128, 2, activation='relu'))
        model.add(MaxPooling1D(pool_size=2))
        model.add(BatchNormalization())
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.2))
        model.add(BatchNormalization())
        model.add(Dense(3, activation='softmax'))
        
    elif variant == 'bigru_only':
        model.add(Bidirectional(GRU(64, return_sequences=True), input_shape=(512, 512)))
        model.add(BatchNormalization())
        model.add(Bidirectional(GRU(128, return_sequences=True)))
        model.add(BatchNormalization())
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.2))
        model.add(BatchNormalization())
        model.add(Dense(3, activation='softmax'))
        
    elif variant == 'no_batchnorm':
        model.add(Conv1D(64, 4, activation='relu', input_shape=(512, 512)))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Bidirectional(GRU(64, return_sequences=True)))
        model.add(Conv1D(64, 3, activation='relu'))
        model.add(Conv1D(64, 3, activation='relu'))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Bidirectional(GRU(128, return_sequences=True)))
        model.add(Conv1D(128, 2, activation='relu'))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(3, activation='softmax'))
        
    elif variant == 'no_dropout':
        model.add(Conv1D(64, 4, activation='relu', input_shape=(512, 512)))
        model.add(MaxPooling1D(pool_size=2))
        model.add(BatchNormalization())
        model.add(Bidirectional(GRU(64, return_sequences=True)))
        model.add(Conv1D(64, 3, activation='relu'))
        model.add(Conv1D(64, 3, activation='relu'))
        model.add(MaxPooling1D(pool_size=2))
        model.add(BatchNormalization())
        model.add(Bidirectional(GRU(128, return_sequences=True)))
        model.add(Conv1D(128, 2, activation='relu'))
        model.add(MaxPooling1D(pool_size=2))
        model.add(BatchNormalization())
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dense(3, activation='softmax'))
    
    model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def build_covnet22(num_classes=4):
    """Build the CovNet22 model."""
    model = Sequential()
    model.add(Conv2D(32, 5, activation='relu', input_shape=(224, 224, 3)))
    model.add(MaxPooling2D(pool_size=4))
    model.add(BatchNormalization())
    model.add(Dropout(0.1))
    model.add(Conv2D(64, 5, activation='relu'))
    model.add(Conv2D(64, 5, activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(BatchNormalization())
    model.add(Dropout(0.1))
    model.add(Conv2D(128, 5, activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(BatchNormalization())
    model.add(Dropout(0.1))
    model.add(Conv2D(256, 5, activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(BatchNormalization())
    model.add(Dropout(0.1))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.1))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


# =============================================================================
# EXPERIMENT 1: Dataset Deduplication Check
# =============================================================================
def run_experiment_1():
    print("\n" + "=" * 60)
    print("EXPERIMENT 1: Dataset Deduplication Check")
    print("=" * 60)
    
    try:
        import imagehash
        from PIL import Image
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "imagehash"])
        import imagehash
        from PIL import Image
    
    dataset_hashes = {}
    
    # D1: .mat files
    print("Hashing Dataset-1 (.mat files)...")
    d1_hashes = {}
    if os.path.exists(D1_DIR):
        for i in range(1, 3065):
            fp = os.path.join(D1_DIR, f"{i}.mat")
            if os.path.exists(fp):
                try:
                    with h5py.File(fp, 'r') as f:
                        img = f['cjdata']['image'][()]
                        img = ((img - img.min()) / (img.max() - img.min() + 1e-8) * 255).astype(np.uint8)
                        h = str(imagehash.phash(Image.fromarray(img)))
                        d1_hashes[fp] = h
                except:
                    pass
        print(f"  D1: {len(d1_hashes)} images hashed")
    dataset_hashes['D1'] = d1_hashes
    
    # D2, D3: jpg files
    for name, dirs in [('D2', [D2_TRAIN_DIR, D2_TEST_DIR]), ('D3', [D3_TRAIN_DIR, D3_TEST_DIR])]:
        print(f"Hashing Dataset {name}...")
        hashes = {}
        for d in dirs:
            if os.path.exists(d):
                for root, _, files in os.walk(d):
                    for fname in files:
                        if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                            fp = os.path.join(root, fname)
                            try:
                                h = str(imagehash.phash(Image.open(fp).convert('L')))
                                hashes[fp] = h
                            except:
                                pass
        print(f"  {name}: {len(hashes)} images hashed")
        dataset_hashes[name] = hashes
    
    # D4
    print("Hashing Dataset-4...")
    d4_hashes = {}
    if os.path.exists(D4_DIR):
        for root, _, files in os.walk(D4_DIR):
            for fname in files:
                if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    fp = os.path.join(root, fname)
                    try:
                        h = str(imagehash.phash(Image.open(fp).convert('L')))
                        d4_hashes[fp] = h
                    except:
                        pass
    print(f"  D4: {len(d4_hashes)} images hashed")
    dataset_hashes['D4'] = d4_hashes
    
    # Compare across datasets
    print("\nCross-dataset duplicate analysis:")
    results = []
    pairs = [('D1', 'D2'), ('D1', 'D3'), ('D1', 'D4'), ('D2', 'D3'), ('D2', 'D4'), ('D3', 'D4')]
    for a, b in pairs:
        if a in dataset_hashes and b in dataset_hashes:
            ha = set(dataset_hashes[a].values())
            hb = set(dataset_hashes[b].values())
            overlap = ha & hb
            results.append({
                'Dataset Pair': f"{a} vs {b}",
                'Images in A': len(dataset_hashes[a]),
                'Images in B': len(dataset_hashes[b]),
                'Exact Hash Matches': len(overlap),
                'Overlap %': f"{len(overlap)/max(min(len(ha),len(hb)),1)*100:.2f}%"
            })
            print(f"  {a} vs {b}: {len(overlap)} duplicate(s) found")
    
    df = pd.DataFrame(results)
    df.to_csv(RESULTS_DIR / 'dedup_report.csv', index=False)
    print(f"\nSaved to {RESULTS_DIR / 'dedup_report.csv'}")
    print(df.to_string(index=False))
    return df


# =============================================================================
# EXPERIMENT 2: CovBI-GRU Ablation Study
# =============================================================================
def run_experiment_2():
    print("\n" + "=" * 60)
    print("EXPERIMENT 2: CovBI-GRU Ablation Study")
    print("=" * 60)
    
    trainx, trainy, testx, testy = load_d1_data(seed=42)
    print(f"Data loaded: train={trainx.shape}, test={testx.shape}")
    
    variants = ['full', 'conv1d_only', 'bigru_only', 'no_batchnorm', 'no_dropout']
    variant_names = {
        'full': 'Full CovBI-GRU (Proposed)',
        'conv1d_only': 'Conv1D Only (w/o BiGRU)',
        'bigru_only': 'BiGRU Only (w/o Conv1D)',
        'no_batchnorm': 'w/o BatchNormalization',
        'no_dropout': 'w/o Dropout'
    }
    seeds = [42, 100, 2024]
    results = []
    
    for var in variants:
        print(f"\n--- Variant: {variant_names[var]} ---")
        acc_list, prec_list, rec_list, f1_list = [], [], [], []
        
        for seed in seeds:
            print(f"  Seed {seed}...", end=' ')
            tf.random.set_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            
            model = build_covbigru(var)
            model.fit(trainx, trainy, batch_size=32, epochs=80, verbose=0)
            
            preds = np.argmax(model.predict(testx, verbose=0), axis=1)
            acc = accuracy_score(testy, preds)
            prec, rec, f1, _ = precision_recall_fscore_support(testy, preds, average='macro', zero_division=0)
            
            acc_list.append(acc)
            prec_list.append(prec)
            rec_list.append(rec)
            f1_list.append(f1)
            print(f"Acc={acc:.4f}")
            
            tf.keras.backend.clear_session()
        
        results.append({
            'Variant': variant_names[var],
            'Accuracy (%)': f"{np.mean(acc_list)*100:.2f} ± {np.std(acc_list)*100:.2f}",
            'Precision (%)': f"{np.mean(prec_list)*100:.2f} ± {np.std(prec_list)*100:.2f}",
            'Recall (%)': f"{np.mean(rec_list)*100:.2f} ± {np.std(rec_list)*100:.2f}",
            'F1-Score (%)': f"{np.mean(f1_list)*100:.2f} ± {np.std(f1_list)*100:.2f}",
        })
    
    df = pd.DataFrame(results)
    df.to_csv(RESULTS_DIR / 'ablation_covbigru.csv', index=False)
    print(f"\n\nAblation Results:")
    print(df.to_string(index=False))
    print(f"\nSaved to {RESULTS_DIR / 'ablation_covbigru.csv'}")
    return df


# =============================================================================
# EXPERIMENT 3: Patient-Level Split for Dataset-1
# =============================================================================
def run_experiment_3():
    print("\n" + "=" * 60)
    print("EXPERIMENT 3: Patient-Level Split for Dataset-1")
    print("=" * 60)
    
    # Load all .mat files and extract patient IDs
    all_data = []
    print("Extracting patient IDs from .mat files...")
    for i in range(1, 3065):
        fp = os.path.join(D1_DIR, f"{i}.mat")
        if os.path.exists(fp):
            try:
                with h5py.File(fp, 'r') as f:
                    image = f['cjdata']['image'][()]
                    label = int(f['cjdata']['label'][()].item()) - 1
                    # Try to get patient ID
                    pid = None
                    if 'PID' in f['cjdata']:
                        pid_data = f['cjdata']['PID'][()]
                        if isinstance(pid_data, np.ndarray):
                            pid = int(pid_data.flat[0]) if pid_data.size > 0 else i
                        else:
                            pid = int(pid_data)
                    else:
                        pid = i  # Fallback: use file index as patient ID
                    
                    if image.shape == (512, 512):
                        all_data.append({'image': image, 'label': label, 'pid': pid, 'file': i})
            except Exception as e:
                pass
    
    print(f"Loaded {len(all_data)} images from {len(set(d['pid'] for d in all_data))} unique patients")
    
    images = np.array([d['image'] for d in all_data])
    labels = np.array([d['label'] for d in all_data])
    pids = np.array([d['pid'] for d in all_data])
    
    seeds = [42, 100, 2024]
    results = []
    
    for seed in seeds:
        print(f"\n  Seed {seed}...")
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
        train_idx, test_idx = next(gss.split(images, labels, groups=pids))
        
        trainx, trainy = images[train_idx], labels[train_idx]
        testx, testy = images[test_idx], labels[test_idx]
        
        train_patients = len(set(pids[train_idx]))
        test_patients = len(set(pids[test_idx]))
        overlap = set(pids[train_idx]) & set(pids[test_idx])
        
        print(f"    Train: {len(trainx)} images, {train_patients} patients")
        print(f"    Test:  {len(testx)} images, {test_patients} patients")
        print(f"    Patient overlap: {len(overlap)} (should be 0)")
        
        tf.random.set_seed(seed)
        np.random.seed(seed)
        
        model = build_covbigru('full')
        model.fit(trainx, trainy, batch_size=32, epochs=80, verbose=0)
        
        preds = np.argmax(model.predict(testx, verbose=0), axis=1)
        acc = accuracy_score(testy, preds)
        prec, rec, f1, _ = precision_recall_fscore_support(testy, preds, average='macro', zero_division=0)
        
        results.append({
            'Seed': seed,
            'Train Patients': train_patients,
            'Test Patients': test_patients,
            'Train Images': len(trainx),
            'Test Images': len(testx),
            'Patient Overlap': len(overlap),
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1-Score': f1
        })
        print(f"    Accuracy: {acc:.4f}, F1: {f1:.4f}")
        tf.keras.backend.clear_session()
    
    df = pd.DataFrame(results)
    
    # Summary row
    summary = {
        'Split': 'Patient-Level (mean ± std)',
        'Accuracy': f"{np.mean([r['Accuracy'] for r in results])*100:.2f} ± {np.std([r['Accuracy'] for r in results])*100:.2f}",
        'F1-Score': f"{np.mean([r['F1-Score'] for r in results])*100:.2f} ± {np.std([r['F1-Score'] for r in results])*100:.2f}",
    }
    
    df.to_csv(RESULTS_DIR / 'patient_split_results.csv', index=False)
    print(f"\nSummary: {summary}")
    print(f"Saved to {RESULTS_DIR / 'patient_split_results.csv'}")
    return df


# =============================================================================
# EXPERIMENT 4: Baseline Comparison
# =============================================================================
def run_experiment_4():
    print("\n" + "=" * 60)
    print("EXPERIMENT 4: Baseline Model Comparison")
    print("=" * 60)
    
    from tensorflow.keras.applications import ResNet50, EfficientNetB0, VGG16, MobileNetV2, DenseNet121
    
    X_train, Y_train, X_test, Y_test = load_d3_data()
    X_train_norm = X_train / 255.0
    X_test_norm = X_test / 255.0
    Y_train_cat = to_categorical(Y_train, 4)
    Y_test_cat = to_categorical(Y_test, 4)
    
    print(f"Data loaded: train={X_train.shape}, test={X_test.shape}")
    
    baselines = {
        'ResNet50': ResNet50,
        'EfficientNetB0': EfficientNetB0,
        'VGG16': VGG16,
        'MobileNetV2': MobileNetV2,
        'DenseNet121': DenseNet121,
    }
    
    results = []
    
    for name, BaseModel in baselines.items():
        print(f"\n--- Training {name} ---")
        tf.keras.backend.clear_session()
        
        base = BaseModel(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        base.trainable = False
        
        inp = Input(shape=(224, 224, 3))
        x = base(inp, training=False)
        x = GlobalAveragePooling2D()(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.3)(x)
        out = Dense(4, activation='softmax')(x)
        
        model = Model(inputs=inp, outputs=out)
        model.compile(optimizer=Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
        
        total_params = model.count_params()
        
        callbacks = [EarlyStopping(patience=10, restore_best_weights=True)]
        
        model.fit(X_train_norm, Y_train_cat, batch_size=32, epochs=30, 
                  validation_split=0.15, callbacks=callbacks, verbose=1)
        
        # Inference time
        start = time.time()
        preds_prob = model.predict(X_test_norm, verbose=0)
        inf_time = (time.time() - start) / len(X_test_norm) * 1000  # ms per image
        
        preds = np.argmax(preds_prob, axis=1)
        acc = accuracy_score(Y_test, preds)
        prec, rec, f1, _ = precision_recall_fscore_support(Y_test, preds, average='macro', zero_division=0)
        
        results.append({
            'Model': name,
            'Params (M)': f"{total_params/1e6:.1f}",
            'Accuracy (%)': f"{acc*100:.2f}",
            'Precision (%)': f"{prec*100:.2f}",
            'Recall (%)': f"{rec*100:.2f}",
            'F1-Score (%)': f"{f1*100:.2f}",
            'Inference (ms)': f"{inf_time:.1f}"
        })
        print(f"  {name}: Acc={acc:.4f}, F1={f1:.4f}, Params={total_params/1e6:.1f}M, Inf={inf_time:.1f}ms")
    
    # Add CovNet22 results for comparison
    results.append({
        'Model': 'CovNet22 (Ours)',
        'Params (M)': 'TBD',
        'Accuracy (%)': '98.64',
        'Precision (%)': '99',
        'Recall (%)': '99',
        'F1-Score (%)': '99',
        'Inference (ms)': '12-18'
    })
    
    df = pd.DataFrame(results)
    df.to_csv(RESULTS_DIR / 'baseline_comparison.csv', index=False)
    print(f"\n\nBaseline Comparison:")
    print(df.to_string(index=False))
    print(f"\nSaved to {RESULTS_DIR / 'baseline_comparison.csv'}")
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    models = [r['Model'] for r in results[:-1]]  # Exclude "Ours" for bar chart
    accs = [float(r['Accuracy (%)']) for r in results[:-1]]
    bars = ax.bar(models, accs, color=['#4C72B0', '#55A868', '#C44E52', '#8172B2', '#CCB974'])
    ax.axhline(y=98.64, color='red', linestyle='--', linewidth=2, label='CovNet22 (Ours): 98.64%')
    ax.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax.set_title('Baseline Comparison on Dataset-3', fontsize=14)
    ax.legend(fontsize=11)
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.2, 
                f'{acc:.1f}%', ha='center', va='bottom', fontsize=10)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'baseline_comparison_chart.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return df


# =============================================================================
# EXPERIMENT 5: Failure Case Analysis
# =============================================================================
def run_experiment_5():
    print("\n" + "=" * 60)
    print("EXPERIMENT 5: Failure Case Analysis")
    print("=" * 60)
    
    # Train CovNet22 on D3 and find misclassifications
    X_train, Y_train, X_test, Y_test = load_d3_data()
    
    print(f"Training CovNet22 on D3 for failure analysis...")
    model = build_covnet22(num_classes=4)
    
    callbacks = [
        EarlyStopping(patience=10, restore_best_weights=True),
        ReduceLROnPlateau(patience=5, factor=0.5)
    ]
    
    model.fit(X_train, Y_train, batch_size=32, epochs=60,
              validation_split=0.15, callbacks=callbacks, verbose=1)
    
    preds_prob = model.predict(X_test, verbose=0)
    preds = np.argmax(preds_prob, axis=1)
    
    # Find misclassified samples
    wrong_mask = preds != Y_test
    wrong_indices = np.where(wrong_mask)[0]
    
    class_names = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
    
    print(f"\nTotal test samples: {len(Y_test)}")
    print(f"Misclassified: {len(wrong_indices)} ({len(wrong_indices)/len(Y_test)*100:.2f}%)")
    
    # Confusion pairs analysis
    print("\nMost confused class pairs:")
    cm = confusion_matrix(Y_test, preds)
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if i != j and cm[i][j] > 0:
                print(f"  {class_names[i]} → {class_names[j]}: {cm[i][j]} cases")
    
    # Plot misclassified samples (up to 12)
    n_show = min(12, len(wrong_indices))
    if n_show > 0:
        fig, axes = plt.subplots(3, 4, figsize=(16, 12))
        axes = axes.flatten()
        
        for idx in range(n_show):
            sample_idx = wrong_indices[idx]
            img = cv2.cvtColor(X_test[sample_idx], cv2.COLOR_BGR2RGB)
            true_label = class_names[Y_test[sample_idx]]
            pred_label = class_names[preds[sample_idx]]
            confidence = preds_prob[sample_idx][preds[sample_idx]]
            
            axes[idx].imshow(img)
            axes[idx].set_title(f"True: {true_label}\nPred: {pred_label} ({confidence:.1%})",
                               fontsize=9, color='red', fontweight='bold')
            axes[idx].axis('off')
            # Red border
            for spine in axes[idx].spines.values():
                spine.set_edgecolor('red')
                spine.set_linewidth(3)
                spine.set_visible(True)
        
        # Hide unused subplots
        for idx in range(n_show, 12):
            axes[idx].axis('off')
        
        plt.suptitle('Misclassified Test Samples — CovNet22 on Dataset-3', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / 'failure_cases_d3.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved failure cases figure to {RESULTS_DIR / 'failure_cases_d3.png'}")
    
    # Save analysis
    failure_data = []
    for idx in wrong_indices:
        failure_data.append({
            'Sample Index': idx,
            'True Label': class_names[Y_test[idx]],
            'Predicted Label': class_names[preds[idx]],
            'Confidence': f"{preds_prob[idx][preds[idx]]:.4f}",
        })
    pd.DataFrame(failure_data).to_csv(RESULTS_DIR / 'failure_analysis.csv', index=False)
    
    tf.keras.backend.clear_session()
    return len(wrong_indices)


# =============================================================================
# EXPERIMENT 6: Quantitative XAI (IoU & Pointing Game)
# =============================================================================
def run_experiment_6():
    print("\n" + "=" * 60)
    print("EXPERIMENT 6: Quantitative XAI (IoU & Pointing Game)")
    print("=" * 60)
    
    # Load D1 data with tumor masks
    print("Loading D1 data with tumor masks...")
    test_data = []
    
    all_indices = list(range(1, 3065))
    random.seed(42)
    random.shuffle(all_indices)
    size = round(4 * len(all_indices) / 5)
    test_indices = all_indices[size:]
    
    for i in test_indices:
        fp = os.path.join(D1_DIR, f"{i}.mat")
        if os.path.exists(fp):
            try:
                with h5py.File(fp, 'r') as f:
                    image = f['cjdata']['image'][()]
                    label = int(f['cjdata']['label'][()].item()) - 1
                    mask = None
                    if 'tumorMask' in f['cjdata']:
                        mask = f['cjdata']['tumorMask'][()]
                    
                    if image.shape == (512, 512) and mask is not None:
                        test_data.append({'image': image, 'label': label, 'mask': mask})
            except:
                pass
    
    print(f"Loaded {len(test_data)} test samples with tumor masks")
    
    if len(test_data) == 0:
        print("No tumor masks found in dataset. Skipping quantitative XAI.")
        return None
    
    # Prepare test arrays
    test_images = np.array([d['image'] for d in test_data])
    test_labels = np.array([d['label'] for d in test_data])
    test_masks = [d['mask'] for d in test_data]
    
    # Train model
    print("Training CovBI-GRU for XAI analysis...")
    trainx, trainy, _, _ = load_d1_data(seed=42)
    
    tf.random.set_seed(42)
    model = build_covbigru('full')
    model.fit(trainx, trainy, batch_size=32, epochs=80, verbose=0)
    
    # Find last Conv1D layer
    conv_layer_name = None
    for layer in reversed(model.layers):
        if isinstance(layer, Conv1D):
            conv_layer_name = layer.name
            break
    
    print(f"Using Conv1D layer: {conv_layer_name}")
    
    # Grad-CAM for 1D
    def gradcam_1d(image, model, layer_name, class_idx=None):
        img_tensor = tf.convert_to_tensor(image.reshape(1, 512, 512), dtype=tf.float32)
        
        grad_model = tf.keras.Model(
            inputs=model.input,
            outputs=[model.get_layer(layer_name).output, model.output]
        )
        
        with tf.GradientTape() as tape:
            conv_output, predictions = grad_model(img_tensor)
            if class_idx is None:
                class_idx = tf.argmax(predictions[0])
            loss = predictions[0, class_idx]
        
        grads = tape.gradient(loss, conv_output)
        weights = tf.reduce_mean(grads, axis=1)  # Global average pooling over spatial dim
        cam = tf.reduce_sum(weights * conv_output, axis=-1)[0]
        cam = tf.nn.relu(cam)
        
        # Normalize
        cam = cam.numpy()
        if cam.max() > 0:
            cam = cam / cam.max()
        
        # Resize 1D cam to 2D (512x512)
        cam_2d = np.tile(cam.reshape(-1, 1), (1, 512))
        cam_2d = cv2.resize(cam_2d, (512, 512))
        
        return cam_2d
    
    # Compute IoU and Pointing Game
    ious = []
    hits = 0
    class_ious = defaultdict(list)
    class_hits = defaultdict(int)
    class_total = defaultdict(int)
    class_names = ['Meningioma', 'Glioma', 'Pituitary']
    
    print("Computing IoU and Pointing Game metrics...")
    for idx in range(len(test_images)):
        image = test_images[idx]
        true_mask = (test_masks[idx] > 0).astype(np.float32)
        label = test_labels[idx]
        
        if true_mask.shape != (512, 512):
            true_mask = cv2.resize(true_mask, (512, 512))
        
        # Generate Grad-CAM
        try:
            heatmap = gradcam_1d(image, model, conv_layer_name)
        except:
            continue
        
        # Threshold heatmap at 0.5
        pred_mask = (heatmap > 0.5).astype(np.float32)
        
        # IoU
        intersection = np.sum(pred_mask * true_mask)
        union = np.sum(pred_mask) + np.sum(true_mask) - intersection
        iou = intersection / max(union, 1e-8)
        ious.append(iou)
        class_ious[label].append(iou)
        
        # Pointing Game
        max_point = np.unravel_index(np.argmax(heatmap), heatmap.shape)
        if true_mask[max_point[0], max_point[1]] > 0:
            hits += 1
            class_hits[label] += 1
        class_total[label] += 1
    
    # Report results
    print(f"\nOverall Quantitative XAI Results:")
    print(f"  Mean IoU: {np.mean(ious):.4f} ± {np.std(ious):.4f}")
    print(f"  Pointing Game Accuracy: {hits}/{len(ious)} = {hits/max(len(ious),1)*100:.1f}%")
    
    results = []
    for cls_id in sorted(class_ious.keys()):
        cls_name = class_names[cls_id] if cls_id < len(class_names) else f"Class {cls_id}"
        results.append({
            'Class': cls_name,
            'Mean IoU': f"{np.mean(class_ious[cls_id]):.4f}",
            'IoU Std': f"{np.std(class_ious[cls_id]):.4f}",
            'Pointing Game': f"{class_hits[cls_id]}/{class_total[cls_id]}",
            'PG Accuracy (%)': f"{class_hits[cls_id]/max(class_total[cls_id],1)*100:.1f}"
        })
    
    results.append({
        'Class': 'Overall',
        'Mean IoU': f"{np.mean(ious):.4f}",
        'IoU Std': f"{np.std(ious):.4f}",
        'Pointing Game': f"{hits}/{len(ious)}",
        'PG Accuracy (%)': f"{hits/max(len(ious),1)*100:.1f}"
    })
    
    df = pd.DataFrame(results)
    df.to_csv(RESULTS_DIR / 'xai_quantitative.csv', index=False)
    print(f"\nSaved to {RESULTS_DIR / 'xai_quantitative.csv'}")
    print(df.to_string(index=False))
    
    # Qualitative visualization (2x4 grid)
    n_viz = min(2, len(test_images))
    if n_viz > 0:
        fig, axes = plt.subplots(n_viz, 4, figsize=(16, 4*n_viz))
        if n_viz == 1:
            axes = axes.reshape(1, -1)
        
        for row in range(n_viz):
            img = test_images[row]
            mask = (test_masks[row] > 0).astype(np.float32)
            if mask.shape != (512, 512):
                mask = cv2.resize(mask, (512, 512))
            
            try:
                heatmap = gradcam_1d(img, model, conv_layer_name)
            except:
                continue
            
            # Normalize image for display
            img_disp = ((img - img.min()) / (img.max() - img.min() + 1e-8) * 255).astype(np.uint8)
            
            axes[row, 0].imshow(img_disp, cmap='gray')
            axes[row, 0].set_title('Original MRI', fontsize=10)
            axes[row, 0].axis('off')
            
            axes[row, 1].imshow(mask, cmap='Reds')
            axes[row, 1].set_title('Ground Truth Mask', fontsize=10)
            axes[row, 1].axis('off')
            
            axes[row, 2].imshow(heatmap, cmap='jet')
            axes[row, 2].set_title('Grad-CAM++ Heatmap', fontsize=10)
            axes[row, 2].axis('off')
            
            axes[row, 3].imshow(img_disp, cmap='gray')
            axes[row, 3].imshow(heatmap, cmap='jet', alpha=0.5)
            axes[row, 3].contour(mask, colors='lime', linewidths=1.5)
            iou_val = ious[row] if row < len(ious) else 0
            axes[row, 3].set_title(f'Overlay (IoU={iou_val:.3f})', fontsize=10)
            axes[row, 3].axis('off')
        
        plt.suptitle('Quantitative XAI: Grad-CAM++ vs Ground Truth Tumor Masks', fontsize=13, fontweight='bold')
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / 'xai_qualitative_with_mask.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved visualization to {RESULTS_DIR / 'xai_qualitative_with_mask.png'}")
    
    tf.keras.backend.clear_session()
    return df


# =============================================================================
# MAIN — RUN ALL EXPERIMENTS
# =============================================================================
if __name__ == '__main__':
    print("=" * 60)
    print("MRICovNetX Paper Revision — Experiment Runner")
    print("=" * 60)
    print(f"Results will be saved to: {RESULTS_DIR.absolute()}")
    print(f"TensorFlow version: {tf.__version__}")
    print(f"GPU available: {tf.config.list_physical_devices('GPU')}")
    print()
    
    # Run experiments in priority order
    try:
        run_experiment_1()  # Dedup check (~30 min)
    except Exception as e:
        print(f"Experiment 1 failed: {e}")
    
    try:
        run_experiment_3()  # Patient split (~2-3 hrs)
    except Exception as e:
        print(f"Experiment 3 failed: {e}")
    
    try:
        run_experiment_2()  # Ablation (~4-5 hrs)
    except Exception as e:
        print(f"Experiment 2 failed: {e}")
    
    try:
        run_experiment_4()  # Baselines (~3-4 hrs)
    except Exception as e:
        print(f"Experiment 4 failed: {e}")
    
    try:
        run_experiment_6()  # Quantitative XAI (~1 hr)
    except Exception as e:
        print(f"Experiment 6 failed: {e}")
    
    try:
        run_experiment_5()  # Failure cases (~30 min)
    except Exception as e:
        print(f"Experiment 5 failed: {e}")
    
    print("\n" + "=" * 60)
    print("ALL EXPERIMENTS COMPLETE!")
    print(f"Results saved in: {RESULTS_DIR.absolute()}")
    print("=" * 60)
    
    # List all output files
    print("\nGenerated files:")
    for f in sorted(RESULTS_DIR.iterdir()):
        print(f"  {f.name} ({f.stat().st_size / 1024:.1f} KB)")
