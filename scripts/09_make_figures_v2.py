"""
=============================================================================
MRICovNetX - Experiment 9: Figures for the definitive (leakage-free) results
=============================================================================
Builds every CovNet22 figure used in the revised manuscript directly from the
artefacts written by 07_definitive_covnet22_eval.py (no simulated data).

    python 09_make_figures_v2.py --out <overleaf>/figs/v2
=============================================================================
"""
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
import argparse
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.manifold import TSNE

R = Path(os.environ.get("MRI_OUT_ROOT", r"C:\MRIData\results_v2"))
DATA = Path(os.environ.get("MRI_DATA_ROOT", r"C:\MRIData"))
SEEDS = [42, 123, 2024]
REP = 42  # representative seed for single-run figures

DS = {  # display name -> (result folder, labels, data folder for test images)
    "Dataset-2": ("Dataset2dedup", ["Glioma", "Meningioma", "No tumor", "Pituitary"], DATA / "Dataset_2_Kaggle"),
    "Dataset-3": ("Dataset3", ["Glioma", "Meningioma", "No tumor", "Pituitary"], DATA / "Dataset_3_Nickparvar" / "Testing"),
    "Dataset-4": ("Dataset4", ["Brain_Glioma", "Brain_Menin", "Brain Tumor"], DATA / "Dataset_4_Mendeley" / "Brain_Cancer"),
}
COL = {"Dataset-1": "#4C72B0", "Dataset-2": "#55A868", "Dataset-3": "#DD8452", "Dataset-4": "#8172B3"}
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.dpi": 300, "savefig.bbox": "tight", "font.family": "DejaVu Sans"})


def run_dir(ds, seed=REP, variant="full"):
    return R / DS[ds][0] / f"{variant}_seed{seed}"


def per_class_mean(ds):
    frames = [pd.read_csv(run_dir(ds, s) / "per_class.csv") for s in SEEDS]
    df = pd.concat(frames)
    return df.groupby("cls", sort=False).agg(["mean", "std"])


def fig_distributions(out):
    for ds, (_, labels, _) in DS.items():
        y = np.load(run_dir(ds) / "test_labels.npy")
        cnt = np.bincount(y, minlength=len(labels))
        fig, ax = plt.subplots(figsize=(4.2, 3.2))
        bars = ax.bar(labels, cnt, color=plt.cm.Set2(np.arange(len(labels))))
        for b, c in zip(bars, cnt):
            ax.text(b.get_x() + b.get_width() / 2, c, f"{c}\n({c / cnt.sum() * 100:.1f}%)",
                    ha="center", va="bottom", fontsize=8)
        ax.set_ylabel("Test images")
        ax.set_title(f"{ds} test set (n = {cnt.sum():,})", fontsize=10)
        ax.set_ylim(0, cnt.max() * 1.25)
        plt.xticks(rotation=15)
        fig.savefig(out / f"dist_{ds[-1]}.pdf"); plt.close(fig)


def fig_training_curves(out):
    for ds in DS:
        h = pd.read_csv(run_dir(ds) / "history.csv")
        best = int(json.loads((run_dir(ds) / "result.json").read_text())["best_epoch"])
        fig, ax = plt.subplots(1, 2, figsize=(8.5, 3.0))
        ax[0].plot(h.epoch, h.train_acc, label="Training"); ax[0].plot(h.epoch, h.val_acc, label="Validation")
        ax[1].plot(h.epoch, h.train_loss, label="Training"); ax[1].plot(h.epoch, h.val_loss, label="Validation")
        for a, t in zip(ax, ["Accuracy (%)", "Cross-entropy loss"]):
            a.axvline(best, ls="--", c="grey", lw=0.8)
            a.set_xlabel("Epoch"); a.set_ylabel(t); a.legend(frameon=False)
        ax[0].set_title(f"{ds}: accuracy (best epoch {best})", fontsize=10)
        ax[1].set_title(f"{ds}: loss", fontsize=10)
        fig.savefig(out / f"train_curves_{ds[-1]}.pdf"); plt.close(fig)
    # validation summary
    fig, ax = plt.subplots(figsize=(6, 3.2))
    for ds in DS:
        h = pd.read_csv(run_dir(ds) / "history.csv")
        ax.plot(h.epoch, h.val_acc, label=ds, c=COL[ds])
    ax.set_xlabel("Epoch"); ax.set_ylabel("Validation accuracy (%)")
    ax.set_title("CovNet22 validation accuracy (seed 42)", fontsize=10); ax.legend(frameon=False)
    fig.savefig(out / "val_acc_summary.pdf"); plt.close(fig)


def fig_confusion_and_roc(out):
    for ds, (_, labels, _) in DS.items():
        cm = np.loadtxt(run_dir(ds) / "confusion_matrix.csv", delimiter=",").astype(int)
        fig, ax = plt.subplots(figsize=(4.2, 3.8))
        im = ax.imshow(cm, cmap="Blues")
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(j, i, cm[i, j], ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=10)
        ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        acc = np.trace(cm) / cm.sum() * 100
        ax.set_title(f"{ds} (seed 42, n = {cm.sum():,}, acc = {acc:.2f}%)", fontsize=9)
        fig.colorbar(im, fraction=0.046)
        fig.savefig(out / f"cm_{ds[-1]}.pdf"); plt.close(fig)

        roc = json.loads((run_dir(ds) / "roc.json").read_text())
        fig, ax = plt.subplots(figsize=(4.2, 3.8))
        for lab, d in roc.items():
            ax.plot(d["fpr"], d["tpr"], label=f"{lab} (AUC = {d['auc']:.4f})")
        ax.plot([0, 1], [0, 1], "k--", lw=0.7)
        ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
        ax.set_title(f"{ds} one-vs-rest ROC (seed 42)", fontsize=9)
        ax.legend(frameon=False, fontsize=8, loc="lower right")
        # inset zoom
        ins = ax.inset_axes([0.35, 0.35, 0.35, 0.3])
        for lab, d in roc.items():
            ins.plot(d["fpr"], d["tpr"])
        ins.set_xlim(0, 0.1); ins.set_ylim(0.9, 1.0); ins.tick_params(labelsize=6)
        fig.savefig(out / f"roc_{ds[-1]}.pdf"); plt.close(fig)


def fig_tsne(out):
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.8))
    for a, (ds, (_, labels, _)) in zip(ax, DS.items()):
        f = np.load(run_dir(ds) / "test_features.npy")
        y = np.load(run_dir(ds) / "test_labels.npy")
        z = TSNE(n_components=2, perplexity=30, init="pca", random_state=0).fit_transform(f)
        for i, lab in enumerate(labels):
            a.scatter(z[y == i, 0], z[y == i, 1], s=5, alpha=0.7, label=lab)
        a.set_title(f"{ds} (n = {len(y):,} test images)", fontsize=10)
        a.set_xticks([]); a.set_yticks([])
        a.legend(frameon=False, fontsize=7, markerscale=2)
    fig.savefig(out / "tsne_d2_d4.pdf"); plt.close(fig)


def fig_classwise(out):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4), sharey=True)
    for a, ds in zip(axes, DS):
        pc = per_class_mean(ds)
        x = np.arange(len(pc))
        for k, (m, c) in enumerate([("precision", "#4C72B0"), ("recall", "#55A868"), ("F1", "#C44E52")]):
            a.bar(x + (k - 1) * 0.27, pc[(m, "mean")], 0.27, yerr=pc[(m, "std")], capsize=2,
                  label=m.capitalize() if m != "F1" else "F1-score", color=c)
        a.set_xticks(x); a.set_xticklabels(pc.index, rotation=15)
        a.set_title(ds, fontsize=10); a.set_ylim(0.8 if ds != "Dataset-2" else 0.7, 1.005)
    axes[0].set_ylabel("Score (mean ± SD, 3 seeds)")
    axes[0].legend(frameon=False, fontsize=8, loc="lower left")
    fig.savefig(out / "classwise_metrics.pdf"); plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4), sharey=True)
    for a, ds in zip(axes, DS):
        pc = per_class_mean(ds)
        x = np.arange(len(pc))
        a.bar(x - 0.2, pc[("recall", "mean")], 0.4, yerr=pc[("recall", "std")], capsize=2,
              label="Sensitivity", color="#4C72B0")
        a.bar(x + 0.2, pc[("TNR", "mean")], 0.4, yerr=pc[("TNR", "std")], capsize=2,
              label="Specificity", color="#DD8452")
        a.set_xticks(x); a.set_xticklabels(pc.index, rotation=15)
        a.set_title(ds, fontsize=10); a.set_ylim(0.85, 1.005)
    axes[0].set_ylabel("Mean ± SD (3 seeds)"); axes[0].legend(frameon=False, fontsize=8, loc="lower left")
    fig.savefig(out / "sens_spec.pdf"); plt.close(fig)


def fig_summary(out, d1):
    main = pd.read_csv(R / "covnet22_definitive_per_seed.csv")
    rows = [("Dataset-1\n(CovBI-GRU, patient-level)", d1["acc"], d1["prec"], d1["rec"], d1["f1"], np.nan)]
    name_map = {"Dataset-2-dedup": "Dataset-2\n(CovNet22, de-duplicated)",
                "Dataset-2": "Dataset-2\n(CovNet22, canonical split)",
                "Dataset-3": "Dataset-3\n(CovNet22)", "Dataset-4": "Dataset-4\n(CovNet22)"}
    for k in ["Dataset-2-dedup", "Dataset-2", "Dataset-3", "Dataset-4"]:
        g = main[main.dataset == k]
        rows.append((name_map[k], g.accuracy.mean(), g.precision.mean(), g.recall.mean(), g.f1.mean(),
                     g.auc_macro.mean() * 100))
    df = pd.DataFrame(rows, columns=["", "Accuracy", "Precision", "Recall", "F1-score", "AUC×100"]).set_index("")
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    im = ax.imshow(df.values, cmap="RdYlGn", vmin=70, vmax=100, aspect="auto")
    for i in range(df.shape[0]):
        for j in range(df.shape[1]):
            v = df.values[i, j]
            ax.text(j, i, "n/a" if np.isnan(v) else f"{v:.2f}", ha="center", va="center", fontsize=9)
    ax.set_xticks(range(df.shape[1])); ax.set_xticklabels(df.columns)
    ax.set_yticks(range(df.shape[0])); ax.set_yticklabels(df.index, fontsize=8)
    ax.set_title("Mean test performance over three seeds (%)", fontsize=10)
    fig.colorbar(im, fraction=0.03)
    fig.savefig(out / "summary_heatmap.pdf"); plt.close(fig)

    # seed variability (real per-seed values; D1 shown as mean ± SD)
    fig, ax = plt.subplots(figsize=(7, 3.4))
    labels = ["D1\n(patient)", "D2\n(dedup)", "D2\n(canonical)", "D3", "D4"]
    keys = [None, "Dataset-2-dedup", "Dataset-2", "Dataset-3", "Dataset-4"]
    cols = [COL["Dataset-1"], COL["Dataset-2"], "#A6D8A8", COL["Dataset-3"], COL["Dataset-4"]]
    for i, (k, c) in enumerate(zip(keys, cols)):
        if k is None:
            m, s = d1["acc"], d1["acc_sd"]
        else:
            v = main[main.dataset == k].accuracy.values
            m, s = v.mean(), v.std(ddof=1)
            ax.scatter(np.full(len(v), i) + np.array([-0.08, 0, 0.08]), v, c="k", s=12, zorder=3)
        ax.bar(i, m, 0.6, color=c, alpha=0.8)
        ax.errorbar(i, m, yerr=s, c="k", capsize=4)
        ax.text(i, m + s + 0.6, f"{m:.2f}±{s:.2f}", ha="center", fontsize=8)
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels)
    ax.set_ylim(70, 102); ax.set_ylabel("Test accuracy (%)")
    ax.set_title("Seed-to-seed variability (dots = individual seeds)", fontsize=10)
    fig.savefig(out / "seed_variability.pdf"); plt.close(fig)

    # radar
    cats = ["Accuracy", "Precision", "Recall", "F1-score"]
    ang = np.linspace(0, 2 * np.pi, len(cats), endpoint=False).tolist(); ang += ang[:1]
    fig = plt.figure(figsize=(4.6, 4.6)); ax = fig.add_subplot(111, polar=True)
    for (name, *vals), c in zip([r for r in rows if "canonical" not in r[0]],
                                [COL["Dataset-1"], COL["Dataset-2"], COL["Dataset-3"], COL["Dataset-4"]]):
        v = list(vals[:4]) + [vals[0]]
        ax.plot(ang, v, c=c, label=name.replace("\n", " ")); ax.fill(ang, v, c=c, alpha=0.08)
    ax.set_xticks(ang[:-1]); ax.set_xticklabels(cats); ax.set_ylim(75, 100)
    ax.legend(frameon=False, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.06))
    fig.savefig(out / "performance_radar.pdf"); plt.close(fig)


def fig_ablation(out):
    main = pd.read_csv(R / "covnet22_definitive_per_seed.csv")
    abl = pd.read_csv(R / "covnet22_ablation_d3_per_seed.csv")
    full = main[main.dataset == "Dataset-3"].assign(variant="full")
    df = pd.concat([full, abl])
    order = [("full", "Full CovNet22"), ("no_aug", "w/o augmentation"), ("no_bn", "w/o batch norm."),
             ("shallow", "Reduced depth"), ("no_dropout", "w/o dropout")]
    g = df.groupby("variant")
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    x = np.arange(len(order))
    for k, (m, c) in enumerate([("accuracy", "#4C72B0"), ("precision", "#55A868"),
                                ("recall", "#DD8452"), ("f1", "#C44E52")]):
        mu = [g.get_group(v)[m].mean() for v, _ in order]
        sd = [g.get_group(v)[m].std(ddof=1) for v, _ in order]
        ax.bar(x + (k - 1.5) * 0.2, mu, 0.2, yerr=sd, capsize=2, color=c,
               label={"f1": "F1-score"}.get(m, m.capitalize()))
    base = g.get_group("full").accuracy.mean()
    for i, (v, _) in enumerate(order[1:], 1):
        d = g.get_group(v).accuracy.mean() - base
        ax.text(i, 100.3, f"{d:+.2f}", ha="center", fontsize=8, color="#C44E52")
    ax.set_xticks(x); ax.set_xticklabels([n for _, n in order])
    ax.set_ylim(94, 100.8); ax.set_ylabel("Score (%) — mean ± SD, 3 seeds")
    ax.legend(frameon=False, fontsize=8, ncol=4, loc="lower center")
    ax.set_title("CovNet22 ablation on Dataset-3 (Δ accuracy vs. full model in red)", fontsize=10)
    fig.savefig(out / "ablation_d3.pdf"); plt.close(fig)


def fig_failures(out):
    mis = pd.read_csv(run_dir("Dataset-3") / "misclassified.csv")
    n = len(mis)
    cols = min(n, 5); rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(2.4 * cols, 2.7 * rows))
    axes = np.atleast_1d(axes).ravel()
    for a in axes:
        a.axis("off")
    for a, (_, r) in zip(axes, mis.iterrows()):
        im = Image.open(DS["Dataset-3"][2] / r.file).convert("L")
        a.imshow(im, cmap="gray")
        a.set_title(f"{Path(r.file).name}\nTrue: {r.true}\nPred: {r.pred} ({r.pred_conf:.1f}%)",
                    fontsize=7, color="#B00020")
    fig.savefig(out / "failure_cases_d3_v2.png"); plt.close(fig)


def fig_confidence(out):
    labels = DS["Dataset-3"][1]
    p = np.load(run_dir("Dataset-3") / "test_probs.npy"); y = np.load(run_dir("Dataset-3") / "test_labels.npy")
    pred = p.argmax(1)
    fig, ax = plt.subplots(figsize=(6, 3.2))
    data = [p[(y == i) & (pred == i), i] * 100 for i in range(len(labels))]
    ax.boxplot(data, labels=labels, showfliers=True, flierprops=dict(markersize=2))
    ax.set_ylabel("Predicted-class confidence (%)")
    ax.set_title("Dataset-3: confidence of correct predictions (seed 42)", fontsize=10)
    fig.savefig(out / "confidence_d3.pdf"); plt.close(fig)


def fig_comparison(out):
    main = pd.read_csv(R / "covnet22_definitive_per_seed.csv")
    lit = [("Abiwinanda et al.", 84.1), ("Afshar et al.", 90.8), ("Cheng et al.", 91.28), ("Shanaka et al.", 92.0),
           ("Emrah et al.", 92.6), ("Munira A. H. (D2)", 93.5), ("Kang & Ullah (D2)", 93.72),
           ("Anaraki et al.", 94.2), ("Sajjad et al.", 94.58), ("Swati et al.", 94.82), ("Momina et al.", 95.9),
           ("Sultan et al.", 96.13), ("Islam et al.", 97.8)]
    ours = [("Ours D1 (image-level)", 98.03), ("Ours D1 (patient-level)", 82.73)]
    for k, nm in [("Dataset-2-dedup", "Ours D2 (de-duplicated)"), ("Dataset-3", "Ours D3"), ("Dataset-4", "Ours D4")]:
        ours.append((nm, main[main.dataset == k].accuracy.mean()))
    allr = lit + ours
    fig, ax = plt.subplots(figsize=(7, 6))
    y = np.arange(len(allr))
    ax.barh(y, [v for _, v in allr], color=["#B0B0B0"] * len(lit) + ["#DD8452"] * len(ours))
    for i, (_, v) in enumerate(allr):
        ax.text(v + 0.3, i, f"{v:.2f}", va="center", fontsize=8)
    ax.set_yticks(y); ax.set_yticklabels([n for n, _ in allr], fontsize=8)
    ax.set_xlim(75, 102); ax.set_xlabel("Reported accuracy (%)")
    ax.set_title("Comparison with published results (protocols differ across studies)", fontsize=10)
    fig.savefig(out / "comparison_literature.pdf"); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    # acc_sd = sample SD (ddof=1). covbigru_ablation_3seeds.csv stores the population SD (0.51);
    # on the 579-slice test set the per-seed accuracies are uniquely constrained to give a
    # sample SD of 0.62 (see results/v2/covbigru_ablation_3seeds_sampleSD.csv).
    ap.add_argument("--d1", default="82.73,0.62,81.26,84.77,82.35",
                    help="Dataset-1 patient-level acc,acc_sd(ddof=1),prec,rec,f1")
    ap.add_argument("--only-summary", action="store_true")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    v = [float(t) for t in a.d1.split(",")]
    d1 = dict(acc=v[0], acc_sd=v[1], prec=v[2], rec=v[3], f1=v[4])
    fns = [] if a.only_summary else [fig_distributions, fig_training_curves, fig_confusion_and_roc, fig_tsne,
                                     fig_classwise, fig_ablation, fig_failures, fig_confidence, fig_comparison]
    for fn in fns:
        fn(out); print("done", fn.__name__)
    fig_summary(out, d1); print("done fig_summary")


if __name__ == "__main__":
    main()
