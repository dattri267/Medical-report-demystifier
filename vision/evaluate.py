"""
vision/evaluate.py

Evaluates BOTH checkpoints (baseline and fine-tuned) on the held-out TEST
set - data neither model has seen, and which had no influence on any
checkpoint-selection or hyperparameter decision made in Phases 6-7.

Produces:
    - Per-class ROC-AUC for both models
    - Macro AUC (mean across all 14 classes) for both
    - ROC curve grid plot (14 subplots, both models overlaid)
    - outputs/metrics/test_evaluation.csv

Usage:
    python -m vision.evaluate
"""

import os

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve

from vision.constants import LABELS
from vision.preprocessing import build_dataset

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = os.path.join(_THIS_DIR, "..", "models", "checkpoints")
BASELINE_PATH = os.path.join(CHECKPOINT_DIR, "baseline_best.keras")
FINETUNED_PATH = os.path.join(CHECKPOINT_DIR, "finetuned_best.keras")

OUTPUTS_DIR = os.path.join(_THIS_DIR, "..", "outputs")
METRICS_DIR = os.path.join(OUTPUTS_DIR, "metrics")
FIGURES_DIR = os.path.join(OUTPUTS_DIR, "figures")


def get_predictions(model, dataset):
    """
    Runs a model over an entire tf.data.Dataset and collects predictions
    alongside true labels, in order (dataset must NOT be shuffled - test
    set uses training=False in build_dataset, so order matches the CSV).
    """
    y_true_batches = []
    y_pred_batches = []

    for images, labels in dataset:
        preds = model.predict(images, verbose=0)
        y_true_batches.append(labels.numpy())
        y_pred_batches.append(preds)

    y_true = np.concatenate(y_true_batches, axis=0)
    y_pred = np.concatenate(y_pred_batches, axis=0)
    return y_true, y_pred


def compute_per_class_auc(y_true, y_pred):
    """
    Computes ROC-AUC independently for each of the 14 disease columns.
    Each disease is scored on its own binary yes/no task - this is the
    correct way to evaluate multi-label classification, as opposed to
    a single combined "accuracy" number that would be misleading given
    the severe class imbalance found in Phase 3.
    """
    aucs = {}
    for i, label in enumerate(LABELS):
        aucs[label] = roc_auc_score(y_true[:, i], y_pred[:, i])
    return aucs


def plot_roc_curves(y_true_base, y_pred_base, y_true_ft, y_pred_ft, save_path):
    """
    Plots a 4x4 grid of ROC curves (one per disease, 14 used, 2 blank),
    comparing baseline vs fine-tuned on the same axes for direct comparison.
    """
    fig, axes = plt.subplots(4, 4, figsize=(18, 16))
    axes = axes.flatten()

    for i, label in enumerate(LABELS):
        ax = axes[i]

        fpr_base, tpr_base, _ = roc_curve(y_true_base[:, i], y_pred_base[:, i])
        fpr_ft, tpr_ft, _ = roc_curve(y_true_ft[:, i], y_pred_ft[:, i])

        auc_base = roc_auc_score(y_true_base[:, i], y_pred_base[:, i])
        auc_ft = roc_auc_score(y_true_ft[:, i], y_pred_ft[:, i])

        ax.plot(fpr_base, tpr_base, label=f"Baseline (AUC={auc_base:.3f})", color="tab:blue")
        ax.plot(fpr_ft, tpr_ft, label=f"Fine-tuned (AUC={auc_ft:.3f})", color="tab:orange")
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=0.8)  # random-chance reference

        ax.set_title(label, fontsize=10)
        ax.set_xlabel("False Positive Rate", fontsize=8)
        ax.set_ylabel("True Positive Rate", fontsize=8)
        ax.legend(fontsize=7)

    # Hide the 2 unused subplots (14 diseases, 16 grid slots)
    for j in range(len(LABELS), len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def main():
    os.makedirs(METRICS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    print("Building test dataset...")
    test_ds = build_dataset("data/splits/test.csv", training=False)

    print(f"Loading baseline model from {BASELINE_PATH}...")
    baseline_model = tf.keras.models.load_model(BASELINE_PATH)

    print(f"Loading fine-tuned model from {FINETUNED_PATH}...")
    finetuned_model = tf.keras.models.load_model(FINETUNED_PATH)

    print("Running baseline predictions on test set...")
    y_true_base, y_pred_base = get_predictions(baseline_model, test_ds)

    print("Running fine-tuned predictions on test set...")
    # Rebuild the dataset iterator - a tf.data.Dataset is exhausted after
    # one full pass, so we need a fresh one for the second model.
    test_ds = build_dataset("data/splits/test.csv", training=False)
    y_true_ft, y_pred_ft = get_predictions(finetuned_model, test_ds)

    print("Computing per-class AUC...")
    auc_base = compute_per_class_auc(y_true_base, y_pred_base)
    auc_ft = compute_per_class_auc(y_true_ft, y_pred_ft)

    macro_auc_base = np.mean(list(auc_base.values()))
    macro_auc_ft = np.mean(list(auc_ft.values()))

    results_df = pd.DataFrame({
        "Disease": LABELS,
        "Baseline_AUC": [auc_base[l] for l in LABELS],
        "Finetuned_AUC": [auc_ft[l] for l in LABELS],
    })
    results_df["Difference"] = results_df["Finetuned_AUC"] - results_df["Baseline_AUC"]
    results_df = results_df.sort_values("Baseline_AUC", ascending=False)

    print("\n=== Per-Class Test Set AUC ===")
    print(results_df.to_string(index=False))
    print(f"\nMacro AUC - Baseline:   {macro_auc_base:.4f}")
    print(f"Macro AUC - Fine-tuned: {macro_auc_ft:.4f}")

    csv_path = os.path.join(METRICS_DIR, "test_evaluation.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"\nSaved results table to: {csv_path}")

    print("Plotting ROC curves...")
    roc_path = os.path.join(FIGURES_DIR, "roc_curves.png")
    plot_roc_curves(y_true_base, y_pred_base, y_true_ft, y_pred_ft, roc_path)
    print(f"Saved ROC curve grid to: {roc_path}")


if __name__ == "__main__":
    main()