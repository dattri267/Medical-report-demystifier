"""
vision/calibration.py

Implements temperature scaling and Expected Calibration Error (ECE) for
the baseline model.

IMPORTANT CAVEAT: calibration makes the model's predicted PROBABILITIES
more trustworthy as probabilities (e.g. "70% confident" predictions are
actually right about 70% of the time). It does NOT make the underlying
model more medically accurate, and is not a form of clinical validation.

Usage:
    python -m vision.calibration
"""

import os

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

from vision.constants import LABELS
from vision.preprocessing import build_dataset

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_PATH = os.path.join(_THIS_DIR, "..", "models", "checkpoints", "baseline_best.keras")
OUTPUTS_DIR = os.path.join(_THIS_DIR, "..", "outputs")
METRICS_DIR = os.path.join(OUTPUTS_DIR, "metrics")
FIGURES_DIR = os.path.join(OUTPUTS_DIR, "figures")


def get_predictions(model, dataset):
    y_true_batches = []
    y_pred_batches = []
    for images, labels in dataset:
        preds = model.predict(images, verbose=0)
        y_true_batches.append(labels.numpy())
        y_pred_batches.append(preds)
    y_true = np.concatenate(y_true_batches, axis=0)
    y_pred = np.concatenate(y_pred_batches, axis=0)
    return y_true, y_pred


def prob_to_logit(p, eps=1e-7):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def apply_temperature(logits, T):
    return 1 / (1 + np.exp(-logits / T))


def fit_temperature(val_logits, val_labels):
    """
    Finds the single scalar T that minimizes binary cross-entropy on the
    VALIDATION set (never the test set - that would leak test information
    into a modeling decision, the same mistake we avoided in Phase 6-7).
    """
    def nll(T):
        calibrated = apply_temperature(val_logits, T)
        calibrated = np.clip(calibrated, 1e-7, 1 - 1e-7)
        loss = -np.mean(val_labels * np.log(calibrated) + (1 - val_labels) * np.log(1 - calibrated))
        return loss

    result = minimize_scalar(nll, bounds=(0.05, 10.0), method="bounded")
    return result.x


def compute_ece(y_true, y_pred, n_bins=10):
    """
    Expected Calibration Error: bins predictions by confidence, checks
    whether actual accuracy within each bin matches the average predicted
    confidence in that bin. Flattens across all 14 classes and all samples
    - each (sample, class) pair is treated as one binary prediction.
    """
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()

    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    bin_confidences = []
    bin_accuracies = []
    bin_counts = []

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        in_bin = (y_pred_flat >= lo) & (y_pred_flat < hi) if i < n_bins - 1 else (y_pred_flat >= lo) & (y_pred_flat <= hi)
        count = in_bin.sum()
        if count == 0:
            bin_confidences.append(np.nan)
            bin_accuracies.append(np.nan)
            bin_counts.append(0)
            continue

        avg_confidence = y_pred_flat[in_bin].mean()
        avg_accuracy = y_true_flat[in_bin].mean()  # fraction actually positive in this bin

        bin_confidences.append(avg_confidence)
        bin_accuracies.append(avg_accuracy)
        bin_counts.append(count)

        ece += (count / len(y_pred_flat)) * abs(avg_confidence - avg_accuracy)

    return ece, bin_confidences, bin_accuracies, bin_counts


def plot_reliability_diagram(bin_conf_before, bin_acc_before, bin_conf_after, bin_acc_after, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, bin_conf, bin_acc, title in [
        (axes[0], bin_conf_before, bin_acc_before, "Before Calibration"),
        (axes[1], bin_conf_after, bin_acc_after, "After Calibration"),
    ]:
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
        valid = ~np.isnan(bin_conf)
        ax.plot(np.array(bin_conf)[valid], np.array(bin_acc)[valid], marker="o", color="tab:blue")
        ax.set_xlabel("Predicted confidence")
        ax.set_ylabel("Actual frequency of positive")
        ax.set_title(title)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def main():
    os.makedirs(METRICS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    print(f"Loading baseline model from {CHECKPOINT_PATH}...")
    model = tf.keras.models.load_model(CHECKPOINT_PATH)

    print("Getting validation set predictions (to FIT temperature)...")
    val_ds = build_dataset("data/splits/val.csv", training=False)
    val_true, val_pred = get_predictions(model, val_ds)
    val_logits = prob_to_logit(val_pred)

    print("Fitting temperature scaling parameter T on validation set...")
    T = fit_temperature(val_logits, val_true)
    print(f"Fitted T = {T:.4f}")
    if T > 1:
        print("(T > 1 means the model was OVER-confident before calibration - "
              "calibration will pull probabilities toward 0.5)")
    else:
        print("(T < 1 means the model was UNDER-confident before calibration - "
              "calibration will push probabilities toward the extremes)")

    print("\nGetting test set predictions (for HONEST, held-out evaluation)...")
    test_ds = build_dataset("data/splits/test.csv", training=False)
    test_true, test_pred_before = get_predictions(model, test_ds)
    test_logits = prob_to_logit(test_pred_before)
    test_pred_after = apply_temperature(test_logits, T)

    print("Computing ECE before and after calibration (on test set)...")
    ece_before, bin_conf_before, bin_acc_before, _ = compute_ece(test_true, test_pred_before)
    ece_after, bin_conf_after, bin_acc_after, _ = compute_ece(test_true, test_pred_after)

    print(f"\nECE before calibration: {ece_before:.4f}")
    print(f"ECE after calibration:  {ece_after:.4f}")

    results = {
        "fitted_temperature": float(T),
        "ece_before_calibration": float(ece_before),
        "ece_after_calibration": float(ece_after),
        "note": "Calibration improves probability trustworthiness. It does NOT "
                "improve or validate medical accuracy - AUC/ranking performance "
                "is unchanged by temperature scaling, since it's a monotonic "
                "transformation applied per-class.",
    }
    results_df = pd.DataFrame([results])
    results_df.to_csv(os.path.join(METRICS_DIR, "calibration_results.csv"), index=False)
    print(f"\nSaved calibration results to: {METRICS_DIR}/calibration_results.csv")

    print("Plotting reliability diagram...")
    reliability_path = os.path.join(FIGURES_DIR, "reliability_diagram.png")
    plot_reliability_diagram(bin_conf_before, bin_acc_before, bin_conf_after, bin_acc_after, reliability_path)
    print(f"Saved reliability diagram to: {reliability_path}")


if __name__ == "__main__":
    main()
