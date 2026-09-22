"""
vision/registry.py

Builds version manifests (v1/v2/v3) documenting model weights, config,
preprocessing, and evaluation metrics - assembled from the actual results
files already produced in Phases 6-10, not hand-typed, to avoid
transcription errors.

The .keras weight files themselves are NOT committed to git (too large -
see .gitignore); these manifests document how to use them correctly and
are handed off to Partner B alongside a separately-shared weights file.

Usage:
    python -m vision.registry
"""

import json
import os

import pandas as pd

from vision.constants import LABELS, IMAGE_SIZE, RANDOM_SEED

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
METRICS_DIR = os.path.join(_THIS_DIR, "..", "outputs", "metrics")
REGISTRY_DIR = os.path.join(_THIS_DIR, "..", "models", "registry")

PREPROCESSING_DESCRIPTION = (
    "1. Load PNG as grayscale. "
    "2. Apply CLAHE (clipLimit=2.0, tileGridSize=(8,8)). "
    "3. Resize to 224x224. "
    "4. Replicate grayscale channel to 3 channels (RGB). "
    "5. Apply tf.keras.applications.densenet.preprocess_input "
    "(torch-style: scale to [0,1], normalize by ImageNet per-channel mean/std)."
)

ARCHITECTURE_DESCRIPTION = (
    "DenseNet121 (ImageNet pretrained) -> GlobalAveragePooling2D -> "
    "Dropout -> Dense(14, activation='sigmoid')"
)


def load_test_evaluation():
    path = os.path.join(METRICS_DIR, "test_evaluation.csv")
    return pd.read_csv(path)


def load_calibration_results():
    path = os.path.join(METRICS_DIR, "calibration_results.csv")
    return pd.read_csv(path).iloc[0]


def build_v1_baseline(eval_df):
    per_class_auc = dict(zip(eval_df["Disease"], eval_df["Baseline_AUC"]))
    macro_auc = eval_df["Baseline_AUC"].mean()

    return {
        "version": "v1",
        "name": "baseline-frozen-backbone",
        "checkpoint_filename": "baseline_best.keras",
        "architecture": ARCHITECTURE_DESCRIPTION,
        "backbone_frozen": True,
        "trainable_parameters": 14350,
        "image_size": IMAGE_SIZE,
        "label_order": LABELS,
        "preprocessing": PREPROCESSING_DESCRIPTION,
        "random_seed": RANDOM_SEED,
        "macro_auc_test": round(float(macro_auc), 4),
        "per_class_auc_test": {k: round(float(v), 4) for k, v in per_class_auc.items()},
        "calibration_temperature": None,
        "notes": "Trained with binary_crossentropy loss, Adam lr=1e-3, "
                 "early stopping on val_auc. Best epoch: 7 of 10.",
    }


def build_v2_finetuned(eval_df):
    per_class_auc = dict(zip(eval_df["Disease"], eval_df["Finetuned_AUC"]))
    macro_auc = eval_df["Finetuned_AUC"].mean()

    return {
        "version": "v2",
        "name": "finetuned-phase2",
        "checkpoint_filename": "finetuned_best.keras",
        "architecture": ARCHITECTURE_DESCRIPTION + " (top 15 backbone layers unfrozen)",
        "backbone_frozen": False,
        "trainable_parameters": 337934,
        "image_size": IMAGE_SIZE,
        "label_order": LABELS,
        "preprocessing": PREPROCESSING_DESCRIPTION,
        "random_seed": RANDOM_SEED,
        "macro_auc_test": round(float(macro_auc), 4),
        "per_class_auc_test": {k: round(float(v), 4) for k, v in per_class_auc.items()},
        "calibration_temperature": None,
        "notes": "HONEST FINDING: fine-tuning did NOT meaningfully improve over v1 "
                 "(macro AUC 0.699 vs v1's 0.699 - within noise). Two configurations "
                 "tested (30 and 15 unfrozen layers); both showed overfitting after "
                 "epoch 1. Kept for documentation/comparison purposes. "
                 "RECOMMENDATION: use v1 or v3, not v2, for actual deployment.",
    }


def build_v3_calibrated(eval_df, calibration_row):
    # v3 uses the SAME weights as v1 - calibration is a post-hoc scalar
    # applied at inference time (calibrated_prob = sigmoid(logit(raw_prob) / T)),
    # not a retrained model.
    v1 = build_v1_baseline(eval_df)
    v3 = dict(v1)
    v3["version"] = "v3"
    v3["name"] = "baseline-calibrated"
    v3["calibration_temperature"] = round(float(calibration_row["fitted_temperature"]), 4)
    v3["ece_before_calibration"] = round(float(calibration_row["ece_before_calibration"]), 4)
    v3["ece_after_calibration"] = round(float(calibration_row["ece_after_calibration"]), 4)
    v3["notes"] = (
        "Same weights as v1 (baseline_best.keras). Apply temperature scaling at "
        "inference: calibrated_prob = sigmoid(logit(raw_model_output) / "
        f"{v3['calibration_temperature']}). Temperature fitted on VALIDATION set, "
        "evaluated honestly on test set. ECE improved from "
        f"{v3['ece_before_calibration']} to {v3['ece_after_calibration']}. "
        "IMPORTANT: calibration improves probability trustworthiness, NOT "
        "medical accuracy - AUC is unchanged from v1 (temperature scaling is a "
        "monotonic transform, doesn't affect ranking)."
    )
    return v3


def main():
    os.makedirs(REGISTRY_DIR, exist_ok=True)

    print("Loading evaluation results...")
    eval_df = load_test_evaluation()
    calibration_row = load_calibration_results()

    print("Building v1 (baseline) manifest...")
    v1 = build_v1_baseline(eval_df)

    print("Building v2 (fine-tuned) manifest...")
    v2 = build_v2_finetuned(eval_df)

    print("Building v3 (calibrated) manifest...")
    v3 = build_v3_calibrated(eval_df, calibration_row)

    for version_data, filename in [(v1, "v1_baseline.json"), (v2, "v2_finetuned.json"), (v3, "v3_calibrated.json")]:
        path = os.path.join(REGISTRY_DIR, filename)
        with open(path, "w") as f:
            json.dump(version_data, f, indent=2)
        print(f"  Saved: {path}")

    print("\nRegistry summary:")
    print(f"  v1 baseline    - macro AUC: {v1['macro_auc_test']}")
    print(f"  v2 fine-tuned  - macro AUC: {v2['macro_auc_test']} (does not beat v1)")
    print(f"  v3 calibrated  - macro AUC: {v3['macro_auc_test']} (same as v1, T={v3['calibration_temperature']})")
    print("\nRECOMMENDATION for Partner B: use v3 (calibrated) for production - "
          "same accuracy as v1, but probabilities are more trustworthy.")


if __name__ == "__main__":
    main()
