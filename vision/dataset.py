"""
vision/dataset.py

Loads NIH ChestX-ray14 metadata, filters to images actually present on
disk, builds multi-label targets, and produces a PATIENT-LEVEL
train/val/test split (never splitting one patient's images across sets).

Run this from the project root:
    python vision/dataset.py
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from vision.constants import LABELS, RANDOM_SEED

# ---------------------------------------------------------------------
# Config - adjust these paths/numbers if your layout differs
# ---------------------------------------------------------------------
RAW_CSV_PATH = "data/raw/Data_Entry_2017.csv"
IMAGES_ROOT = "data/raw/images"       # where the extracted PNGs live
SPLITS_DIR = "data/splits"

TARGET_TOTAL_IMAGES = 10000            # approx dev-sample size
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
TEST_FRAC = 0.15


def load_metadata(csv_path: str) -> pd.DataFrame:
    """Load the NIH CSV and drop the stray trailing unnamed column."""
    df = pd.read_csv(csv_path)
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
    return df


def filter_to_available_images(df: pd.DataFrame, images_root: str) -> pd.DataFrame:
    """
    We only downloaded 3 of 12 NIH archives, so most rows in the CSV
    reference images we don't have on disk. Scan what actually exists
    and keep only matching rows.
    """
    available = set(os.listdir(images_root))
    before = len(df)
    df = df[df["Image Index"].isin(available)].reset_index(drop=True)
    print(f"Filtered metadata: {before} rows -> {len(df)} rows "
          f"(images actually present on disk)")
    return df


def build_multihot_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    'Finding Labels' looks like 'Cardiomegaly|Emphysema' or 'No Finding'.
    Convert this into 14 binary columns, one per LABELS entry.
    """
    for label in LABELS:
        df[label] = df["Finding Labels"].apply(
            lambda x: 1 if label in x.split("|") else 0
        )
    return df


def patient_level_split(df: pd.DataFrame):
    """
    Split by Patient ID so no patient's images appear in more than one
    of train/val/test. This is the #1 rule for avoiding patient leakage.

    Two-stage split:
      1. train vs (val+test) by patient group
      2. val vs test by patient group, within the leftover patients
    """
    groups = df["Patient ID"].values

    gss1 = GroupShuffleSplit(n_splits=1, train_size=TRAIN_FRAC,
                              random_state=RANDOM_SEED)
    train_idx, temp_idx = next(gss1.split(df, groups=groups))

    train_df = df.iloc[train_idx].reset_index(drop=True)
    temp_df = df.iloc[temp_idx].reset_index(drop=True)

    # split temp into val/test (proportion within the remainder)
    val_ratio_within_temp = VAL_FRAC / (VAL_FRAC + TEST_FRAC)
    gss2 = GroupShuffleSplit(n_splits=1, train_size=val_ratio_within_temp,
                              random_state=RANDOM_SEED)
    temp_groups = temp_df["Patient ID"].values
    val_idx, test_idx = next(gss2.split(temp_df, groups=temp_groups))

    val_df = temp_df.iloc[val_idx].reset_index(drop=True)
    test_df = temp_df.iloc[test_idx].reset_index(drop=True)

    # Sanity check: confirm zero patient overlap across splits
    train_patients = set(train_df["Patient ID"])
    val_patients = set(val_df["Patient ID"])
    test_patients = set(test_df["Patient ID"])
    assert not (train_patients & val_patients), "Patient leakage: train/val overlap!"
    assert not (train_patients & test_patients), "Patient leakage: train/test overlap!"
    assert not (val_patients & test_patients), "Patient leakage: val/test overlap!"

    return train_df, val_df, test_df


def subsample_by_patient(df: pd.DataFrame, target_n: int, seed: int) -> pd.DataFrame:
    """
    Reduce a split down toward target_n images WITHOUT breaking up any
    patient's images. We shuffle patients, then keep adding whole
    patients until we reach (or just pass) target_n images.
    """
    if len(df) <= target_n:
        return df  # nothing to trim

    rng = np.random.RandomState(seed)
    patient_ids = df["Patient ID"].unique()
    rng.shuffle(patient_ids)

    kept_patients = []
    running_total = 0
    for pid in patient_ids:
        if running_total >= target_n:
            break
        n_images_for_patient = (df["Patient ID"] == pid).sum()
        kept_patients.append(pid)
        running_total += n_images_for_patient

    return df[df["Patient ID"].isin(kept_patients)].reset_index(drop=True)


def main():
    print("Loading metadata...")
    df = load_metadata(RAW_CSV_PATH)

    print("Filtering to images present on disk...")
    df = filter_to_available_images(df, IMAGES_ROOT)

    print("Building multi-hot label columns...")
    df = build_multihot_labels(df)

    print("Splitting by patient (train/val/test)...")
    train_df, val_df, test_df = patient_level_split(df)
    print(f"  Before subsampling -> train: {len(train_df)}, "
          f"val: {len(val_df)}, test: {len(test_df)}")

    print(f"Subsampling toward ~{TARGET_TOTAL_IMAGES} total images...")
    train_target = int(TARGET_TOTAL_IMAGES * TRAIN_FRAC)
    val_target = int(TARGET_TOTAL_IMAGES * VAL_FRAC)
    test_target = int(TARGET_TOTAL_IMAGES * TEST_FRAC)

    train_df = subsample_by_patient(train_df, train_target, RANDOM_SEED)
    val_df = subsample_by_patient(val_df, val_target, RANDOM_SEED + 1)
    test_df = subsample_by_patient(test_df, test_target, RANDOM_SEED + 2)

    print(f"  After subsampling  -> train: {len(train_df)}, "
          f"val: {len(val_df)}, test: {len(test_df)}")
    print(f"  Total images: {len(train_df) + len(val_df) + len(test_df)}")

    os.makedirs(SPLITS_DIR, exist_ok=True)
    train_df.to_csv(os.path.join(SPLITS_DIR, "train.csv"), index=False)
    val_df.to_csv(os.path.join(SPLITS_DIR, "val.csv"), index=False)
    test_df.to_csv(os.path.join(SPLITS_DIR, "test.csv"), index=False)
    print(f"Saved train.csv, val.csv, test.csv to {SPLITS_DIR}/")

    # Quick class balance snapshot so you can eyeball it before Phase 3's
    # full EDA notebook.
    print("\nPositive-sample counts per label (train set):")
    print(train_df[LABELS].sum().sort_values(ascending=False))


if __name__ == "__main__":
    main()