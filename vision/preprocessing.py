"""
vision/preprocessing.py

Builds a tf.data.Dataset pipeline for chest X-ray images:
load -> grayscale-to-RGB -> CLAHE -> resize -> normalize -> (augment) -> batch

Usage (see bottom of file for a runnable test):
    from vision.preprocessing import build_dataset
    train_ds = build_dataset("data/splits/train.csv", training=True)
"""

import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf

from vision.constants import LABELS, IMAGE_SIZE

# Anchor this path to the location of THIS file, not the caller's working
# directory. Without this, the pipeline breaks depending on whether you run
# it as `python -m vision.preprocessing` (cwd = project root) or from a
# notebook inside notebooks/ (cwd = notebooks/) - exactly the bug that
# bit us here.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGES_ROOT = os.path.join(_THIS_DIR, "..", "data", "raw", "images")

BATCH_SIZE = 16  # modest default since we're on CPU - increase later if you get GPU access


def _apply_clahe(image_uint8: np.ndarray) -> np.ndarray:
    """
    Apply CLAHE to a single grayscale image (numpy array, uint8, shape [H, W]).
    This runs OUTSIDE the TensorFlow graph via tf.numpy_function, since
    OpenCV's CLAHE has no native TensorFlow equivalent.
    """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(image_uint8)


def _load_and_preprocess(filepath, label):
    """
    Loads one image from disk and applies the full preprocessing chain.
    Runs once per image, called by .map() below.
    """
    # Read raw file bytes, decode as grayscale PNG
    image_bytes = tf.io.read_file(filepath)
    image = tf.io.decode_png(image_bytes, channels=1)  # shape: [H, W, 1]

    # CLAHE requires a numpy uint8 array - wrap the OpenCV call so it can
    # run inside the tf.data graph via tf.numpy_function
    image_uint8 = tf.squeeze(image, axis=-1)  # [H, W], still uint8
    image_clahe = tf.numpy_function(func=_apply_clahe, inp=[image_uint8], Tout=tf.uint8)
    image_clahe.set_shape([None, None])  # numpy_function loses shape info - restore it

    # Grayscale -> 3-channel (replicate the single channel 3x)
    image_rgb = tf.stack([image_clahe, image_clahe, image_clahe], axis=-1)  # [H, W, 3]

    # Resize to DenseNet121's expected input size
    image_resized = tf.image.resize(image_rgb, [IMAGE_SIZE, IMAGE_SIZE])

    # DenseNet-specific normalization (matches ImageNet pretraining stats)
    image_normalized = tf.keras.applications.densenet.preprocess_input(image_resized)

    return image_normalized, label


def _augment(image, label):
    """
    Light, medically-safe augmentation for TRAINING data only.
    Horizontal flip is standard in chest X-ray literature (e.g. CheXNet).
    We deliberately avoid vertical flips and large rotations, since those
    would distort real anatomical orientation.
    """
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.1)
    image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
    return image, label


def build_dataset(csv_path: str, training: bool = False, batch_size: int = BATCH_SIZE):
    """
    Build a tf.data.Dataset from one of the split CSVs.

    Args:
        csv_path: path to train.csv / val.csv / test.csv
        training: if True, shuffles and applies augmentation
        batch_size: number of images per batch

    Returns:
        A tf.data.Dataset yielding (image_batch, label_batch) tuples.
    """
    # If a relative csv_path is given, resolve it relative to the project
    # root (same anchor trick as IMAGES_ROOT above) so this works whether
    # called from a script at the project root or a notebook in notebooks/.
    if not os.path.isabs(csv_path):
        project_root = os.path.join(_THIS_DIR, "..")
        csv_path = os.path.join(project_root, csv_path.lstrip("./").replace("../", ""))

    df = pd.read_csv(csv_path)

    filepaths = [os.path.join(IMAGES_ROOT, fname) for fname in df["Image Index"]]
    labels = df[LABELS].values.astype(np.float32)

    ds = tf.data.Dataset.from_tensor_slices((filepaths, labels))

    if training:
        ds = ds.shuffle(buffer_size=len(df), seed=42, reshuffle_each_iteration=True)

    ds = ds.map(_load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)

    if training:
        ds = ds.map(_augment, num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds


if __name__ == "__main__":
    # Quick smoke test: build the train pipeline and pull one batch through it
    print("Building train dataset...")
    train_ds = build_dataset("data/splits/train.csv", training=True)

    print("Pulling one batch...")
    for images, labels in train_ds.take(1):
        print("Image batch shape:", images.shape)
        print("Label batch shape:", labels.shape)
        print("Image value range:", float(tf.reduce_min(images)), "to", float(tf.reduce_max(images)))
        print("Sample label row:", labels[0].numpy())