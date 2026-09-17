"""
vision/train.py

Trains the DenseNet121 baseline (frozen backbone) on the chest X-ray
dataset, with:
    - W&B logging (loss, AUC, accuracy curves)
    - Checkpointing (saves the best model by validation AUC)
    - Early stopping (stops if validation AUC stops improving)
    - Learning rate reduction on plateau

Usage:
    python -m vision.train              # full training run
    python -m vision.train --quick      # fast sanity check (few batches, 1 epoch)
"""

import argparse
import os

import tensorflow as tf
import wandb
from wandb.integration.keras import WandbMetricsLogger

from vision.constants import RANDOM_SEED
from vision.model import build_model, compile_model
from vision.preprocessing import build_dataset

tf.random.set_seed(RANDOM_SEED)

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = os.path.join(_THIS_DIR, "..", "models", "checkpoints")
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "baseline_best.keras")

EPOCHS = 10
LEARNING_RATE = 1e-3
EARLY_STOPPING_PATIENCE = 3


def main(quick: bool = False):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    run_name = "quick-sanity-check" if quick else "baseline-frozen-backbone"
    wandb.init(
        project="medical-report-demystifier",
        name=run_name,
        config={
            "architecture": "DenseNet121",
            "backbone_frozen": True,
            "learning_rate": LEARNING_RATE,
            "epochs": 1 if quick else EPOCHS,
            "batch_size": 16,
        },
    )

    print("Building datasets...")
    train_ds = build_dataset("data/splits/train.csv", training=True)
    val_ds = build_dataset("data/splits/val.csv", training=False)

    if quick:
        # Limit to a handful of batches so this finishes in under a minute,
        # just to confirm the full train/val loop runs without errors.
        train_ds = train_ds.take(3)
        val_ds = val_ds.take(2)

    print("Building model...")
    model = build_model(freeze_backbone=True)
    model = compile_model(model, learning_rate=LEARNING_RATE)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=CHECKPOINT_PATH,
            monitor="val_auc",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc",
            mode="max",
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            verbose=1,
        ),
        WandbMetricsLogger(),
    ]

    print(f"Starting training ({'QUICK sanity check' if quick else 'full run'})...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=1 if quick else EPOCHS,
        callbacks=callbacks,
    )

    wandb.finish()

    print("\nTraining complete.")
    print(f"Best model checkpoint saved to: {CHECKPOINT_PATH}")
    print(f"Final training AUC: {history.history['auc'][-1]:.4f}")
    print(f"Final validation AUC: {history.history['val_auc'][-1]:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--quick", action="store_true",
        help="Run a fast sanity check (few batches, 1 epoch) instead of full training"
    )
    args = parser.parse_args()
    main(quick=args.quick)