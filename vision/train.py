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
from vision.model import build_model, compile_model, unfreeze_top_backbone_layers
from vision.preprocessing import build_dataset

tf.random.set_seed(RANDOM_SEED)

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = os.path.join(_THIS_DIR, "..", "models", "checkpoints")
BASELINE_CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "baseline_best.keras")
FINETUNED_CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "finetuned_best.keras")

EPOCHS = 10
LEARNING_RATE = 1e-3
FINETUNE_LEARNING_RATE = 1e-5  # ~100x smaller - gentle nudges, not overwrites
FINETUNE_EPOCHS = 10
FINETUNE_UNFREEZE_LAYERS = 15
FINETUNE_DROPOUT_RATE = 0.5
EARLY_STOPPING_PATIENCE = 3


def main(quick: bool = False, finetune: bool = False):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    if finetune:
        run_name = "quick-finetune-check" if quick else "finetuned-phase2"
    else:
        run_name = "quick-sanity-check" if quick else "baseline-frozen-backbone"

    wandb.init(
        project="medical-report-demystifier",
        name=run_name,
        config={
            "architecture": "DenseNet121",
            "phase": "fine-tuning" if finetune else "frozen-baseline",
            "learning_rate": FINETUNE_LEARNING_RATE if finetune else LEARNING_RATE,
            "epochs": 1 if quick else (FINETUNE_EPOCHS if finetune else EPOCHS),
            "batch_size": 16,
            "loss": "focal" if finetune else "binary_crossentropy",
        },
    )

    print("Building datasets...")
    train_ds = build_dataset("data/splits/train.csv", training=True)
    val_ds = build_dataset("data/splits/val.csv", training=False)

    if quick:
        train_ds = train_ds.take(3)
        val_ds = val_ds.take(2)

    if finetune:
        print(f"Loading baseline checkpoint from {BASELINE_CHECKPOINT_PATH}...")
        if not os.path.exists(BASELINE_CHECKPOINT_PATH):
            raise FileNotFoundError(
                "No baseline checkpoint found. Run `python -m vision.train` "
                "(Phase 6, without --finetune) first to produce baseline_best.keras."
            )
        model = build_model(freeze_backbone=True, dropout_rate=FINETUNE_DROPOUT_RATE)
        model.load_weights(BASELINE_CHECKPOINT_PATH)

        print(f"Unfreezing top {FINETUNE_UNFREEZE_LAYERS} backbone layers...")
        model = unfreeze_top_backbone_layers(model, num_layers=FINETUNE_UNFREEZE_LAYERS)

        print("Recompiling with focal loss and a much lower learning rate...")
        model = compile_model(model, learning_rate=FINETUNE_LEARNING_RATE, use_focal_loss=True)

        checkpoint_path = FINETUNED_CHECKPOINT_PATH
        n_epochs = FINETUNE_EPOCHS
    else:
        print("Building model...")
        model = build_model(freeze_backbone=True)
        model = compile_model(model, learning_rate=LEARNING_RATE)
        checkpoint_path = BASELINE_CHECKPOINT_PATH
        n_epochs = EPOCHS

    trainable_count = sum(tf.size(w).numpy() for w in model.trainable_weights)
    print(f"Trainable parameters this run: {trainable_count:,}")

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
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

    print(f"Starting training ({'QUICK sanity check' if quick else ('fine-tuning' if finetune else 'full baseline run')})...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=1 if quick else n_epochs,
        callbacks=callbacks,
    )

    wandb.finish()

    print("\nTraining complete.")
    print(f"Best model checkpoint saved to: {checkpoint_path}")
    print(f"Final training AUC: {history.history['auc'][-1]:.4f}")
    print(f"Final validation AUC: {history.history['val_auc'][-1]:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--quick", action="store_true",
        help="Run a fast sanity check (few batches, 1 epoch) instead of full training"
    )
    parser.add_argument(
        "--finetune", action="store_true",
        help="Phase 7: load baseline checkpoint, unfreeze top backbone layers, "
             "train with focal loss and a low learning rate"
    )
    args = parser.parse_args()
    main(quick=args.quick, finetune=args.finetune)
