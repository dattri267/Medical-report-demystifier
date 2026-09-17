"""
vision/model.py

Builds the DenseNet121-based multi-label classification model:

    DenseNet121 (ImageNet pretrained, frozen)
        -> GlobalAveragePooling2D
        -> Dropout
        -> Dense(14, activation="sigmoid")

Sigmoid (not softmax) because this is MULTI-LABEL classification -
an X-ray can show multiple diseases at once, so each of the 14 outputs
is an independent yes/no probability, not a single pick-one distribution.
"""

import tensorflow as tf
from tensorflow.keras import layers, models

from vision.constants import NUM_CLASSES, IMAGE_SIZE


def build_model(freeze_backbone: bool = True, dropout_rate: float = 0.3) -> tf.keras.Model:
    """
    Build the DenseNet121 baseline model.

    Args:
        freeze_backbone: if True, DenseNet121's pretrained weights are
            frozen (not updated during training) - used for the initial
            baseline. Set False later during Phase 7 fine-tuning.
        dropout_rate: dropout applied before the final classification
            layer, to reduce overfitting on our relatively small dataset.

    Returns:
        A compiled tf.keras.Model ready for training.
    """
    # Load DenseNet121 with ImageNet pretrained weights.
    # include_top=False strips off the original 1000-class ImageNet
    # classification head - we're replacing it with our own 14-class head.
    base_model = tf.keras.applications.DenseNet121(
        include_top=False,
        weights="imagenet",
        input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
        pooling=None,  # we add our own pooling layer below
    )

    base_model.trainable = not freeze_backbone

    inputs = tf.keras.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
    x = base_model(inputs, training=False if freeze_backbone else None)

    # GlobalAveragePooling collapses each feature map to a single number
    # (its average), turning DenseNet's spatial feature maps into a flat
    # vector the Dense layer can use - far fewer parameters than Flatten,
    # and more robust to exactly where in the image a feature appears.
    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dropout(dropout_rate)(x)

    # 14 independent sigmoid outputs - one probability per disease,
    # NOT a softmax (which would incorrectly force them to sum to 1).
    outputs = layers.Dense(NUM_CLASSES, activation="sigmoid")(x)

    model = models.Model(inputs, outputs, name="chestxray_densenet121")

    return model


def compile_model(model: tf.keras.Model, learning_rate: float = 1e-3) -> tf.keras.Model:
    """
    Compile the model with an appropriate optimizer, loss, and metrics
    for multi-label classification.

    binary_crossentropy is used (not categorical_crossentropy) because
    each of the 14 outputs is treated as an independent binary decision.
    """
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.AUC(name="auc", multi_label=True),
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
        ],
    )
    return model


if __name__ == "__main__":
    print("Building baseline model (frozen backbone)...")
    model = build_model(freeze_backbone=True)
    model = compile_model(model)

    print("\nModel summary:")
    model.summary()

    trainable_params = sum(tf.size(w).numpy() for w in model.trainable_weights)
    total_params = model.count_params()
    print(f"\nTrainable parameters: {trainable_params:,}")
    print(f"Total parameters: {total_params:,}")
    print(f"Frozen parameters: {total_params - trainable_params:,}")