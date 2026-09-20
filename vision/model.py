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
    """
    base_model = tf.keras.applications.DenseNet121(
        include_top=False,
        weights="imagenet",
        input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
        pooling=None,
    )

    base_model.trainable = not freeze_backbone

    inputs = tf.keras.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
    x = base_model(inputs, training=False if freeze_backbone else None)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="sigmoid")(x)

    model = models.Model(inputs, outputs, name="chestxray_densenet121")

    return model


def compile_model(model: tf.keras.Model, learning_rate: float = 1e-3,
                   use_focal_loss: bool = False) -> tf.keras.Model:
    """
    Compile the model with an appropriate optimizer, loss, and metrics
    for multi-label classification.

    Args:
        use_focal_loss: if True, uses focal loss instead of plain
            binary_crossentropy. Focal loss down-weights "easy" examples
            (where the model is already confident and correct) and keeps
            full gradient signal on "hard" examples - directly helping
            with the severe class imbalance found in Phase 3's EDA
            (e.g. Hernia: 26 positives vs Infiltration: 1,032).
    """
    if use_focal_loss:
        loss = tf.keras.losses.BinaryFocalCrossentropy(
            apply_class_balancing=True,
            alpha=0.25,
            gamma=2.0,
        )
    else:
        loss = "binary_crossentropy"

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss,
        metrics=[
            tf.keras.metrics.AUC(name="auc", multi_label=True),
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
        ],
    )
    return model


def unfreeze_top_backbone_layers(model: tf.keras.Model, num_layers: int = 30) -> tf.keras.Model:
    """
    Unfreeze the top N layers of the DenseNet121 backbone for fine-tuning.

    The earliest layers (generic edge/texture detectors) stay frozen since
    they transfer well regardless of image domain. Only the LAST num_layers
    layers - the more abstract, task-specific ones - become trainable,
    so they can adapt to chest X-ray characteristics specifically.

    Must be called AFTER compile_model with a much lower learning rate
    (e.g. 1e-5), since these are pretrained weights we want to nudge
    gently, not overwrite.
    """
    backbone = model.get_layer("densenet121")
    backbone.trainable = True

    for layer in backbone.layers[:-num_layers]:
        layer.trainable = False
    for layer in backbone.layers[-num_layers:]:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
        else:
            layer.trainable = True

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
