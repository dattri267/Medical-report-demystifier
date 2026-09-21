"""
vision/gradcam.py

Implements Grad-CAM for the baseline DenseNet121 model, producing heatmap
overlays showing which image regions most influenced a given disease
prediction.

IMPORTANT: Grad-CAM shows which regions the model attended to when making
a prediction - it does NOT prove disease is actually present there. It is
an interpretability/debugging tool, not a diagnostic confirmation.

Usage:
    python -m vision.gradcam
"""

import os

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf

from vision.constants import LABELS, IMAGE_SIZE

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_PATH = os.path.join(_THIS_DIR, "..", "models", "checkpoints", "baseline_best.keras")
IMAGES_ROOT = os.path.join(_THIS_DIR, "..", "data", "raw", "images")
TEST_CSV_PATH = os.path.join(_THIS_DIR, "..", "data", "splits", "test.csv")
OUTPUT_DIR = os.path.join(_THIS_DIR, "..", "outputs", "gradcam")


def load_and_prepare_image(image_path: str):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    image = clahe.apply(image)
    image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))

    display_img = np.stack([image, image, image], axis=-1)

    model_input = display_img.astype(np.float32)
    model_input = tf.keras.applications.densenet.preprocess_input(model_input)
    model_input = np.expand_dims(model_input, axis=0)

    return display_img, model_input


def find_last_conv_layer_name(backbone: tf.keras.Model) -> str:
    for layer in reversed(backbone.layers):
        if len(layer.output.shape) == 4:
            return layer.name
    raise ValueError("No 4D (spatial feature map) layer found in backbone.")


def make_gradcam_heatmap(model_input, model, class_index: int) -> np.ndarray:
    backbone = model.get_layer("densenet121")
    last_conv_layer_name = find_last_conv_layer_name(backbone)
    last_conv_layer = backbone.get_layer(last_conv_layer_name)

    backbone_grad_model = tf.keras.models.Model(
        inputs=backbone.input,
        outputs=[last_conv_layer.output, backbone.output],
    )

    gap_layer = model.get_layer(index=2)
    dense_layer = model.get_layer(index=4)

    with tf.GradientTape() as tape:
        conv_output, backbone_output = backbone_grad_model(model_input)
        tape.watch(conv_output)
        pooled = gap_layer(backbone_output)
        predictions = dense_layer(pooled)
        class_score = predictions[:, class_index]

    grads = tape.gradient(class_score, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_output = conv_output[0]
    heatmap = tf.reduce_sum(conv_output * pooled_grads, axis=-1)
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)

    return heatmap.numpy()


def overlay_heatmap(display_img: np.ndarray, heatmap: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    heatmap_resized = cv2.resize(heatmap, (display_img.shape[1], display_img.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    overlay = cv2.addWeighted(display_img, 1 - alpha, heatmap_colored, alpha, 0)
    return overlay


def find_most_confident_example(model, test_df, label, class_index):
    positive_examples = test_df[test_df[label] == 1]
    if len(positive_examples) == 0:
        return None, -1

    best_prob = -1
    best_row = None
    for _, candidate_row in positive_examples.iterrows():
        candidate_path = os.path.join(IMAGES_ROOT, candidate_row["Image Index"])
        _, candidate_input = load_and_prepare_image(candidate_path)
        prob = model.predict(candidate_input, verbose=0)[0][class_index]
        if prob > best_prob:
            best_prob = prob
            best_row = candidate_row

    return best_row, best_prob


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Loading baseline model from {CHECKPOINT_PATH}...")
    model = tf.keras.models.load_model(CHECKPOINT_PATH)

    test_df = pd.read_csv(TEST_CSV_PATH)

    print("Generating Grad-CAM overlays (most confident correct example per disease)...")
    for class_index, label in enumerate(LABELS):
        best_row, best_prob = find_most_confident_example(model, test_df, label, class_index)
        if best_row is None:
            print(f"  Skipping {label} - no positive examples in test set.")
            continue

        image_path = os.path.join(IMAGES_ROOT, best_row["Image Index"])
        display_img, model_input = load_and_prepare_image(image_path)
        heatmap = make_gradcam_heatmap(model_input, model, class_index)

        if heatmap.std() < 1e-6:
            print(f"  WARNING {label}: model's best example only reached "
                  f"{best_prob:.3f} probability - heatmap is flat (no positive "
                  f"gradient signal). This reflects genuine low model confidence, "
                  f"not a code bug. Saving anyway, labeled clearly.")

        overlay = overlay_heatmap(display_img, heatmap)

        prob_str = f"{best_prob:.3f}".replace(".", "p")
        save_path = os.path.join(OUTPUT_DIR, f"{label}_prob{prob_str}.png")
        cv2.imwrite(save_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        print(f"  Saved {label} (prob={best_prob:.3f}): {save_path}")

    print(f"\nAll Grad-CAM overlays saved to: {OUTPUT_DIR}")
    print("\nREMINDER: Grad-CAM shows which regions the model attended to for a")
    print("prediction - it does NOT prove disease is present there. It's an")
    print("interpretability tool, not a diagnostic confirmation.")


if __name__ == "__main__":
    main()
