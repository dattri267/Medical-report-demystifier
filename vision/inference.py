"""
vision/inference.py

Clean, simple prediction interface - Partner B calls predict() or the
HTTP API (vision/api.py) without needing to understand TensorFlow,
DenseNet121, or any training code.

Uses v3 (calibrated): baseline_best.keras weights + the temperature
scaling parameter fitted in Phase 10 (read from
models/registry/v3_calibrated.json), applied automatically.
"""

import json
import os

import cv2
import numpy as np
import tensorflow as tf

from vision.constants import LABELS, IMAGE_SIZE
from vision.gradcam import make_gradcam_heatmap, overlay_heatmap

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_PATH = os.path.join(_THIS_DIR, "..", "models", "checkpoints", "baseline_best.keras")
REGISTRY_PATH = os.path.join(_THIS_DIR, "..", "models", "registry", "v3_calibrated.json")

_model = None
_temperature = None


def _load_model_and_temperature():
    """Loads the model and calibration temperature once, then reuses them
    (avoids reloading the model on every single prediction)."""
    global _model, _temperature
    if _model is None:
        _model = tf.keras.models.load_model(CHECKPOINT_PATH)
        with open(REGISTRY_PATH) as f:
            manifest = json.load(f)
        _temperature = manifest["calibration_temperature"]
    return _model, _temperature


def _prob_to_logit(p, eps=1e-7):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def _apply_temperature(logits, T):
    return 1 / (1 + np.exp(-logits / T))


def preprocess_image_bytes(image_bytes: bytes):
    """
    Decodes raw image bytes (as received over HTTP) into model-ready
    input. Mirrors vision/preprocessing.py's pipeline exactly (CLAHE,
    resize, grayscale->RGB, DenseNet normalization) but works on bytes
    instead of a file path, since the API receives an upload, not a path.
    """
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("Could not decode image - must be a valid PNG or JPEG.")

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    image = clahe.apply(image)
    image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))

    display_img = np.stack([image, image, image], axis=-1)

    model_input = display_img.astype(np.float32)
    model_input = tf.keras.applications.densenet.preprocess_input(model_input)
    model_input = np.expand_dims(model_input, axis=0)

    return display_img, model_input


def predict(image_bytes: bytes) -> dict:
    """
    Main prediction function. Input: raw image bytes (PNG/JPEG).
    Output: dict matching the project spec -
        {
            "predictions": {"Atelectasis": 0.12, ...},   # all 14, calibrated
            "top_predictions": [{"disease": ..., "probability": ...}, ...],  # top 5
            "model_version": "v3"
        }
    """
    model, T = _load_model_and_temperature()
    _, model_input = preprocess_image_bytes(image_bytes)

    raw_probs = model.predict(model_input, verbose=0)[0]
    logits = _prob_to_logit(raw_probs)
    calibrated_probs = _apply_temperature(logits, T)

    predictions = {label: float(prob) for label, prob in zip(LABELS, calibrated_probs)}
    sorted_items = sorted(predictions.items(), key=lambda x: x[1], reverse=True)[:5]
    top_predictions = [{"disease": d, "probability": p} for d, p in sorted_items]

    return {
        "predictions": predictions,
        "top_predictions": top_predictions,
        "model_version": "v3",
    }


def generate_gradcam(image_bytes: bytes, disease: str) -> bytes:
    """
    Generates a Grad-CAM heatmap overlay for the given disease.
    Returns PNG-encoded image bytes, ready to send back over HTTP.
    """
    if disease not in LABELS:
        raise ValueError(f"Unknown disease '{disease}'. Must be one of: {LABELS}")

    model, _ = _load_model_and_temperature()
    display_img, model_input = preprocess_image_bytes(image_bytes)
    class_index = LABELS.index(disease)

    heatmap = make_gradcam_heatmap(model_input, model, class_index)
    overlay = overlay_heatmap(display_img, heatmap)

    success, buffer = cv2.imencode(".png", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    if not success:
        raise RuntimeError("Failed to encode Grad-CAM overlay as PNG.")
    return buffer.tobytes()
