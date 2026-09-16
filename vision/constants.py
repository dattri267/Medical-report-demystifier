"""
Shared constants for the Medical Report Demystifier project.

CRITICAL: The actual label list lives in shared/labels.json at the repo
root - NOT hardcoded here. That file is the single source of truth for
BOTH the Python (this file) and Java/Spring Boot (Partner B) sides of
the project. If the label list needs to change, edit shared/labels.json
only, never hardcode a second copy.
"""

import json
import os

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SHARED_CONFIG_PATH = os.path.join(_THIS_DIR, "..", "shared", "labels.json")

with open(_SHARED_CONFIG_PATH, "r") as f:
    _config = json.load(f)

LABELS = _config["labels"]
NUM_CLASSES = len(LABELS)
IMAGE_SIZE = _config.get("image_size", 224)

RANDOM_SEED = 42  # used everywhere for reproducibility (splits, training, sampling)
