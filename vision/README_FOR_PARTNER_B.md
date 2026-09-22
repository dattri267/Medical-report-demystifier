# Vision Engine - Integration Guide for Partner B

The vision model runs as a separate Python service. Your Spring Boot
backend calls it over HTTP, the same way it calls Gemini's API.

## Running the service

From the project root, with the Python venv activated:

    python -m uvicorn vision.api:app --port 8000

This starts the service at `http://localhost:8000`. For production,
this would run on its own server/container - ask if you need help
with that when we get to deployment.

## Endpoints

### GET /health
Returns `{"status": "ok"}` - use this to check the service is running.

### POST /predict
Send a chest X-ray image as multipart/form-data, field name `image`.

Example (Java, using Spring's WebClient or RestTemplate to call this):
    POST http://localhost:8000/predict
    Content-Type: multipart/form-data
    field "image" = <the X-ray file bytes>

Response (JSON):
    {
      "predictions": {
        "Atelectasis": 0.40,
        "Cardiomegaly": 0.04,
        ... (all 14 diseases, calibrated probabilities)
      },
      "top_predictions": [
        {"disease": "Atelectasis", "probability": 0.40},
        ... (top 5, sorted highest first)
      ],
      "model_version": "v3"
    }

The 14 disease names and their order come from `shared/labels.json` at
the repo root - import that same file into your Java code if you need
the label list, so we never have two different copies that could drift
out of sync.

### POST /gradcam?disease=Infiltration
Same image upload, plus a `disease` query parameter (must be one of the
14 exact names in shared/labels.json). Returns a PNG image directly
(not JSON) - the Grad-CAM heatmap overlay for that disease.

IMPORTANT: if the model's predicted probability for that disease is very
low (which /predict tells you), the Grad-CAM overlay will look flat/blank.
This is expected, not a bug - there's no meaningful "why" to show when the
model doesn't believe the disease is present.

## Model details (for your discrepancy checker / documentation)

- Architecture: DenseNet121 (frozen backbone) + custom classification head
- Test set macro AUC: 0.699
- Calibrated with temperature scaling (T=1.098) - probabilities are
  reasonably trustworthy but NOT clinically validated
- Full version history and per-class metrics: see models/registry/*.json
- This is a student/research prototype - NOT for actual diagnosis
