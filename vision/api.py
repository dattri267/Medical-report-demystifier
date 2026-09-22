"""
vision/api.py

A small HTTP API wrapping vision/inference.py, so Partner B's Java/Spring
Boot backend can call the vision model without needing Python, TensorFlow,
or any ML code - just a normal HTTP request.

Run locally:
    uvicorn vision.api:app --reload --port 8000

Endpoints:
    POST /predict
        - multipart/form-data with a field named "image" (the X-ray file)
        - returns JSON: {"predictions": {...}, "top_predictions": [...], "model_version": "v3"}

    POST /gradcam?disease=Infiltration
        - multipart/form-data with a field named "image"
        - returns a PNG image (the Grad-CAM overlay) as the response body

    GET /health
        - returns {"status": "ok"} - for Partner B to check the service is running
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import Response

from vision.inference import predict, generate_gradcam
from vision.constants import LABELS

app = FastAPI(title="Medical Report Demystifier - Vision Engine API")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict_endpoint(image: UploadFile = File(...)):
    image_bytes = await image.read()
    try:
        result = predict(image_bytes)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return result


@app.post("/gradcam")
async def gradcam_endpoint(
    image: UploadFile = File(...),
    disease: str = Query(..., description=f"One of: {', '.join(LABELS)}"),
):
    image_bytes = await image.read()
    try:
        overlay_png_bytes = generate_gradcam(image_bytes, disease)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return Response(content=overlay_png_bytes, media_type="image/png")
