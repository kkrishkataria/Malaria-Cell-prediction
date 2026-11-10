from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "malaria_cnn.keras"
METADATA_PATH = ROOT / "models" / "model_metadata.json"
IMAGE_SIZE = (128, 128)

app = FastAPI(title="CellScope AI", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_model: tf.keras.Model | None = None


def get_model() -> tf.keras.Model:
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise HTTPException(
                status_code=503,
                detail="Model not trained yet. Run: python app/train.py",
            )
        _model = tf.keras.models.load_model(MODEL_PATH)
    return _model


def prepare_image(contents: bytes) -> np.ndarray:
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Upload a valid PNG, JPG, or JPEG image.") from exc
    image = image.resize(IMAGE_SIZE)
    # The model begins with a Keras Rescaling layer, so keep inference pixels
    # in the same 0-255 range used by image_dataset_from_directory during training.
    array = np.asarray(image, dtype=np.float32)
    return np.expand_dims(array, axis=0)


@app.get("/api/health")
def health() -> dict:
    return {"status": "ready", "model_trained": MODEL_PATH.exists()}


@app.get("/api/model-info")
def model_info() -> dict:
    if METADATA_PATH.exists():
        return json.loads(METADATA_PATH.read_text())
    return {"model_trained": False, "message": "Train the CNN to see evaluation metrics."}


@app.post("/api/predict")
async def predict(file: UploadFile = File(...)) -> dict:
    if file.content_type not in {"image/jpeg", "image/jpg", "image/png"}:
        raise HTTPException(status_code=400, detail="Only JPG, JPEG, and PNG files are supported.")

    # Directory labels are alphabetical: Parasitized=0, Uninfected=1.
    # Convert the model's Uninfected score into the clinically useful infection score.
    uninfected_probability = float(get_model().predict(prepare_image(await file.read()), verbose=0)[0][0])
    probability = 1 - uninfected_probability
    infected = probability >= 0.5
    return {
        "prediction": "Parasitized" if infected else "Uninfected",
        "confidence": round((probability if infected else 1 - probability) * 100, 2),
        "infection_probability": round(probability * 100, 2),
        "risk_level": "High" if probability >= 0.75 else "Moderate" if probability >= 0.5 else "Low",
    }
