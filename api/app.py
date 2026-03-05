from typing import Any, Dict, List
import os
import sys

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Ensure src/ is on the import path when running from the project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(BASE_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from data_preprocessing import preprocess_input_records
from model_utils import load_trained_model


app = FastAPI(title="Predictive Maintenance API", version="1.0.0")


class PredictRequest(BaseModel):
    records: List[Dict[str, Any]]


class PredictResponseItem(BaseModel):
    probability: float
    prediction: int


class PredictResponse(BaseModel):
    results: List[PredictResponseItem]


@app.on_event("startup")
def load_model_on_startup():
    """
    Load the trained model and preprocessing artifacts once at startup.
    """
    global MODEL, ARTIFACT
    MODEL, ARTIFACT = load_trained_model()
    MODEL.eval()


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest):
    """
    Run inference using the trained model.
    """
    if not payload.records:
        raise HTTPException(status_code=400, detail="No records provided.")

    features = ARTIFACT["features"]
    scaler = ARTIFACT["scaler"]

    try:
        X = preprocess_input_records(payload.records, features=features, scaler=scaler)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    import torch

    x_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(1)

    with torch.no_grad():
        outputs = MODEL(x_tensor).squeeze()

    import numpy as np

    probs = outputs.numpy()

    # Ensure probs is always a 1D array
    if not isinstance(probs, np.ndarray) or probs.ndim == 0:
        probs = np.array([float(probs)])

    preds = (probs >= 0.5).astype(int)

    results = [
        PredictResponseItem(probability=float(p), prediction=int(label))
        for p, label in zip(probs, preds)
    ]

    return PredictResponse(results=results)

