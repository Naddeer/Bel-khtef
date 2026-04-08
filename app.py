"""
BelKhtef — FastAPI Backend (Week 4)
- Health check
- Vehicles listing
- Batch prediction endpoint
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import json
import os
import re
import logging
import numpy as np

app = FastAPI(
    title="BelKhtef API",
    description="API serving processed Tayara vehicle listings with ML predictions.",
    version="1.0.0",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.getenv("BELKHTEF_DATA_DIR", os.path.join(os.path.dirname(__file__), "data"))
DATA_FILE = os.path.join(BASE_DIR, "vehicles_gold.json")
MODEL_DIR = os.path.join(BASE_DIR, "models")

# Try to load gold data from data/ folder, fallback to root
if not os.path.exists(DATA_FILE):
    DATA_FILE = os.path.join(os.path.dirname(__file__), "vehicles_gold.json")

# Lazy-loaded model
_model = None


def _load_model():
    """Load the best serialized model (try xgboost first, then RF)."""
    global _model
    if _model is not None:
        return _model
    for name in ["xgboost.pkl", "random_forest.pkl"]:
        path = os.path.join(MODEL_DIR, name)
        if os.path.exists(path):
            import joblib
            _model = joblib.load(path)
            logging.info(f"Loaded model from {path}")
            return _model
    return None


# ── Schemas ────────────────────────────────────────────────────────────────

class VehicleInput(BaseModel):
    title: str
    price_tnd: float
    year: int

class PredictionResult(BaseModel):
    title: str
    price_tnd: float
    year: int
    is_good_deal: bool
    confidence: Optional[float] = None

class BatchRequest(BaseModel):
    vehicles: List[VehicleInput]

class BatchResponse(BaseModel):
    predictions: List[PredictionResult]
    model_loaded: bool


# ── Endpoints ──────────────────────────────────────────────────────────────

@app.get("/api/health")
def health_check():
    model = _load_model()
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "data_file_exists": os.path.exists(DATA_FILE),
    }


@app.get("/api/vehicles")
def get_vehicles():
    try:
        if not os.path.exists(DATA_FILE):
            raise HTTPException(status_code=404, detail="Data file not found")
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/predict/batch", response_model=BatchResponse)
def batch_predict(req: BatchRequest):
    """Batch prediction: classify a list of vehicles as good deals or not."""
    model = _load_model()
    predictions: List[PredictionResult] = []

    if model is None:
        # No model available → return heuristic-based results
        for v in req.vehicles:
            predictions.append(PredictionResult(
                title=v.title,
                price_tnd=v.price_tnd,
                year=v.year,
                is_good_deal=False,
                confidence=None,
            ))
        return BatchResponse(predictions=predictions, model_loaded=False)

    # Build feature matrix (must match modeling.py order: year, brand_enc, price_tnd)
    from sklearn.preprocessing import LabelEncoder

    brands = [_extract_brand(v.title) for v in req.vehicles]
    le = LabelEncoder()
    le.fit(brands if brands else ["UNKNOWN"])
    brand_enc = le.transform(brands)

    X = np.column_stack([
        [v.year for v in req.vehicles],
        brand_enc,
        [v.price_tnd for v in req.vehicles],
    ])

    preds = model.predict(X)
    probas = None
    if hasattr(model, "predict_proba"):
        probas = model.predict_proba(X)[:, 1]

    for i, v in enumerate(req.vehicles):
        predictions.append(PredictionResult(
            title=v.title,
            price_tnd=v.price_tnd,
            year=v.year,
            is_good_deal=bool(preds[i]),
            confidence=float(probas[i]) if probas is not None else None,
        ))

    return BatchResponse(predictions=predictions, model_loaded=True)


def _extract_brand(title: str) -> str:
    if not title:
        return "UNKNOWN"
    first_word = title.strip().split()[0].upper()
    return re.sub(r"[^A-Z]", "", first_word) or "UNKNOWN"


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
