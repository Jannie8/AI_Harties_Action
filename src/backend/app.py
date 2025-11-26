from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import os
import numpy as np
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware


# -----------------------------
# Paths (relative to repo root)
# -----------------------------
MODELS_DIR = "models/trained"
REPORTS_DIR = "models/reports"
DATA_PROCESSED = "data/processed"

RISK_MODEL_PATH = os.path.join(MODELS_DIR, "hyacinth_risk_model.joblib")
REG_MODEL_PATH = os.path.join(MODELS_DIR, "model_regression.joblib")
FEATURES_PATH = os.path.join(REPORTS_DIR, "model_features.csv")
FORECAST_PATH = os.path.join(DATA_PROCESSED, "hyacinth_forecast.csv")

try:
    clf_risk = joblib.load(RISK_MODEL_PATH)
except:
    clf_risk = None

try:
    reg_model = joblib.load(REG_MODEL_PATH)
except:
    reg_model = None

# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI(
    title="Harties Action Hyacinth API",
    description="Backend for bloom risk and coverage prediction.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------
# Load models & feature order
# -----------------------------
risk_model = None
reg_model = None
feature_order = None

if os.path.exists(RISK_MODEL_PATH):
    risk_model = joblib.load(RISK_MODEL_PATH)

if os.path.exists(REG_MODEL_PATH):
    reg_model = joblib.load(REG_MODEL_PATH)

if os.path.exists(FEATURES_PATH):
    feature_order = pd.read_csv(FEATURES_PATH, header=None)[0].tolist()
else:
    # Fallback: hardcode features if CSV not available
    feature_order = [
        "EC_Phys_Water",
        "pH_Diss_Water",
        "PO4_P_Diss_Water",
        "NO3_NO2_N_Diss_Water",
        "NH4_N_Diss_Water",
    ]


# -----------------------------
# Request / Response Schemas
# -----------------------------
class WaterQualityInput(BaseModel):
    EC_Phys_Water: float
    pH_Diss_Water: float
    PO4_P_Diss_Water: float
    NO3_NO2_N_Diss_Water: float
    NH4_N_Diss_Water: float


class BloomRiskResponse(BaseModel):
    bloom_probability: float
    bloom_class: int
    interpretation: str


class CoveragePredictionResponse(BaseModel):
    predicted_coverage_percent: float
    bloom_probability: float
    bloom_class: int
    interpretation: str



# -----------------------------
# Helper to build feature vector
# -----------------------------
def build_feature_vector(payload: WaterQualityInput) -> np.ndarray:
    """Create feature vector in correct order for the models."""
    data = {
        "EC_Phys_Water": payload.EC_Phys_Water,
        "pH_Diss_Water": payload.pH_Diss_Water,
        "PO4_P_Diss_Water": payload.PO4_P_Diss_Water,
        "NO3_NO2_N_Diss_Water": payload.NO3_NO2_N_Diss_Water,
        "NH4_N_Diss_Water": payload.NH4_N_Diss_Water,
    }
    # order features exactly as training
    vector = [data[f] for f in feature_order if f in data]
    return np.array(vector).reshape(1, -1)


# -----------------------------
# Endpoints
# -----------------------------
@app.get("/")
def root():
    return {
        "message": "Harties Action Hyacinth API is running.",
        "models_loaded": {
            "risk_model": risk_model is not None,
            "regression_model": reg_model is not None,
        },
        "features": feature_order,
    }


@app.post("/predict/risk", response_model=BloomRiskResponse)
def predict_bloom_risk(payload: WaterQualityInput):
    if risk_model is None:
        return BloomRiskResponse(
            bloom_probability=0.0,
            bloom_class=0,
            interpretation="Risk model not loaded on server.",
        )

    X = build_feature_vector(payload)
    prob = float(risk_model.predict_proba(X)[:, 1][0])
    cls = int(prob >= 0.5)

    if prob < 0.3:
        txt = "Low bloom risk"
    elif prob < 0.7:
        txt = "Moderate bloom risk"
    else:
        txt = "High bloom risk"

    return BloomRiskResponse(
        bloom_probability=prob,
        bloom_class=cls,
        interpretation=txt,
    )


@app.post("/predict/coverage", response_model=CoveragePredictionResponse)
def predict_coverage(payload: WaterQualityInput):
    if reg_model is None or clf_risk is None:
        return CoveragePredictionResponse(
            predicted_coverage_percent=-1.0,
            bloom_class=-1,
            bloom_probability=0.0,
            interpretation="Models not loaded"
        )

    # Convert input into model feature vector
    X = build_feature_vector(payload)

    # 🌿 1 — Predict coverage percentage (regression model)
    coverage_pred = float(reg_model.predict(X)[0])

    # 🌿 2 — Predict bloom risk (classification model)
    risk_prob = float(clf_risk.predict_proba(X)[0][1])  # probability of class 1 (bloom)
    risk_class = int(clf_risk.predict(X)[0])

    # 🌿 3 — Interpret the result
    if risk_prob > 0.85:
        interp = "Very high bloom risk — immediate intervention likely required."
    elif risk_prob > 0.60:
        interp = "High bloom risk — monitor closely and prepare to deploy teams."
    elif risk_prob > 0.30:
        interp = "Moderate risk — conditions may enable hyacinth growth."
    else:
        interp = "Low bloom risk — conditions are stable."

    return CoveragePredictionResponse(
        predicted_coverage_percent=coverage_pred,
        bloom_probability=risk_prob,
        bloom_class=risk_class,
        interpretation=interp
    )



@app.get("/forecast")
def get_forecast(limit: int = 30):
    """
    Return the latest hyacinth coverage forecast produced by your notebook.
    """
    if not os.path.exists(FORECAST_PATH):
        return {"error": "Forecast file not found."}

    df = pd.read_csv(FORECAST_PATH)
    df = df.tail(limit)
    return df.to_dict(orient="records")
