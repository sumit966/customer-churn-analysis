"""FastAPI service for churn prediction and analytics."""
import sys
import os
import json
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional


app = FastAPI(
    title="Customer Churn Prediction API",
    description="Predict churn probability and return churn drivers",
    version="1.0.0",
)

MODEL = None
PREPROCESSOR = None
METADATA = None


class CustomerFeatures(BaseModel):
    gender: str = Field(..., examples=["Male"])
    senior_citizen: int = Field(0, ge=0, le=1)
    partner: str = Field(..., examples=["Yes"])
    dependents: str = Field(..., examples=["No"])
    tenure: int = Field(..., ge=0, le=100)
    contract: str = Field(..., examples=["Month-to-month"])
    internet_service: str = Field(..., examples=["Fiber optic"])
    tech_support: str = Field(..., examples=["No"])
    payment_method: str = Field(..., examples=["Electronic check"])
    paperless_billing: str = Field(..., examples=["Yes"])
    monthly_charges: float = Field(..., ge=0)
    total_charges: float = Field(..., ge=0)


class ChurnResponse(BaseModel):
    churn_probability: float
    risk_level: str
    top_drivers: list


@app.on_event("startup")
def load():
    global MODEL, PREPROCESSOR, METADATA
    try:
        MODEL = joblib.load("models/churn_model.joblib")
        PREPROCESSOR = joblib.load("models/preprocessor.joblib")
        with open("models/metadata.json") as f:
            METADATA = json.load(f)
        print("[OK] Churn model loaded")
    except FileNotFoundError:
        print("[WARN] Model not found. Run: python src/train.py")


def to_df(c: CustomerFeatures) -> pd.DataFrame:
    return pd.DataFrame([c.model_dump()])


@app.get("/")
def root():
    return {"message": "Customer Churn Prediction API", "docs": "/docs"}


@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": MODEL is not None}


@app.post("/predict", response_model=ChurnResponse)
def predict(customer: CustomerFeatures):
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    df = to_df(customer)
    X = PREPROCESSOR.transform(df)
    proba = float(MODEL.predict_proba(X)[0, 1])

    if proba < 0.3:
        risk = "Low"
    elif proba < 0.6:
        risk = "Medium"
    else:
        risk = "High"

    drivers = METADATA.get("top_churn_drivers", [])[:5] if METADATA else []

    return ChurnResponse(
        churn_probability=round(proba, 4),
        risk_level=risk,
        top_drivers=drivers,
    )


@app.get("/drivers")
def drivers():
    if METADATA is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"top_churn_drivers": METADATA.get("top_churn_drivers", [])}


@app.get("/metrics")
def metrics():
    if METADATA is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return METADATA.get("all_results", {})
