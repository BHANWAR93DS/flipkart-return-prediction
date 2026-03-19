# ============================================================
# api/main.py
# PURPOSE : FastAPI application for model serving
# RUN WITH: uvicorn api.main:app --host 0.0.0.0 --port 8000
# ============================================================

import os
os.environ["GIT_PYTHON_REFRESH"] = "quiet"

import logging
import yaml
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import (
    PredictionRequest,
    PredictionResponse,
    HealthResponse
)

# ── Logger ────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("FlipkartAPI")


# ── Global Model Store ────────────────────────────────────────
# WHY: We load model ONCE when server starts — not on every
#      request. Loading model per request = 2-3 sec latency!
model_store = {
    "model":    None,
    "scaler":   None,
    "encoders": None,
    "config":   None,
}


# ── Config Loader ─────────────────────────────────────────────
def load_config(path: str = "config/config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ── Model Loader ──────────────────────────────────────────────
def load_models() -> bool:
    """
    Load model, scaler, and encoders at startup.
    Returns True if successful, False otherwise.
    """
    try:
        config = load_config()
        model_store["config"] = config

        # Load model
        model_path = "models/model.pkl"
        if not Path(model_path).exists():
            logger.error(f"Model not found: {model_path}")
            return False
        model_store["model"] = joblib.load(model_path)
        logger.info(f"Model loaded: {model_path}")

        # Load scaler
        scaler_path = "models/scaler.pkl"
        if Path(scaler_path).exists():
            model_store["scaler"] = joblib.load(scaler_path)
            logger.info(f"Scaler loaded: {scaler_path}")

        # Load encoders
        encoders_path = "models/encoders.pkl"
        if Path(encoders_path).exists():
            model_store["encoders"] = joblib.load(encoders_path)
            logger.info(f"Encoders loaded: {encoders_path}")

        logger.info("All models loaded successfully!")
        return True

    except Exception as e:
        logger.error(f"Error loading models: {e}")
        return False


# ── Lifespan (startup/shutdown) ───────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting Flipkart Return Prediction API...")
    success = load_models()
    if success:
        logger.info("API ready to serve predictions!")
    else:
        logger.warning("API started but model not loaded!")
    yield
    # Shutdown
    logger.info("Shutting down API...")


# ── FastAPI App ───────────────────────────────────────────────
app = FastAPI(
    title="Flipkart Return Prediction API",
    description="""
    ## Flipkart Product Return Prediction

    Predict whether a Flipkart order will be returned by the customer.

    ### How to use:
    1. Send a POST request to `/predict` with order details
    2. Get back return probability and risk level

    ### Risk Levels:
    - **LOW**    : < 30% return probability
    - **MEDIUM** : 30-60% return probability
    - **HIGH**   : > 60% return probability
    """,
    version="1.0.0",
    lifespan=lifespan
)

# Allow all origins for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Feature Engineering Helper ────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply same feature engineering as training pipeline.
    WHY: Model was trained on engineered features — we must
         apply EXACT same transformations at prediction time.
    """
    # Return rate
    df["return_rate"] = np.where(
        df["CustomerPurchaseHistory"] > 0,
        df["CustomerReturnHistory"] / df["CustomerPurchaseHistory"],
        0.0
    )

    # Price features
    df["effective_price"] = (
        df["ProductPrice"] * (1 - df["DiscountApplied"] / 100)
    )
    df["is_high_value"] = (df["ProductPrice"] > 25000).astype(int)
    df["discount_price_ratio"] = (
        df["DiscountApplied"] / (df["ProductPrice"] + 1)
    )

    # Rating features
    df["is_low_rated"]      = (df["ProductRating"] < 3.0).astype(int)
    df["rating_risk_score"] = (5 - df["ProductRating"]) * df["return_rate"]

    # Quantity features
    df["is_bulk_order"] = (df["Quantity"] > 3).astype(int)

    # Age group
    df["age_group"] = pd.cut(
        df["CustomerAge"],
        bins=[0, 25, 35, 45, 100],
        labels=[0, 1, 2, 3]
    ).astype(int)

    # Risk score
    df["risk_score"] = (
        (df["return_rate"] * 3) +
        (df["is_low_rated"] * 2) +
        (df["is_high_value"] * 1) +
        (df["is_bulk_order"] * 1) +
        ((df["DiscountApplied"] > 40).astype(int) * 1)
    )

    return df


# ── Encode Helper ─────────────────────────────────────────────
def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply saved label encoders to categorical columns."""
    encoders = model_store["encoders"]
    config   = model_store["config"]

    if encoders is None:
        return df

    cat_cols = config["features"]["categorical_features"]
    for col in cat_cols:
        if col in df.columns and col in encoders:
            le = encoders[col]
            # Handle unseen categories
            val = df[col].astype(str).iloc[0]
            if val not in le.classes_:
                val = le.classes_[0]
            df[col] = le.transform([val])

    return df


# ── Risk Level Helper ─────────────────────────────────────────
def get_risk_level(probability: float) -> str:
    if probability < 0.30:
        return "LOW"
    elif probability < 0.60:
        return "MEDIUM"
    else:
        return "HIGH"


# ══════════════════════════════════════════════════════════════
# API ENDPOINTS
# ══════════════════════════════════════════════════════════════

# ── Health Check ──────────────────────────────────────────────
@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint — basic info."""
    return HealthResponse(
        status="running",
        model_loaded=model_store["model"] is not None,
        version="1.0.0",
        message="Flipkart Return Prediction API is running!"
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.
    WHY: Load balancers and monitoring tools ping /health
         to check if service is alive. Used in Docker and K8s.
    """
    model_loaded = model_store["model"] is not None
    return HealthResponse(
        status="healthy" if model_loaded else "unhealthy",
        model_loaded=model_loaded,
        version="1.0.0",
        message="Model ready!" if model_loaded else "Model not loaded!"
    )


# ── Prediction Endpoint ───────────────────────────────────────
@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Main prediction endpoint.

    Send order details → Get return prediction back.

    WHY POST not GET: Prediction requests contain sensitive
    customer data — POST keeps it in body, not URL.
    """
    # Check model is loaded
    if model_store["model"] is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please check server logs."
        )

    try:
        # Step 1: Convert request to DataFrame
        input_data = {
            "Category":                 request.Category,
            "ProductPrice":             request.ProductPrice,
            "Quantity":                 request.Quantity,
            "PaymentMethod":            request.PaymentMethod,
            "CustomerAge":              request.CustomerAge,
            "CustomerGender":           request.CustomerGender,
            "CustomerPurchaseHistory":  request.CustomerPurchaseHistory,
            "CustomerReturnHistory":    request.CustomerReturnHistory,
            "ProductRating":            request.ProductRating,
            "Product_Warranty":         request.Product_Warranty,
            "ShippingMode":             request.ShippingMode,
            "DiscountApplied":          request.DiscountApplied,
        }
        df = pd.DataFrame([input_data])

        # Step 2: Encode categorical features
        df = encode_features(df)

        # Step 3: Engineer features
        df = engineer_features(df)

        # Step 4: Scale numerical features
        if model_store["scaler"] is not None:
            config   = model_store["config"]
            num_cols = [
                c for c in config["features"]["numerical_features"]
                if c in df.columns
            ]
            df[num_cols] = model_store["scaler"].transform(df[num_cols])

        # Step 5: Ensure correct column order
        # Must match training feature order exactly!
        expected_cols = [
            "Category", "ProductPrice", "Quantity", "PaymentMethod",
            "CustomerAge", "CustomerGender", "CustomerPurchaseHistory",
            "CustomerReturnHistory", "ProductRating", "Product_Warranty",
            "ShippingMode", "DiscountApplied", "return_rate",
            "effective_price", "is_high_value", "discount_price_ratio",
            "is_low_rated", "rating_risk_score", "is_bulk_order",
            "age_group", "risk_score"
        ]
        df = df[expected_cols]

        # Step 6: Make prediction
        model       = model_store["model"]
        prediction  = int(model.predict(df)[0])
        probability = model.predict_proba(df)[0]

        prob_return    = round(float(probability[1]), 4)
        prob_no_return = round(float(probability[0]), 4)
        risk_level     = get_risk_level(prob_return)

        # Step 7: Build response message
        if prediction == 1:
            message = (
                f"⚠️ HIGH RETURN RISK — "
                f"{prob_return*100:.1f}% probability of return. "
                f"Risk Level: {risk_level}"
            )
        else:
            message = (
                f"✅ LOW RETURN RISK — "
                f"{prob_no_return*100:.1f}% probability of keeping. "
                f"Risk Level: {risk_level}"
            )

        logger.info(
            f"Prediction: {prediction} | "
            f"Prob: {prob_return:.4f} | "
            f"Risk: {risk_level}"
        )

        return PredictionResponse(
            prediction            = prediction,
            probability_return    = prob_return,
            probability_no_return = prob_no_return,
            risk_level            = risk_level,
            message               = message
        )

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


# ── Model Info Endpoint ───────────────────────────────────────
@app.get("/model-info")
async def model_info():
    """
    Return information about the loaded model.
    WHY: Useful for debugging — tells you which model
         version is currently serving predictions.
    """
    if model_store["model"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    model  = model_store["model"]
    config = model_store["config"]

    return {
        "model_type":   type(model).__name__,
        "n_estimators": model.n_estimators,
        "max_depth":    model.max_depth,
        "n_features":   model.n_features_in_,
        "experiment":   config["mlflow"]["experiment_name"],
        "version":      config["project"]["version"],
    }


# ── Batch Prediction Endpoint ─────────────────────────────────
@app.post("/predict/batch")
async def predict_batch(requests: list[PredictionRequest]):
    """
    Batch prediction — predict for multiple orders at once.
    WHY: More efficient than calling /predict N times.
         Used in bulk operations like daily batch scoring.
    """
    if model_store["model"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if len(requests) > 1000:
        raise HTTPException(
            status_code=400,
            detail="Max 1000 records per batch request"
        )

    results = []
    for req in requests:
        try:
            # Reuse single prediction logic
            single_result = await predict(req)
            results.append(single_result)
        except Exception as e:
            results.append({"error": str(e)})

    return {
        "total":       len(requests),
        "predictions": results
    }