# tests/test_api.py

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def test_schemas_import():
    """API schemas import hoti hain."""
    from api.schemas import PredictionRequest, PredictionResponse
    assert PredictionRequest is not None
    assert PredictionResponse is not None


def test_prediction_request_schema():
    """PredictionRequest schema valid data accept karta hai."""
    from api.schemas import PredictionRequest
    req = PredictionRequest(
        Category="Electronics",
        ProductPrice=25000.0,
        Quantity=1,
        PaymentMethod="UPI",
        CustomerAge=30,
        CustomerGender="Male",
        CustomerPurchaseHistory=10,
        CustomerReturnHistory=2,
        ProductRating=3.5,
        Product_Warranty="1 year",
        ShippingMode="Standard",
        DiscountApplied=10.0
    )
    assert req.Category == "Electronics"
    assert req.ProductPrice == 25000.0


def test_health_response_schema():
    """HealthResponse schema valid hai."""
    from api.schemas import HealthResponse
    resp = HealthResponse(
        status="healthy",
        model_loaded=True,
        version="1.0.0",
        message="Model ready!"
    )
    assert resp.status == "healthy"
    assert resp.model_loaded is True