# ============================================================
# api/schemas.py
# PURPOSE : Request and Response data models for FastAPI
# ============================================================

from pydantic import BaseModel, Field
from typing import Optional


class PredictionRequest(BaseModel):
    """
    Input data schema for prediction endpoint.
    These are the EXACT fields our model expects.
    """
    # Original features
    Category: str = Field(
        example="Electronics",
        description="Product category"
    )
    ProductPrice: float = Field(
        example=25000.0,
        description="Product price in INR",
        gt=0
    )
    Quantity: int = Field(
        example=1,
        description="Number of items ordered",
        gt=0
    )
    PaymentMethod: str = Field(
        example="UPI",
        description="Payment method used"
    )
    CustomerAge: int = Field(
        example=30,
        description="Customer age in years",
        gt=0
    )
    CustomerGender: str = Field(
        example="Male",
        description="Customer gender"
    )
    CustomerPurchaseHistory: int = Field(
        example=10,
        description="Total past purchases"
    )
    CustomerReturnHistory: int = Field(
        example=2,
        description="Total past returns"
    )
    ProductRating: float = Field(
        example=3.5,
        description="Product rating (1-5)",
        ge=1.0,
        le=5.0
    )
    Product_Warranty: str = Field(
        example="1 year",
        description="Warranty period"
    )
    ShippingMode: str = Field(
        example="Standard",
        description="Shipping mode"
    )
    DiscountApplied: float = Field(
        example=10.0,
        description="Discount percentage (0-100)",
        ge=0.0,
        le=100.0
    )

    class Config:
        json_schema_extra = {
            "example": {
                "Category": "Electronics",
                "ProductPrice": 25000.0,
                "Quantity": 1,
                "PaymentMethod": "UPI",
                "CustomerAge": 30,
                "CustomerGender": "Male",
                "CustomerPurchaseHistory": 10,
                "CustomerReturnHistory": 2,
                "ProductRating": 3.5,
                "Product_Warranty": "1 year",
                "ShippingMode": "Standard",
                "DiscountApplied": 10.0
            }
        }


class PredictionResponse(BaseModel):
    """Output schema returned by prediction endpoint."""
    prediction: int = Field(
        description="0 = Not Returned, 1 = Returned"
    )
    probability_return: float = Field(
        description="Probability of return (0-1)"
    )
    probability_no_return: float = Field(
        description="Probability of no return (0-1)"
    )
    risk_level: str = Field(
        description="LOW / MEDIUM / HIGH risk"
    )
    message: str = Field(
        description="Human readable prediction message"
    )


class HealthResponse(BaseModel):
    """Health check response schema."""
    status: str
    model_loaded: bool
    version: str
    message: str