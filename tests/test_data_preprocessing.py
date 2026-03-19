# tests/test_preprocessing.py

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def create_sample_df():
    """Test ke liye sample DataFrame banao."""
    return pd.DataFrame({
        "Category":               ["Electronics", "Books"],
        "ProductPrice":           [25000.0, 500.0],
        "Quantity":               [1, 2],
        "PaymentMethod":          ["UPI", "COD"],
        "CustomerAge":            [30, 25],
        "CustomerGender":         ["Male", "Female"],
        "CustomerPurchaseHistory":[10, 5],
        "CustomerReturnHistory":  [2, 1],
        "ProductRating":          [3.5, 4.0],
        "Product_Warranty":       ["1 year", "No Warranty"],
        "ShippingMode":           ["Standard", "Express"],
        "DiscountApplied":        [10.0, 5.0],
        "Return_Risk":            [0, 1],
    })


def test_dataframe_creation():
    """DataFrame sahi banta hai."""
    df = create_sample_df()
    assert len(df) == 2
    assert "Return_Risk" in df.columns


def test_return_rate_calculation():
    """Return rate sahi calculate hoti hai."""
    df = create_sample_df()
    df["return_rate"] = np.where(
        df["CustomerPurchaseHistory"] > 0,
        df["CustomerReturnHistory"] / df["CustomerPurchaseHistory"],
        0.0
    )
    assert "return_rate" in df.columns
    assert df["return_rate"].iloc[0] == pytest.approx(0.2)


def test_effective_price_calculation():
    """Effective price sahi calculate hoti hai."""
    df = create_sample_df()
    df["effective_price"] = (
        df["ProductPrice"] * (1 - df["DiscountApplied"] / 100)
    )
    assert df["effective_price"].iloc[0] == pytest.approx(22500.0)


def test_no_null_values():
    """Sample data mein null values nahi hain."""
    df = create_sample_df()
    assert df.isnull().sum().sum() == 0


def test_target_column_binary():
    """Target column binary hai (0 or 1)."""
    df = create_sample_df()
    assert set(df["Return_Risk"].unique()).issubset({0, 1})