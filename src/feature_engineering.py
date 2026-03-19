# ============================================================
# src/feature_engineering.py
# PURPOSE : Create new features from existing columns
# CALLED BY: pipeline/training_pipeline.py
# ============================================================

import logging
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

# ── Logger ────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("FeatureEngineering")


# ── Config Loader ─────────────────────────────────────────────
def load_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


# ── Step 1: Drop Index Column ─────────────────────────────────
def drop_index_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop 'Unnamed: 0' — this is just a row index, not a feature.

    WHY: Row index has zero predictive power. Keeping it would
         confuse the model — row number doesn't predict returns.
    """
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
        logger.info("Dropped 'Unnamed: 0' index column")
    return df


# ── Step 2: Create Return Rate Feature ───────────────────────
def create_return_rate_feature(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create 'return_rate' = CustomerReturnHistory / CustomerPurchaseHistory

    WHY: A customer who returned 30 out of 50 orders (60% return
         rate) is far more likely to return than one who returned
         5 out of 100 orders (5% return rate).
         This single feature captures customer behavior pattern.
    """
    df["return_rate"] = np.where(
        df["CustomerPurchaseHistory"] > 0,
        df["CustomerReturnHistory"] / df["CustomerPurchaseHistory"],
        0.0
    )
    df["return_rate"] = df["return_rate"].round(4)

    logger.info(
        f"Created 'return_rate' | "
        f"mean={df['return_rate'].mean():.3f} | "
        f"max={df['return_rate'].max():.3f}"
    )
    return df


# ── Step 3: Create Price-Discount Feature ────────────────────
def create_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create price-related features.

    WHY:
    - 'effective_price' = actual amount paid after discount
      A Rs.50,000 item at 80% discount (paid Rs.10,000) behaves
      differently than a Rs.10,000 item at 0% discount.
    - 'is_high_value' = flag for expensive items (>25,000)
      High value items are returned more due to quality issues.
    - 'discount_price_ratio' = discount relative to price
      High discount on expensive item = suspicious quality.
    """
    # Effective price after discount
    df["effective_price"] = (
        df["ProductPrice"] * (1 - df["DiscountApplied"] / 100)
    ).round(2)

    # High value item flag
    df["is_high_value"] = (df["ProductPrice"] > 25000).astype(int)

    # Discount to price ratio
    df["discount_price_ratio"] = (
        df["DiscountApplied"] / (df["ProductPrice"] + 1)
    ).round(6)

    logger.info(
        f"Created price features | "
        f"high_value_items: {df['is_high_value'].sum():,} "
        f"({df['is_high_value'].mean():.1%})"
    )
    return df


# ── Step 4: Create Rating Risk Feature ───────────────────────
def create_rating_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create rating-based risk features.

    WHY:
    - 'is_low_rated' = products with rating < 3 are returned more
    - 'rating_risk_score' = combines rating with return history
      A customer with high return history buying low-rated product
      = very high return risk.
    """
    # Low rated product flag
    df["is_low_rated"] = (df["ProductRating"] < 3.0).astype(int)

    # Rating risk score — combines product quality with customer behavior
    df["rating_risk_score"] = (
        (5 - df["ProductRating"]) * df["return_rate"]
    ).round(4)

    logger.info(
        f"Created rating features | "
        f"low_rated_products: {df['is_low_rated'].sum():,} "
        f"({df['is_low_rated'].mean():.1%})"
    )
    return df


# ── Step 5: Create Quantity Feature ──────────────────────────
def create_quantity_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create quantity-based features.

    WHY: Customers who order large quantities are more likely
         to return some items — especially in fashion category
         (order 5 sizes, keep 1, return 4).
    """
    df["is_bulk_order"] = (df["Quantity"] > 3).astype(int)

    logger.info(
        f"Created quantity features | "
        f"bulk_orders: {df['is_bulk_order'].sum():,} "
        f"({df['is_bulk_order'].mean():.1%})"
    )
    return df


# ── Step 6: Create Age Group Feature ─────────────────────────
def create_age_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create age group feature.

    WHY: Different age groups have different return behaviors.
         Young customers (18-25) return more impulsive purchases.
         Middle-aged (26-45) are more deliberate buyers.
         Senior (46+) return less often.
    """
    df["age_group"] = pd.cut(
        df["CustomerAge"],
        bins=[0, 25, 35, 45, 100],
        labels=[0, 1, 2, 3]  # Young, Young-Adult, Middle, Senior
    ).astype(int)

    logger.info(
        f"Created age_group feature | "
        f"distribution: {df['age_group'].value_counts().to_dict()}"
    )
    return df


# ── Step 7: Create Combined Risk Score ───────────────────────
def create_risk_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create an overall return risk score combining multiple signals.

    WHY: This meta-feature summarizes all risk factors into one
         number — helps tree-based models split more efficiently.

    RISK FACTORS:
    - High return history (weight: 3)
    - Low product rating (weight: 2)
    - High value item (weight: 1)
    - Bulk order (weight: 1)
    - High discount (weight: 1)
    """
    df["risk_score"] = (
        (df["return_rate"] * 3) +
        (df["is_low_rated"] * 2) +
        (df["is_high_value"] * 1) +
        (df["is_bulk_order"] * 1) +
        ((df["DiscountApplied"] > 40).astype(int) * 1)
    ).round(4)

    logger.info(
        f"Created 'risk_score' | "
        f"mean={df['risk_score'].mean():.3f} | "
        f"max={df['risk_score'].max():.3f}"
    )
    return df


# ── Step 8: Train-Test Split ──────────────────────────────────
def split_data(
    df: pd.DataFrame,
    config: dict
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split data into train and test sets.

    WHY: We NEVER train and test on same data — that gives
         fake high accuracy. 80% training, 20% testing.

    STRATIFY: We use stratify=y to ensure both train and test
              have same class ratio (85/15) — very important
              for imbalanced datasets.
    """
    target = config["features"]["target_column"]

    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size    = config["data"]["test_size"],
        random_state = config["data"]["random_state"],
        stratify     = y  # Maintain class ratio in both splits
    )

    logger.info(f"Train set: {X_train.shape[0]:,} rows")
    logger.info(f"Test set : {X_test.shape[0]:,} rows")
    logger.info(
        f"Train class ratio: "
        f"{y_train.value_counts(normalize=True).to_dict()}"
    )
    logger.info(
        f"Test class ratio : "
        f"{y_test.value_counts(normalize=True).to_dict()}"
    )

    return X_train, X_test, y_train, y_test


# ── Print Feature Summary ─────────────────────────────────────
def print_feature_summary(
    df_before: pd.DataFrame,
    df_after: pd.DataFrame
) -> None:
    """Print before/after feature comparison."""
    original_cols = set(df_before.columns)
    new_cols      = set(df_after.columns) - original_cols

    print("\n" + "=" * 65)
    print("   FEATURE ENGINEERING SUMMARY")
    print("=" * 65)
    print(f"  Features before : {len(df_before.columns)}")
    print(f"  Features after  : {len(df_after.columns)}")
    print(f"  New features    : {len(new_cols)}")
    print(f"\n  ✨ New features created:")
    for col in sorted(new_cols):
        mean_val = df_after[col].mean()
        print(f"    + {col:<30} mean={mean_val:.4f}")
    print(f"\n  📋 All features going into model:")
    target = "Return_Risk"
    for i, col in enumerate(df_after.columns, 1):
        marker = " ← TARGET" if col == target else ""
        print(f"    {i:>2}. {col}{marker}")
    print("=" * 65 + "\n")


# ── Main Entry Point ──────────────────────────────────────────
def run_feature_engineering(
    df: pd.DataFrame,
    config_path: str = "config/config.yaml"
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Main function called by training pipeline.

    RETURNS: X_train, X_test, y_train, y_test
    """
    logger.info("=" * 50)
    logger.info("STARTING FEATURE ENGINEERING")
    logger.info("=" * 50)

    config    = load_config(config_path)
    df_before = df.copy()

    # Run all feature engineering steps
    logger.info("Step 1: Dropping index column...")
    df = drop_index_column(df)

    logger.info("Step 2: Creating return rate feature...")
    df = create_return_rate_feature(df)

    logger.info("Step 3: Creating price features...")
    df = create_price_features(df)

    logger.info("Step 4: Creating rating features...")
    df = create_rating_features(df)

    logger.info("Step 5: Creating quantity features...")
    df = create_quantity_features(df)

    logger.info("Step 6: Creating age group feature...")
    df = create_age_features(df)

    logger.info("Step 7: Creating combined risk score...")
    df = create_risk_score(df)

    # Print summary
    print_feature_summary(df_before, df)

    # Step 8: Train-test split
    logger.info("Step 8: Splitting into train/test sets...")
    X_train, X_test, y_train, y_test = split_data(df, config)

    # Save reference data for monitoring (20% of training data)
    reference_path = config["data"]["reference_data_path"]
    Path(reference_path).parent.mkdir(parents=True, exist_ok=True)
    reference_df = pd.concat(
        [X_train.head(10000), y_train.head(10000)], axis=1
    )
    reference_df.to_csv(reference_path, index=False)
    logger.info(f"Reference data saved to: {reference_path}")

    logger.info("FEATURE ENGINEERING COMPLETED SUCCESSFULLY ✓")
    logger.info("=" * 50)

    return X_train, X_test, y_train, y_test


# ── Standalone Run ────────────────────────────────────────────
if __name__ == "__main__":
    from src.data_ingestion    import run_data_ingestion
    from src.data_validation   import run_data_validation
    from src.data_preprocessing import run_data_preprocessing

    # Run pipeline steps
    df = run_data_ingestion()

    passed = run_data_validation(df)
    if not passed:
        print("❌ Validation failed!")
        exit(1)

    df_processed, encoders = run_data_preprocessing(df)

    X_train, X_test, y_train, y_test = run_feature_engineering(
        df_processed
    )

    print(f"\n✅ Feature Engineering complete!")
    print(f"   X_train shape : {X_train.shape}")
    print(f"   X_test shape  : {X_test.shape}")
    print(f"   Features      : {list(X_train.columns)}")
    print(f"   New features  : return_rate, effective_price,")
    print(f"                   is_high_value, discount_price_ratio,")
    print(f"                   is_low_rated, rating_risk_score,")
    print(f"                   is_bulk_order, age_group, risk_score")