# ============================================================
# src/data_preprocessing.py
# PURPOSE : Clean raw data and prepare for feature engineering
# CALLED BY: pipeline/training_pipeline.py
# ============================================================

import logging
import yaml
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import LabelEncoder

# ── Logger ────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("DataPreprocessing")


# ── Config Loader ─────────────────────────────────────────────
def load_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


# ── Step 1: Drop Unnecessary Columns ─────────────────────────
def drop_unnecessary_columns(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Drop columns that are not useful for ML model.

    WHY: Columns like OrderID, dates, free-text names add noise
         and cause data leakage if kept. We only keep features
         that will be available at PREDICTION TIME.

    LEAKAGE EXAMPLE: ReturnDate tells us if item was returned —
         but at prediction time, we don't have ReturnDate yet!
         Keeping it would give 100% accuracy but zero real value.
    """
    drop_cols = config["features"]["drop_columns"]

    # Only drop columns that actually exist
    cols_to_drop = [c for c in drop_cols if c in df.columns]
    df = df.drop(columns=cols_to_drop)

    logger.info(f"Dropped {len(cols_to_drop)} columns: {cols_to_drop}")
    logger.info(f"Remaining columns: {list(df.columns)}")
    return df


# ── Step 2: Handle Missing Values ────────────────────────────
def handle_missing_values(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Fill missing values with appropriate strategies.

    WHY: ML models cannot handle NaN values — they will crash.
         We use:
         - Median for numerical (robust to outliers)
         - Mode for categorical (most common value)
    """
    numerical_cols   = config["features"]["numerical_features"]
    categorical_cols = config["features"]["categorical_features"]

    null_before = df.isnull().sum().sum()

    # Fill numerical nulls with median
    for col in numerical_cols:
        if col in df.columns and df[col].isnull().sum() > 0:
            median_val = df[col].median()
            df[col]    = df[col].fillna(median_val)
            logger.info(f"  Filled '{col}' nulls with median: {median_val:.2f}")

    # Fill categorical nulls with mode
    for col in categorical_cols:
        if col in df.columns and df[col].isnull().sum() > 0:
            mode_val = df[col].mode()[0]
            df[col]  = df[col].fillna(mode_val)
            logger.info(f"  Filled '{col}' nulls with mode: {mode_val}")

    null_after = df.isnull().sum().sum()
    logger.info(f"Missing values: {null_before} → {null_after}")
    return df


# ── Step 3: Remove Duplicate Rows ────────────────────────────
def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate rows from the dataset.

    WHY: Duplicate rows cause data leakage — same row can appear
         in both train and test set giving false high accuracy.
    """
    before = len(df)
    df     = df.drop_duplicates()
    after  = len(df)
    removed = before - after

    if removed > 0:
        logger.info(f"Removed {removed:,} duplicate rows")
    else:
        logger.info("No duplicate rows found")

    return df


# ── Step 4: Encode Categorical Columns ───────────────────────
def encode_categorical_columns(
    df: pd.DataFrame,
    config: dict,
    encoders: dict = None,
    is_training: bool = True
) -> tuple[pd.DataFrame, dict]:
    """
    Convert categorical text columns to numbers using LabelEncoder.

    WHY: ML models only understand numbers, not strings like
         'UPI', 'COD', 'Express', 'Standard'.

    IMPORTANT: We save the encoders during training and reuse
               them during prediction — so encoding is consistent.
               'UPI' must always map to same number.

    RETURNS: (encoded DataFrame, encoders dict)
    """
    categorical_cols = config["features"]["categorical_features"]

    if encoders is None:
        encoders = {}

    for col in categorical_cols:
        if col not in df.columns:
            continue

        if is_training:
            # Training: fit new encoder and save it
            le          = LabelEncoder()
            df[col]     = le.fit_transform(df[col].astype(str))
            encoders[col] = le
            logger.info(
                f"  Encoded '{col}': "
                f"{list(le.classes_)} → {list(range(len(le.classes_)))}"
            )
        else:
            # Prediction: use saved encoder
            if col in encoders:
                le = encoders[col]
                # Handle unseen categories gracefully
                df[col] = df[col].astype(str).map(
                    lambda x: x if x in le.classes_ else le.classes_[0]
                )
                df[col] = le.transform(df[col])
            else:
                logger.warning(f"No encoder found for '{col}' — skipping")

    logger.info(f"Encoded {len(categorical_cols)} categorical columns")
    return df, encoders


# ── Step 5: Fix Data Types ────────────────────────────────────
def fix_data_types(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Ensure all columns have correct data types.

    WHY: Pandas sometimes loads integers as floats or
         vice versa. Consistent types = consistent results.
    """
    numerical_cols = config["features"]["numerical_features"]

    for col in numerical_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    logger.info("Data types fixed for all numerical columns")
    return df


# ── Step 6: Save Encoders ─────────────────────────────────────
def save_encoders(encoders: dict, path: str = "models/encoders.pkl") -> None:
    """
    Save label encoders to disk for use during prediction.

    WHY: When API receives a prediction request, it needs to
         encode 'UPI' the same way it was encoded during training.
         Without saved encoders, predictions would be wrong.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(encoders, path)
    logger.info(f"Encoders saved to: {path}")


# ── Step 7: Load Encoders ─────────────────────────────────────
def load_encoders(path: str = "models/encoders.pkl") -> dict:
    """Load saved encoders for prediction time use."""
    if not Path(path).exists():
        raise FileNotFoundError(f"Encoders not found at: {path}")
    encoders = joblib.load(path)
    logger.info(f"Encoders loaded from: {path}")
    return encoders


# ── Print Preprocessing Summary ───────────────────────────────
def print_preprocessing_summary(
    df_before: pd.DataFrame,
    df_after: pd.DataFrame
) -> None:
    """Print before/after comparison."""
    print("\n" + "=" * 65)
    print("   PREPROCESSING SUMMARY")
    print("=" * 65)
    print(f"  Rows    : {len(df_before):>10,} → {len(df_after):>10,}")
    print(f"  Columns : {len(df_before.columns):>10} → {len(df_after.columns):>10}")
    print(f"  Nulls   : {df_before.isnull().sum().sum():>10,} → "
          f"{df_after.isnull().sum().sum():>10,}")
    print(f"\n  Final columns in processed data:")
    for i, col in enumerate(df_after.columns, 1):
        dtype = str(df_after[col].dtype)
        print(f"    {i:>2}. {col:<35} [{dtype}]")
    print("=" * 65 + "\n")


# ── Main Entry Point ──────────────────────────────────────────
def run_data_preprocessing(
    df: pd.DataFrame,
    config_path: str = "config/config.yaml",
    is_training: bool = True
) -> tuple[pd.DataFrame, dict]:
    """
    Main function called by training pipeline.

    RETURNS: (processed DataFrame, encoders dict)
    """
    logger.info("=" * 50)
    logger.info("STARTING DATA PREPROCESSING")
    logger.info("=" * 50)

    config    = load_config(config_path)
    df_before = df.copy()

    # Run all preprocessing steps in order
    logger.info("Step 1: Dropping unnecessary columns...")
    df = drop_unnecessary_columns(df, config)

    logger.info("Step 2: Removing duplicate rows...")
    df = remove_duplicates(df)

    logger.info("Step 3: Handling missing values...")
    df = handle_missing_values(df, config)

    logger.info("Step 4: Fixing data types...")
    df = fix_data_types(df, config)

    logger.info("Step 5: Encoding categorical columns...")
    df, encoders = encode_categorical_columns(
        df, config, is_training=is_training
    )

    # Save encoders for prediction time
    if is_training:
        save_encoders(encoders)

    # Save processed data to disk
    processed_path = config["data"]["processed_data_path"]
    Path(processed_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(processed_path, index=False)
    logger.info(f"Processed data saved to: {processed_path}")

    # Print summary
    print_preprocessing_summary(df_before, df)

    logger.info("DATA PREPROCESSING COMPLETED SUCCESSFULLY ✓")
    logger.info("=" * 50)

    return df, encoders


# ── Standalone Run ────────────────────────────────────────────
if __name__ == "__main__":
    from src.data_ingestion   import run_data_ingestion
    from src.data_validation  import run_data_validation

    # Step 1: Load data
    df = run_data_ingestion()

    # Step 2: Validate
    passed = run_data_validation(df)
    if not passed:
        print("❌ Validation failed — fix errors first!")
        exit(1)

    # Step 3: Preprocess
    df_processed, encoders = run_data_preprocessing(df)

    print(f"\n✅ Preprocessing complete!")
    print(f"   Shape    : {df_processed.shape}")
    print(f"   Saved to : data/processed/processed_data.csv")
    print(f"   Encoders : models/encoders.pkl")