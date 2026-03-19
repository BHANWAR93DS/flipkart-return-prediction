# ============================================================
# src/data_ingestion.py
# PURPOSE : Load real Flipkart CSV and report data quality
# CALLED BY: pipeline/training_pipeline.py
# ============================================================

import logging
import yaml
import pandas as pd
from pathlib import Path

# ── Logger ────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("DataIngestion")


# ── Config Loader ─────────────────────────────────────────────
def load_config(config_path: str = "config/config.yaml") -> dict:
    """Load central config.yaml file."""
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(
            f"Config file not found at: {config_path}\n"
            f"Make sure you are running from the project ROOT folder.\n"
            f"Current directory: {Path.cwd()}"
        )
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    logger.info(f"Config loaded from: {config_path}")
    return config


# ── Raw Data Loader ───────────────────────────────────────────
def load_raw_data(input_path: str) -> pd.DataFrame:
    """
    Load CSV file from disk into a DataFrame.

    WHY: We load raw data exactly as-is — no modifications.
         All changes happen in preprocessing, never here.
    """
    data_file = Path(input_path)

    if not data_file.exists():
        raise FileNotFoundError(
            f"\n❌ Data file not found: {input_path}\n"
            f"   Please copy your CSV file to: data/raw/flipkart_returns.csv\n"
            f"   Current working directory: {Path.cwd()}"
        )

    logger.info(f"Loading data from: {input_path}")
    df = pd.read_csv(input_path)
    logger.info(f"Data loaded successfully!")
    logger.info(f"  Shape : {df.shape[0]:,} rows × {df.shape[1]} columns")
    logger.info(f"  Memory: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    return df


# ── Data Summary Reporter ─────────────────────────────────────
def print_data_summary(df: pd.DataFrame) -> None:
    """
    Print a full quality overview of the loaded dataset.

    WHY: Engineers must LOOK at the data before building
         any pipeline. This catches obvious issues early.
    """
    print("\n" + "=" * 65)
    print("   FLIPKART DATASET — INGESTION SUMMARY REPORT")
    print("=" * 65)

    # Basic shape
    print(f"\n  Total Rows    : {df.shape[0]:>10,}")
    print(f"  Total Columns : {df.shape[1]:>10}")
    total_cells = df.shape[0] * df.shape[1]
    total_nulls = df.isnull().sum().sum()
    print(f"  Total Cells   : {total_cells:>10,}")
    print(f"  Total Nulls   : {total_nulls:>10,}  ({total_nulls/total_cells:.2%})")

    # Target column check
    if "Return_Risk" in df.columns:
        print(f"\n  Target Column : Return_Risk")
        vc = df["Return_Risk"].value_counts()
        for val, cnt in vc.items():
            print(f"    Class {val} → {cnt:,} rows ({cnt/len(df):.2%})")
    else:
        print("\n  ⚠️  WARNING: 'Return_Risk' column NOT found!")
        print(f"  Columns present: {list(df.columns)}")

    # Null counts per column
    print(f"\n  {'Column':<35} {'Type':<12} {'Nulls':>8} {'Null%':>8}")
    print("  " + "-" * 63)
    for col in df.columns:
        dtype  = str(df[col].dtype)
        nulls  = df[col].isnull().sum()
        pct    = nulls / len(df) * 100
        flag   = " ← ⚠️ HIGH" if pct > 5 else ""
        print(f"  {col:<35} {dtype:<12} {nulls:>8,} {pct:>7.1f}%{flag}")

    # Numerical columns quick stats
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if num_cols:
        print(f"\n  Numerical Columns Quick Stats:")
        print("  " + "-" * 63)
        print(df[num_cols].describe().round(2).to_string())

    # Categorical columns
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    if cat_cols:
        print(f"\n  Categorical Columns — Unique Value Counts:")
        print("  " + "-" * 63)
        for col in cat_cols:
            unique_count = df[col].nunique()
            top_val      = df[col].value_counts().index[0]
            print(f"  {col:<35} {unique_count:>5} unique  | top: {top_val}")

    print("\n" + "=" * 65 + "\n")


# ── Main Entry Point ──────────────────────────────────────────
def run_data_ingestion(config_path: str = "config/config.yaml") -> pd.DataFrame:
    """
    Main function called by training pipeline.

    RETURNS: Raw DataFrame — untouched, exactly as loaded from CSV.
    """
    logger.info("=" * 50)
    logger.info("STARTING DATA INGESTION")
    logger.info("=" * 50)

    # Step 1: Load config
    config = load_config(config_path)

    # Step 2: Get file path from config (no hardcoding!)
    raw_data_path = config["data"]["raw_data_path"]

    # Step 3: Load the CSV
    df = load_raw_data(raw_data_path)

    # Step 4: Print summary for engineer to review
    print_data_summary(df)

    logger.info("DATA INGESTION COMPLETED SUCCESSFULLY ✓")
    logger.info("=" * 50)

    return df


# ── Standalone Run ────────────────────────────────────────────
if __name__ == "__main__":
    df = run_data_ingestion()
    print(f"✅ Data ingestion complete!")
    print(f"   Rows    : {df.shape[0]:,}")
    print(f"   Columns : {df.shape[1]}")
    print(f"\n   Column names found:")
    for i, col in enumerate(df.columns, 1):
        print(f"   {i:>2}. {col}")