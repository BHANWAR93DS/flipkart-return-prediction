# ============================================================
# src/data_validation.py
# PURPOSE : Validate raw data quality before preprocessing
# CALLED BY: pipeline/training_pipeline.py
# ============================================================

import logging
import yaml
import pandas as pd
import numpy as np
from pathlib import Path

# ── Logger ────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("DataValidation")


# ── Config Loader ─────────────────────────────────────────────
def load_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


# ── Validation Check Functions ────────────────────────────────

def check_required_columns(df: pd.DataFrame, config: dict) -> dict:
    """
    Check all required columns are present in the dataset.
    WHY: If a column is missing, every downstream step will crash.
    """
    result = {"check": "required_columns", "passed": True, "details": []}

    required_cols = (
        config["features"]["numerical_features"] +
        config["features"]["categorical_features"] +
        [config["features"]["target_column"]]
    )

    missing = [col for col in required_cols if col not in df.columns]

    if missing:
        result["passed"] = False
        result["details"] = f"MISSING COLUMNS: {missing}"
        logger.error(f"❌ Required columns missing: {missing}")
    else:
        result["details"] = f"All {len(required_cols)} required columns present"
        logger.info(f"✅ Required columns check PASSED")

    return result


def check_data_types(df: pd.DataFrame, config: dict) -> dict:
    """
    Check numerical columns are actually numeric.
    WHY: Sometimes numbers get loaded as strings — 
         sklearn will crash if you pass string to a model.
    """
    result = {"check": "data_types", "passed": True, "details": []}
    issues = []

    for col in config["features"]["numerical_features"]:
        if col in df.columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                issues.append(f"{col} should be numeric but is {df[col].dtype}")

    if issues:
        result["passed"] = False
        result["details"] = issues
        logger.error(f"❌ Data type issues: {issues}")
    else:
        result["details"] = "All numerical columns have correct types"
        logger.info(f"✅ Data types check PASSED")

    return result


def check_null_values(df: pd.DataFrame, config: dict) -> dict:
    """
    Check null values are within acceptable threshold.
    WHY: Too many nulls means data pipeline is broken upstream.
         We allow max 10% nulls per column.
    """
    result = {"check": "null_values", "passed": True, "details": []}
    issues = []
    MAX_NULL_PCT = 10.0  # Maximum allowed null percentage

    feature_cols = (
        config["features"]["numerical_features"] +
        config["features"]["categorical_features"]
    )

    for col in feature_cols:
        if col in df.columns:
            null_pct = df[col].isnull().sum() / len(df) * 100
            if null_pct > MAX_NULL_PCT:
                issues.append(f"{col}: {null_pct:.1f}% nulls (max allowed: {MAX_NULL_PCT}%)")

    if issues:
        result["passed"] = False
        result["details"] = issues
        logger.error(f"❌ High null values: {issues}")
    else:
        result["details"] = f"All columns within {MAX_NULL_PCT}% null threshold"
        logger.info(f"✅ Null values check PASSED")

    return result


def check_target_column(df: pd.DataFrame, config: dict) -> dict:
    """
    Check target column has only valid values (0 and 1).
    WHY: If target has unexpected values, model training will fail
         or produce wrong results silently.
    """
    result = {"check": "target_column", "passed": True, "details": []}
    target = config["features"]["target_column"]

    if target not in df.columns:
        result["passed"] = False
        result["details"] = f"Target column '{target}' not found!"
        logger.error(f"❌ Target column missing!")
        return result

    unique_vals = sorted(df[target].unique().tolist())
    expected_vals = [0, 1]

    if unique_vals != expected_vals:
        result["passed"] = False
        result["details"] = f"Expected [0,1] but found: {unique_vals}"
        logger.error(f"❌ Target column has unexpected values: {unique_vals}")
    else:
        class_counts = df[target].value_counts()
        class_0_pct = class_counts.get(0, 0) / len(df) * 100
        class_1_pct = class_counts.get(1, 0) / len(df) * 100
        result["details"] = (
            f"Valid binary target | "
            f"Class 0: {class_0_pct:.1f}% | "
            f"Class 1: {class_1_pct:.1f}%"
        )
        logger.info(f"✅ Target column check PASSED")

    return result


def check_class_imbalance(df: pd.DataFrame, config: dict) -> dict:
    """
    Check and WARN if class imbalance is severe.
    WHY: 85/15 split means model will ignore minority class.
         We warn but don't fail — we handle it with class_weight.
    """
    result = {"check": "class_imbalance", "passed": True, "details": []}
    target = config["features"]["target_column"]

    if target not in df.columns:
        result["passed"] = False
        result["details"] = "Target column not found"
        return result

    minority_pct = df[target].value_counts(normalize=True).min() * 100

    if minority_pct < 5.0:
        result["passed"] = False
        result["details"] = (
            f"SEVERE imbalance: minority class only {minority_pct:.1f}%. "
            f"Consider oversampling."
        )
        logger.warning(f"⚠️ Severe class imbalance: {minority_pct:.1f}%")
    elif minority_pct < 20.0:
        result["details"] = (
            f"Moderate imbalance: minority class {minority_pct:.1f}%. "
            f"Using class_weight='balanced' to handle this."
        )
        logger.warning(f"⚠️ Moderate class imbalance: {minority_pct:.1f}% — handled with class_weight")
    else:
        result["details"] = f"Balanced classes: minority {minority_pct:.1f}%"
        logger.info(f"✅ Class balance check PASSED")

    return result


def check_data_volume(df: pd.DataFrame) -> dict:
    """
    Check dataset has enough rows for training.
    WHY: Too few rows = model can't learn patterns.
         Minimum 1000 rows required for meaningful ML.
    """
    result = {"check": "data_volume", "passed": True, "details": []}
    MIN_ROWS = 1000

    if len(df) < MIN_ROWS:
        result["passed"] = False
        result["details"] = f"Only {len(df)} rows. Minimum required: {MIN_ROWS}"
        logger.error(f"❌ Insufficient data: {len(df)} rows")
    else:
        result["details"] = f"{len(df):,} rows — sufficient for training"
        logger.info(f"✅ Data volume check PASSED: {len(df):,} rows")

    return result


def check_duplicate_rows(df: pd.DataFrame) -> dict:
    """
    Check for duplicate rows in the dataset.
    WHY: Duplicates can cause data leakage — same row in 
         both train and test set = artificially high accuracy.
    """
    result = {"check": "duplicate_rows", "passed": True, "details": []}
    
    dup_count = df.duplicated().sum()
    dup_pct = dup_count / len(df) * 100

    if dup_pct > 5.0:
        result["passed"] = False
        result["details"] = f"{dup_count:,} duplicate rows ({dup_pct:.1f}%) — too many!"
        logger.error(f"❌ Too many duplicates: {dup_count:,} rows")
    elif dup_count > 0:
        result["details"] = f"{dup_count:,} duplicate rows ({dup_pct:.1f}%) — acceptable"
        logger.warning(f"⚠️ Found {dup_count:,} duplicate rows — will be removed in preprocessing")
    else:
        result["details"] = "No duplicate rows found"
        logger.info(f"✅ Duplicate check PASSED")

    return result


def check_numerical_ranges(df: pd.DataFrame, config: dict) -> dict:
    """
    Check numerical columns for extreme outliers.
    WHY: Outliers like age=999 or price=-1 are data errors
         that will skew the model.
    """
    result = {"check": "numerical_ranges", "passed": True, "details": []}

    # Define valid ranges for our specific columns
    valid_ranges = {
        "ProductPrice":             (0, 200000),
        "CustomerAge":              (18, 100),
        "Quantity":                 (1, 100),
        "DiscountApplied":          (0, 100),
        "CustomerPurchaseHistory":  (0, 10000),
        "CustomerReturnHistory":    (0, 10000),
        "ProductRating":            (1, 5),
    }

    issues = []
    for col, (min_val, max_val) in valid_ranges.items():
        if col in df.columns:
            out_of_range = df[
                (df[col] < min_val) | (df[col] > max_val)
            ].shape[0]
            if out_of_range > 0:
                pct = out_of_range / len(df) * 100
                issues.append(
                    f"{col}: {out_of_range:,} values outside "
                    f"[{min_val}, {max_val}] ({pct:.2f}%)"
                )

    if issues:
        result["passed"] = False
        result["details"] = issues
        logger.warning(f"⚠️ Range issues found: {issues}")
    else:
        result["details"] = "All numerical columns within valid ranges"
        logger.info(f"✅ Numerical ranges check PASSED")

    return result


# ── Main Validation Runner ────────────────────────────────────
def run_data_validation(
    df: pd.DataFrame,
    config_path: str = "config/config.yaml"
) -> bool:
    """
    Run all validation checks and print a full report.

    RETURNS: True if all CRITICAL checks pass, False otherwise.
             Pipeline should STOP if this returns False.
    """
    logger.info("=" * 50)
    logger.info("STARTING DATA VALIDATION")
    logger.info("=" * 50)

    config = load_config(config_path)

    # ── Run All Checks ────────────────────────────────────────
    results = [
        check_required_columns(df, config),
        check_data_types(df, config),
        check_null_values(df, config),
        check_target_column(df, config),
        check_class_imbalance(df, config),
        check_data_volume(df),
        check_duplicate_rows(df),
        check_numerical_ranges(df, config),
    ]

    # ── Print Validation Report ───────────────────────────────
    print("\n" + "=" * 65)
    print("   DATA VALIDATION REPORT")
    print("=" * 65)

    critical_failed = 0
    warning_count   = 0

    # These checks MUST pass — pipeline stops if they fail
    critical_checks = [
        "required_columns",
        "data_types",
        "target_column",
        "data_volume"
    ]

    for r in results:
        check_name = r["check"]
        passed     = r["passed"]
        details    = r["details"]
        is_critical = check_name in critical_checks

        if passed:
            status = "✅ PASSED"
        elif is_critical:
            status = "❌ FAILED (CRITICAL)"
            critical_failed += 1
        else:
            status = "⚠️  WARNING"
            warning_count += 1

        print(f"\n  [{status}] {check_name.upper().replace('_', ' ')}")
        if isinstance(details, list):
            for d in details:
                print(f"    → {d}")
        else:
            print(f"    → {details}")

    # ── Final Summary ─────────────────────────────────────────
    print("\n" + "=" * 65)
    print(f"  SUMMARY:")
    print(f"  Total Checks  : {len(results)}")
    print(f"  Passed        : {sum(1 for r in results if r['passed'])}")
    print(f"  Warnings      : {warning_count}")
    print(f"  Critical Fails: {critical_failed}")

    if critical_failed == 0:
        print(f"\n  🎉 VALIDATION PASSED — Pipeline can proceed!")
        logger.info("DATA VALIDATION PASSED ✓")
        all_passed = True
    else:
        print(f"\n  🛑 VALIDATION FAILED — Fix issues before proceeding!")
        logger.error("DATA VALIDATION FAILED ✗")
        all_passed = False

    print("=" * 65 + "\n")

    return all_passed


# ── Standalone Run ────────────────────────────────────────────
if __name__ == "__main__":
    from src.data_ingestion import run_data_ingestion
    df = run_data_ingestion()
    passed = run_data_validation(df)
    if passed:
        print("✅ Data is valid — ready for preprocessing!")
    else:
        print("❌ Fix validation errors before proceeding!")