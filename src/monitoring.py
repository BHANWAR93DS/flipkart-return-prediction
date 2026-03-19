# ============================================================
# src/monitoring.py
# PURPOSE : Monitor data drift and model performance
# ============================================================

import os
os.environ["GIT_PYTHON_REFRESH"] = "quiet"

import logging
import yaml
import json
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("Monitoring")


def load_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_reference_data(config: dict) -> pd.DataFrame:
    """Load reference data saved during training."""
    ref_path = config["data"]["reference_data_path"]
    if not Path(ref_path).exists():
        raise FileNotFoundError(f"Reference data not found: {ref_path}")
    df = pd.read_csv(ref_path)
    logger.info(f"Reference data loaded: {df.shape}")
    return df


def load_current_data(config: dict) -> pd.DataFrame:
    """Load current/production data for comparison."""
    proc_path = config["data"]["processed_data_path"]
    if not Path(proc_path).exists():
        raise FileNotFoundError(f"Processed data not found: {proc_path}")
    df = pd.read_csv(proc_path)
    df = df.tail(10000).reset_index(drop=True)
    logger.info(f"Current data loaded: {df.shape}")
    return df


def compute_drift_manually(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    config: dict
) -> tuple:
    """
    Compute data drift manually.
    PSI < 0.1  = No drift
    PSI < 0.2  = Slight drift
    PSI >= 0.2 = Significant drift
    """
    feature_cols = (
        config["features"]["numerical_features"] +
        config["features"]["categorical_features"]
    )

    cols = [c for c in feature_cols
            if c in reference.columns and c in current.columns]

    drift_results   = {}
    drifted_features = []

    for col in cols:
        try:
            ref_mean = float(reference[col].mean())
            cur_mean = float(current[col].mean())
            ref_std  = float(reference[col].std())

            if ref_std > 0:
                drift_score = float(abs(cur_mean - ref_mean) / ref_std)
            else:
                drift_score = 0.0

            is_drifted = bool(drift_score > 0.3)

            drift_results[col] = {
                "ref_mean":    round(ref_mean, 4),
                "cur_mean":    round(cur_mean, 4),
                "drift_score": round(drift_score, 4),
                "is_drifted":  int(is_drifted)
            }

            if is_drifted:
                drifted_features.append(col)

        except Exception as e:
            logger.warning(f"Could not compute drift for {col}: {e}")

    return drift_results, drifted_features


def compute_prediction_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    config: dict,
    model
) -> dict:
    """Compute prediction drift."""
    target = config["features"]["target_column"]

    try:
        from src.feature_engineering import (
            create_return_rate_feature,
            create_price_features,
            create_rating_features,
            create_quantity_features,
            create_age_features,
            create_risk_score
        )

        # Engineer features for current data
        cur_engineered = current.copy()
        cur_engineered = create_return_rate_feature(cur_engineered)
        cur_engineered = create_price_features(cur_engineered)
        cur_engineered = create_rating_features(cur_engineered)
        cur_engineered = create_quantity_features(cur_engineered)
        cur_engineered = create_age_features(cur_engineered)
        cur_engineered = create_risk_score(cur_engineered)

        # Prepare features
        ref_features = reference.drop(columns=[target], errors="ignore")
        cur_features = cur_engineered.drop(columns=[target], errors="ignore")

        # Align columns
        common_cols = [c for c in ref_features.columns
                       if c in cur_features.columns]
        ref_features = ref_features[common_cols]
        cur_features = cur_features[common_cols]

        # Predictions
        ref_preds = model.predict(ref_features)
        cur_preds = model.predict(cur_features)

        ref_return_rate = float(ref_preds.mean())
        cur_return_rate = float(cur_preds.mean())
        drift_pct       = float(abs(cur_return_rate - ref_return_rate))

        threshold  = config["monitoring"]["prediction_drift_threshold"]
        is_drifted = bool(drift_pct > threshold)

        return {
            "ref_return_rate":  round(ref_return_rate, 4),
            "cur_return_rate":  round(cur_return_rate, 4),
            "drift_percentage": round(drift_pct, 4),
            "threshold":        threshold,
            "is_drifted":       int(is_drifted)
        }

    except Exception as e:
        logger.error(f"Prediction drift error: {e}")
        return {
            "error":      str(e),
            "is_drifted": 0
        }


def make_serializable(obj):
    """Convert all values to JSON serializable types."""
    if isinstance(obj, bool):
        return int(obj)
    if isinstance(obj, (np.bool_, np.integer)):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [make_serializable(i) for i in obj]
    return obj


def print_monitoring_report(
    drift_results: dict,
    drifted_features: list,
    pred_drift: dict
) -> None:
    """Print a clear monitoring report."""

    print("\n" + "="*65)
    print("   MODEL MONITORING REPORT")
    print(f"   Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*65)

    # Data Drift
    print(f"\n  DATA DRIFT ANALYSIS:")
    print("  " + "-"*60)
    print(f"  {'Feature':<30} {'Ref Mean':>10} {'Cur Mean':>10} {'Drift':>8} {'Status':>10}")
    print("  " + "-"*60)

    for col, result in drift_results.items():
        status = "DRIFT" if result["is_drifted"] else "OK"
        flag   = "⚠️" if result["is_drifted"] else "✅"
        print(
            f"  {col:<30} "
            f"{result['ref_mean']:>10.4f} "
            f"{result['cur_mean']:>10.4f} "
            f"{result['drift_score']:>8.4f} "
            f"  {flag} {status}"
        )

    print(f"\n  Drifted Features : {len(drifted_features)}/{len(drift_results)}")
    if drifted_features:
        print(f"  Features         : {drifted_features}")

    # Prediction Drift
    print(f"\n  PREDICTION DRIFT ANALYSIS:")
    print("  " + "-"*60)
    if "error" not in pred_drift:
        status = "DRIFT" if pred_drift["is_drifted"] else "OK"
        flag   = "⚠️" if pred_drift["is_drifted"] else "✅"
        print(f"  Reference Return Rate : {pred_drift['ref_return_rate']:.4f}")
        print(f"  Current Return Rate   : {pred_drift['cur_return_rate']:.4f}")
        print(f"  Drift Percentage      : {pred_drift['drift_percentage']:.4f}")
        print(f"  Threshold             : {pred_drift['threshold']}")
        print(f"  Status                : {flag} {status}")
    else:
        print(f"  Error: {pred_drift['error']}")

    # Overall Health
    total_drifted  = len(drifted_features)
    total_features = len(drift_results)
    drift_pct      = total_drifted / total_features if total_features > 0 else 0

    print(f"\n  OVERALL HEALTH:")
    print("  " + "-"*60)
    if drift_pct < 0.2:
        print("  🟢 HEALTHY — Model performing within expected range")
        print("  No retraining needed at this time")
    elif drift_pct < 0.5:
        print("  🟡 WARNING — Some drift detected, monitor closely")
        print("  Consider retraining within next cycle")
    else:
        print("  🔴 CRITICAL — Significant drift detected!")
        print("  Immediate retraining recommended!")

    print("="*65 + "\n")


def run_monitoring(config_path: str = "config/config.yaml") -> dict:
    """Main monitoring function."""
    logger.info("="*50)
    logger.info("STARTING MODEL MONITORING")
    logger.info("="*50)

    config = load_config(config_path)

    # Load data
    reference_df = load_reference_data(config)
    current_df   = load_current_data(config)

    # Load model
    model_path = "models/model.pkl"
    model      = joblib.load(model_path)
    logger.info("Model loaded for monitoring")

    # Compute drift
    logger.info("Computing data drift...")
    drift_results, drifted_features = compute_drift_manually(
        reference_df, current_df, config
    )

    logger.info("Computing prediction drift...")
    pred_drift = compute_prediction_drift(
        reference_df, current_df, config, model
    )

    # Print report
    print_monitoring_report(drift_results, drifted_features, pred_drift)

    # Save report — make everything JSON serializable
    report = make_serializable({
        "timestamp":        datetime.now().isoformat(),
        "data_drift":       drift_results,
        "drifted_features": drifted_features,
        "prediction_drift": pred_drift,
        "health_status":    "healthy" if len(drifted_features) == 0
                            else "warning"
    })

    report_path = "logs/monitoring_report.json"
    Path("logs").mkdir(exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Report saved: {report_path}")

    logger.info("MONITORING COMPLETED SUCCESSFULLY ✓")
    logger.info("="*50)

    return report


if __name__ == "__main__":
    report  = run_monitoring()
    drifted = len(report["drifted_features"])
    total   = len(report["data_drift"])
    print(f"✅ Monitoring complete!")
    print(f"   Drifted features : {drifted}/{total}")
    print(f"   Health status    : {report['health_status']}")
    print(f"   Report saved     : logs/monitoring_report.json")