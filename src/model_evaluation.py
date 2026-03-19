# ============================================================
# src/model_evaluation.py
# PURPOSE : Evaluate trained model and log metrics to MLflow
# CALLED BY: pipeline/training_pipeline.py
# ============================================================

# ============================================================
# src/model_evaluation.py
# ============================================================

import os
os.environ["GIT_PYTHON_REFRESH"] = "quiet"

import json
import logging
import yaml
import mlflow
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    precision_score, recall_score,
    confusion_matrix, classification_report
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("ModelEvaluation")


def load_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def compute_metrics(y_true, y_pred, y_prob) -> dict:
    return {
        "accuracy":  round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1_score":  round(f1_score(y_true, y_pred, zero_division=0), 4),
        "roc_auc":   round(roc_auc_score(y_true, y_prob[:, 1]), 4),
    }


def print_confusion_matrix(y_true, y_pred) -> None:
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    print("\n  Confusion Matrix:")
    print("  " + "-" * 40)
    print("                  Predicted")
    print("                  No Return  Returned")
    print(f"  Actual No Return  {tn:>8,}  {fp:>8,}")
    print(f"  Actual Returned   {fn:>8,}  {tp:>8,}")
    print("  " + "-" * 40)
    print(f"\n  True Negatives  (TN): {tn:,}")
    print(f"  False Positives (FP): {fp:,}")
    print(f"  False Negatives (FN): {fn:,} ⚠️")
    print(f"  True Positives  (TP): {tp:,}")


def check_model_quality(metrics: dict, config: dict) -> bool:
    thresholds = config["model"]["evaluation"]
    passed = True
    print("\n  Quality Gate Checks:")
    print("  " + "-" * 55)
    checks = [
        ("accuracy", "min_accuracy"),
        ("f1_score", "min_f1_score"),
        ("roc_auc",  "min_roc_auc"),
    ]
    for metric_name, threshold_key in checks:
        actual    = metrics[metric_name]
        threshold = thresholds[threshold_key]
        status    = "✅ PASS" if actual >= threshold else "❌ FAIL"
        print(f"  {status} | {metric_name:<12} = {actual:.4f} (min: {threshold})")
        if actual < threshold:
            passed = False
    return passed


def print_evaluation_report(
    metrics: dict,
    y_true,
    y_pred,
    quality_passed: bool,
    config: dict
) -> None:
    print("\n" + "=" * 65)
    print("   MODEL EVALUATION REPORT")
    print("=" * 65)
    print(f"\n  {'Accuracy':<20} : {metrics['accuracy']:.4f}  ({metrics['accuracy']*100:.2f}%)")
    print(f"  {'Precision':<20} : {metrics['precision']:.4f}  ({metrics['precision']*100:.2f}%)")
    print(f"  {'Recall':<20} : {metrics['recall']:.4f}  ({metrics['recall']*100:.2f}%)")
    print(f"  {'F1 Score':<20} : {metrics['f1_score']:.4f}  ({metrics['f1_score']*100:.2f}%)")
    print(f"  {'ROC-AUC':<20} : {metrics['roc_auc']:.4f}  ({metrics['roc_auc']*100:.2f}%)")

    print_confusion_matrix(y_true, y_pred)

    print(f"\n  Classification Report:")
    print("  " + "-" * 55)
    report = classification_report(
        y_true, y_pred,
        target_names=["Not Returned", "Returned"]
    )
    for line in report.split("\n"):
        print(f"  {line}")

    print(f"\n  Quality Gate:")
    check_model_quality(metrics, config)

    print("\n" + "=" * 65)
    if quality_passed:
        print("  🎉 MODEL PASSED — Ready for deployment!")
    else:
        print("  ⚠️  Metrics low — proceeding for learning!")
    print("=" * 65 + "\n")


def run_model_evaluation(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    run_id: str,
    config_path: str = "config/config.yaml"
) -> dict:
    logger.info("=" * 50)
    logger.info("STARTING MODEL EVALUATION")
    logger.info("=" * 50)

    config = load_config(config_path)

    # Predictions
    logger.info("Making predictions on test set...")
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    # Metrics
    metrics = compute_metrics(y_test, y_pred, y_prob)
    logger.info(f"Metrics: {metrics}")

    # Quality check
    quality_passed = check_model_quality(metrics, config)

    # Full report — config pass karo
    print_evaluation_report(metrics, y_test, y_pred, quality_passed, config)

    # Log to MLflow
    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    with mlflow.start_run(run_id=run_id):
        mlflow.log_metrics(metrics)
        mlflow.log_metric("quality_passed", int(quality_passed))
    logger.info(f"Metrics logged to MLflow: {run_id}")

    # Save metrics
    metrics_path = "models/metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics saved: {metrics_path}")

    logger.info("MODEL EVALUATION COMPLETED ✓")
    logger.info("=" * 50)

    return metrics


if __name__ == "__main__":
    from src.data_ingestion      import run_data_ingestion
    from src.data_validation     import run_data_validation
    from src.data_preprocessing  import run_data_preprocessing
    from src.feature_engineering import run_feature_engineering
    from src.model_training      import run_model_training

    df = run_data_ingestion()
    if not run_data_validation(df):
        exit(1)
    df_processed, encoders = run_data_preprocessing(df)
    X_train, X_test, y_train, y_test = run_feature_engineering(df_processed)
    model, run_id, X_train_s, X_test_s = run_model_training(
        X_train, X_test, y_train, y_test
    )
    metrics = run_model_evaluation(model, X_test_s, y_test, run_id)

    print(f"\n✅ Evaluation complete!")
    print(f"   Accuracy : {metrics['accuracy']:.4f}")
    print(f"   F1 Score : {metrics['f1_score']:.4f}")
    print(f"   ROC-AUC  : {metrics['roc_auc']:.4f}")