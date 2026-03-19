# ============================================================
# pipeline/training_pipeline.py
# PURPOSE : End-to-end training pipeline — runs all steps
# RUN WITH: python pipeline/training_pipeline.py
# ============================================================

import os
os.environ["GIT_PYTHON_REFRESH"] = "quiet"

import logging
import yaml
from pathlib import Path
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("TrainingPipeline")


def load_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def run_training_pipeline(config_path: str = "config/config.yaml"):
    """
    Complete end-to-end training pipeline.

    Steps:
    1. Data Ingestion
    2. Data Validation
    3. Data Preprocessing
    4. Feature Engineering
    5. Model Training
    6. Model Evaluation
    """

    start_time = datetime.now()

    print("\n" + "="*65)
    print("   FLIPKART RETURN PREDICTION — TRAINING PIPELINE")
    print("="*65)
    print(f"   Started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*65 + "\n")

    # ── Step 1: Data Ingestion ────────────────────────────
    logger.info("STEP 1/6 — Data Ingestion")
    from src.data_ingestion import run_data_ingestion
    df = run_data_ingestion(config_path)
    logger.info(f"✓ Data loaded: {df.shape[0]:,} rows")

    # ── Step 2: Data Validation ───────────────────────────
    logger.info("STEP 2/6 — Data Validation")
    from src.data_validation import run_data_validation
    passed = run_data_validation(df, config_path)
    if not passed:
        logger.error("❌ Validation failed — stopping pipeline!")
        raise ValueError("Data validation failed!")
    logger.info("✓ Validation passed: 8/8 checks")

    # ── Step 3: Data Preprocessing ────────────────────────
    logger.info("STEP 3/6 — Data Preprocessing")
    from src.data_preprocessing import run_data_preprocessing
    df_processed, encoders = run_data_preprocessing(df, config_path)
    logger.info(f"✓ Preprocessing done: {df_processed.shape}")

    # ── Step 4: Feature Engineering ───────────────────────
    logger.info("STEP 4/6 — Feature Engineering")
    from src.feature_engineering import run_feature_engineering
    X_train, X_test, y_train, y_test = run_feature_engineering(
        df_processed, config_path
    )
    logger.info(f"✓ Features ready: {X_train.shape[1]} features")

    # ── Step 5: Model Training ────────────────────────────
    logger.info("STEP 5/6 — Model Training")
    from src.model_training import run_model_training
    model, run_id, X_train_s, X_test_s = run_model_training(
        X_train, X_test, y_train, y_test, config_path
    )
    logger.info(f"✓ Model trained: Run ID = {run_id}")

    # ── Step 6: Model Evaluation ──────────────────────────
    logger.info("STEP 6/6 — Model Evaluation")
    from src.model_evaluation import run_model_evaluation
    metrics = run_model_evaluation(
        model, X_test_s, y_test, run_id, config_path
    )
    logger.info(f"✓ Evaluation done: {metrics}")

    # ── Pipeline Complete ─────────────────────────────────
    end_time  = datetime.now()
    duration  = end_time - start_time
    minutes   = int(duration.total_seconds() // 60)
    seconds   = int(duration.total_seconds() % 60)

    print("\n" + "="*65)
    print("   PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*65)
    print(f"   Duration  : {minutes}m {seconds}s")
    print(f"   Run ID    : {run_id}")
    print(f"   Accuracy  : {metrics['accuracy']:.4f}")
    print(f"   F1 Score  : {metrics['f1_score']:.4f}")
    print(f"   ROC-AUC   : {metrics['roc_auc']:.4f}")
    print(f"\n   Files saved:")
    print(f"   → models/model.pkl")
    print(f"   → models/scaler.pkl")
    print(f"   → models/encoders.pkl")
    print(f"   → models/metrics.json")
    print(f"   → data/processed/processed_data.csv")
    print(f"\n   MLflow UI:")
    print(f"   → mlflow ui --port 5000")
    print(f"   → http://localhost:5000")
    print("="*65 + "\n")

    return metrics, run_id


if __name__ == "__main__":
    metrics, run_id = run_training_pipeline()