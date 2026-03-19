# ============================================================
# src/model_training.py
# PURPOSE : Train ML model and log everything to MLflow
# CALLED BY: pipeline/training_pipeline.py
# ============================================================
import os
os.environ["GIT_PYTHON_REFRESH"] = "quiet"


import logging
import yaml
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler


# ── Logger ────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("ModelTraining")


# ── Config Loader ─────────────────────────────────────────────
def load_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


# ── MLflow Setup ──────────────────────────────────────────────
def setup_mlflow(config: dict) -> None:
    """
    Configure MLflow tracking.

    WHY: MLflow needs to know WHERE to save experiment data.
         We use local 'mlruns' folder — in production this
         would be a remote server (AWS S3, Azure Blob, etc.)
    """
    tracking_uri    = config["mlflow"]["tracking_uri"]
    experiment_name = config["mlflow"]["experiment_name"]

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    logger.info(f"MLflow tracking URI : {tracking_uri}")
    logger.info(f"MLflow experiment   : {experiment_name}")


# ── Feature Scaling ───────────────────────────────────────────
def scale_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    config: dict
) -> tuple[np.ndarray, np.ndarray, StandardScaler]:
    """
    Scale numerical features using StandardScaler.

    WHY: Random Forest doesn't need scaling, but we add it
         for future model compatibility (Logistic Regression,
         SVM, Neural Networks all need scaled features).
         This makes the pipeline model-agnostic.

    IMPORTANT: Fit scaler ONLY on train data, then transform
               both train and test — never fit on test data!
    """
    numerical_features = config["features"]["numerical_features"]

    # Only scale columns that exist in the DataFrame
    cols_to_scale = [
        c for c in numerical_features if c in X_train.columns
    ]

    scaler = StandardScaler()

    X_train_scaled = X_train.copy()
    X_test_scaled  = X_test.copy()

    # Fit on train ONLY, transform both
    X_train_scaled[cols_to_scale] = scaler.fit_transform(
        X_train[cols_to_scale]
    )
    X_test_scaled[cols_to_scale] = scaler.transform(
        X_test[cols_to_scale]
    )

    # Save scaler for prediction time
    scaler_path = "models/scaler.pkl"
    Path(scaler_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, scaler_path)

    logger.info(f"Scaler fitted on {len(cols_to_scale)} features")
    logger.info(f"Scaler saved to: {scaler_path}")

    return X_train_scaled, X_test_scaled, scaler


# ── Model Trainer ─────────────────────────────────────────────
def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config: dict
) -> RandomForestClassifier:
    """
    Train RandomForest model with config-driven hyperparameters.

    WHY RandomForest:
    - Handles class imbalance with class_weight='balanced'
    - No need for feature scaling
    - Gives feature importance scores
    - Works well with mixed numerical + categorical data
    - Less prone to overfitting than single Decision Tree
    """
    params = config["model"]["hyperparameters"]

    logger.info(f"Training RandomForestClassifier...")
    logger.info(f"  Parameters: {params}")
    logger.info(f"  Training samples: {len(X_train):,}")

    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)

    logger.info("Model training complete!")
    return model


# ── Feature Importance Logger ─────────────────────────────────
def get_feature_importance(
    model: RandomForestClassifier,
    feature_names: list
) -> pd.DataFrame:
    """
    Get and display feature importance scores.

    WHY: Feature importance tells us which features the model
         found most useful. This helps us:
         - Remove useless features (reduce complexity)
         - Validate that important features make business sense
         - Explain model decisions to stakeholders
    """
    importance_df = pd.DataFrame({
        "feature":   feature_names,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)

    print("\n" + "=" * 55)
    print("   FEATURE IMPORTANCE (Top 15)")
    print("=" * 55)
    for _, row in importance_df.head(15).iterrows():
        bar   = "█" * int(row["importance"] * 200)
        print(f"  {row['feature']:<28} {row['importance']:.4f} {bar}")
    print("=" * 55 + "\n")

    return importance_df


# ── MLflow Run ────────────────────────────────────────────────
def run_mlflow_training(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    config: dict
) -> tuple[RandomForestClassifier, str]:
    """
    Train model inside MLflow run — logs everything automatically.

    WHAT GETS LOGGED TO MLFLOW:
    - Parameters  : all hyperparameters
    - Metrics     : accuracy, f1, roc_auc (logged in evaluation step)
    - Artifacts   : trained model, feature importance CSV
    - Tags        : model type, dataset size, python version

    RETURNS: (trained model, mlflow run_id)
    """
    setup_mlflow(config)

    with mlflow.start_run(run_name="RandomForest_v1") as run:
        run_id = run.info.run_id
        logger.info(f"MLflow Run ID: {run_id}")

        # ── Log Tags ─────────────────────────────────────────
        mlflow.set_tags({
            "model_type":    "RandomForestClassifier",
            "dataset":       "flipkart_returns",
            "train_samples": str(len(X_train)),
            "test_samples":  str(len(X_test)),
            "features":      str(X_train.shape[1]),
            "engineer":      "MLOps_Pipeline_v1"
        })

        # ── Log Hyperparameters ───────────────────────────────
        params = config["model"]["hyperparameters"]
        mlflow.log_params(params)
        logger.info(f"Logged {len(params)} parameters to MLflow")

        # ── Log Dataset Info ──────────────────────────────────
        mlflow.log_params({
            "train_size":    len(X_train),
            "test_size":     len(X_test),
            "n_features":    X_train.shape[1],
            "class_0_ratio": round(
                (y_train == 0).sum() / len(y_train), 4
            ),
            "class_1_ratio": round(
                (y_train == 1).sum() / len(y_train), 4
            ),
        })

        # ── Scale Features ────────────────────────────────────
        logger.info("Scaling features...")
        X_train_scaled, X_test_scaled, scaler = scale_features(
            X_train, X_test, config
        )

        # ── Train Model ───────────────────────────────────────
        logger.info("Training model...")
        model = train_model(X_train_scaled, y_train, config)

        # ── Feature Importance ────────────────────────────────
        importance_df = get_feature_importance(
            model, list(X_train.columns)
        )

        # Save feature importance as CSV artifact
        importance_path = "models/feature_importance.csv"
        importance_df.to_csv(importance_path, index=False)
        mlflow.log_artifact(importance_path)

        # ── Save & Log Model ──────────────────────────────────
        model_path = "models/model.pkl"
        joblib.dump(model, model_path)

        # Log model to MLflow Model Registry
        mlflow.sklearn.log_model(
        sk_model              = model,
        name                  = config["mlflow"]["registered_model_name"],
        input_example         = X_train_scaled.head(1)
        )
        logger.info(f"Model saved to: {model_path}")
        logger.info(
            f"Model registered as: "
            f"{config['mlflow']['registered_model_name']}"
        )

        # ── Log Scaler as Artifact ────────────────────────────
        mlflow.log_artifact("models/scaler.pkl")
        mlflow.log_artifact("models/encoders.pkl")

        print("\n" + "=" * 55)
        print("   MLFLOW TRACKING SUMMARY")
        print("=" * 55)
        print(f"  Run ID      : {run_id}")
        print(f"  Experiment  : {config['mlflow']['experiment_name']}")
        print(f"  Model       : RandomForestClassifier")
        print(f"  Parameters  : {len(params)} logged")
        print(f"  Artifacts   : model.pkl, scaler.pkl,")
        print(f"                encoders.pkl, feature_importance.csv")
        print(f"  Registry    : {config['mlflow']['registered_model_name']}")
        print("=" * 55 + "\n")

    return model, run_id, X_train_scaled, X_test_scaled


# ── Main Entry Point ──────────────────────────────────────────
def run_model_training(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    config_path: str = "config/config.yaml"
) -> tuple[RandomForestClassifier, str, pd.DataFrame, pd.DataFrame]:
    """
    Main function called by training pipeline.

    RETURNS: (model, run_id, X_train_scaled, X_test_scaled)
    """
    logger.info("=" * 50)
    logger.info("STARTING MODEL TRAINING")
    logger.info("=" * 50)

    config = load_config(config_path)

    model, run_id, X_train_scaled, X_test_scaled = run_mlflow_training(
        X_train, X_test, y_train, y_test, config
    )

    logger.info("MODEL TRAINING COMPLETED SUCCESSFULLY ✓")
    logger.info("=" * 50)

    return model, run_id, X_train_scaled, X_test_scaled


# ── Standalone Run ────────────────────────────────────────────
if __name__ == "__main__":
    from src.data_ingestion      import run_data_ingestion
    from src.data_validation     import run_data_validation
    from src.data_preprocessing  import run_data_preprocessing
    from src.feature_engineering import run_feature_engineering

    # Run full pipeline up to training
    df = run_data_ingestion()

    passed = run_data_validation(df)
    if not passed:
        exit(1)

    df_processed, encoders = run_data_preprocessing(df)
    X_train, X_test, y_train, y_test = run_feature_engineering(
        df_processed
    )

    # Train model
    model, run_id, X_train_s, X_test_s = run_model_training(
        X_train, X_test, y_train, y_test
    )

    print(f"\n✅ Model Training complete!")
    print(f"   Run ID  : {run_id}")
    print(f"   Model   : models/model.pkl")
    print(f"   Scaler  : models/scaler.pkl")
    print(f"\n   View MLflow UI:")
    print(f"   Run: mlflow ui --port 5000")
    print(f"   Open: http://localhost:5000")