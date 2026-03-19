# tests/test_data_ingestion.py

import pytest
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def test_config_loads():
    """Config file load hoti hai."""
    import yaml
    with open("config/config.yaml") as f:
        config = yaml.safe_load(f)
    assert config is not None
    assert "data" in config
    assert "features" in config
    assert "model" in config


def test_target_column_in_config():
    """Target column config mein hai."""
    import yaml
    with open("config/config.yaml") as f:
        config = yaml.safe_load(f)
    assert config["features"]["target_column"] == "Return_Risk"


def test_numerical_features_defined():
    """Numerical features defined hain."""
    import yaml
    with open("config/config.yaml") as f:
        config = yaml.safe_load(f)
    assert len(config["features"]["numerical_features"]) > 0


def test_required_files_exist():
    """Saari required files exist karti hain."""
    required_files = [
        "config/config.yaml",
        "src/data_ingestion.py",
        "src/data_validation.py",
        "src/data_preprocessing.py",
        "src/feature_engineering.py",
        "src/model_training.py",
        "src/model_evaluation.py",
        "api/main.py",
        "api/schemas.py",
    ]
    for f in required_files:
        assert Path(f).exists(), f"Missing: {f}"


def test_models_folder_exists():
    """Models folder exist karta hai."""
    assert Path("models").exists()
    