"""
pytest configuration and shared fixtures.
"""
import sys
import os
from pathlib import Path

# Add project root to sys.path so src.* imports work from tests/
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

import pytest


@pytest.fixture
def project_root() -> Path:
    """Return the project root directory."""
    return PROJECT_ROOT


@pytest.fixture
def cmapss_data_dir() -> Path:
    """Return path to CMAPSSData directory."""
    return PROJECT_ROOT.parent / "CMAPSSData"


@pytest.fixture
def config_path() -> Path:
    """Return path to settings.yaml."""
    return PROJECT_ROOT / "config" / "settings.yaml"


@pytest.fixture
def sample_config() -> dict:
    """Return a minimal config dict for testing."""
    return {
        "dataset": {
            "column_names": [
                "engine_id", "cycle",
                "op_setting_1", "op_setting_2", "op_setting_3",
                *[f"sensor_{i}" for i in range(1, 22)],
            ]
        },
        "features": {
            "selected_sensors": ["sensor_2", "sensor_3", "sensor_4", "sensor_7"],
            "rolling_windows": [5, 10],
            "include_op_settings": True,
            "trend_window": 5,
        },
        "rul": {"cap_value": 125, "use_capped": True},
        "split": {"validation_fraction": 0.2, "random_seed": 42},
        "triage": {
            "critical_rul_threshold": 15,
            "warning_rul_threshold": 40,
            "anomaly_score_threshold": -0.05,
        },
        "anomaly": {
            "contamination": 0.05,
            "n_estimators": 10,
            "random_state": 42,
            "threshold_percentile": 10,
        },
        "mlflow": {
            "experiment_name": "test_experiment",
            "model_name": "test_rul_predictor",
        },
        "paths": {
            "artifacts_dir": "artifacts/models",
            "database": "data/test_maintenance.db",
            "mlflow_tracking_uri": "mlruns",
        },
        "promotion": {"min_mae_improvement_fraction": 0.01},
    }
