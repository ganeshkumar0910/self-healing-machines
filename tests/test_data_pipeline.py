"""Integration and unit tests for CMAPSS data loading, validation, RUL, and features."""

from pathlib import Path
import sys

# Ensure project root is available on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import pytest

from src.data.loader import get_project_root, load_raw_cmapss, load_rul_labels
from src.data.preprocessor import DataPreprocessor
from src.data.rul_calculator import compute_test_rul, compute_train_rul
from src.data.splitter import engine_wise_split
from src.data.validator import run_full_validation, validate_quality
from src.features.engineer import FeatureEngineer


def get_cmapss_file(filename: str) -> Path:
    """Helper to locate CMAPSS data files in expected directories."""
    root = get_project_root()
    candidates = [
        root / "../CMAPSSData" / filename,
        root.parent / "CMAPSSData" / filename,
        Path("e:/Teleport-project/CMAPSSData") / filename,
    ]
    for c in candidates:
        resolved = c.resolve()
        if resolved.is_file():
            return resolved
    return candidates[0]


def test_column_count() -> None:
    """Verify loader reads space-separated CMAPSS files into exactly 26 named columns."""
    train_path = get_cmapss_file("train_FD001.txt")
    if not train_path.is_file():
        pytest.skip(f"train_FD001.txt not found at {train_path}")

    df = load_raw_cmapss(train_path, dataset_id="FD001")
    assert df.shape[1] == 26
    expected_prefixes = ["engine_id", "cycle", "op_setting_1", "op_setting_2", "op_setting_3"]
    for col in expected_prefixes:
        assert col in df.columns

    for i in range(1, 22):
        assert f"sensor_{i}" in df.columns


def test_rul_computation_train() -> None:
    """Verify that uncapped training RUL ends at exactly 0 cycles at failure."""
    train_path = get_cmapss_file("train_FD001.txt")
    if not train_path.is_file():
        pytest.skip("train_FD001.txt not found")

    df = load_raw_cmapss(train_path, dataset_id="FD001")
    df_rul = compute_train_rul(df, cap_value=None)

    assert "RUL" in df_rul.columns
    # The last cycle of each run-to-failure engine must have RUL = 0
    last_cycles = df_rul.groupby("engine_id").last()
    assert (last_cycles["RUL"] == 0.0).all()


def test_rul_never_negative() -> None:
    """Verify that computed training RUL values are strictly non-negative."""
    train_path = get_cmapss_file("train_FD001.txt")
    if not train_path.is_file():
        pytest.skip("train_FD001.txt not found")

    df = load_raw_cmapss(train_path, dataset_id="FD001")
    df_rul = compute_train_rul(df, cap_value=125)
    assert (df_rul["RUL"] >= 0.0).all()


def test_test_rul_alignment() -> None:
    """Verify test RUL alignment matches ground truth (first value for engine 1 = 112)."""
    test_path = get_cmapss_file("test_FD001.txt")
    rul_path = get_cmapss_file("RUL_FD001.txt")

    if not test_path.is_file() or not rul_path.is_file():
        pytest.skip("test_FD001.txt or RUL_FD001.txt not found")

    df_test = load_raw_cmapss(test_path, dataset_id="FD001")
    rul_labels = load_rul_labels(rul_path)

    # In CMAPSS FD001, the first entry in RUL_FD001.txt is 112
    assert rul_labels.loc[1] == 112.0

    df_test_rul = compute_test_rul(df_test, rul_labels)
    engine_1_last = df_test_rul[df_test_rul["engine_id"] == 1].iloc[-1]
    assert float(engine_1_last["RUL"]) == 112.0


def test_engine_wise_split_no_leakage() -> None:
    """Verify that train and validation sets have mutually exclusive engine IDs."""
    data = []
    for eng in range(1, 11):
        for cyc in range(1, 21):
            data.append({"engine_id": eng, "cycle": cyc, "sensor_2": 640.0 + cyc})
    df = pd.DataFrame(data)

    train_df, val_df = engine_wise_split(df, val_fraction=0.3, random_seed=42)

    train_engines = set(train_df["engine_id"].unique())
    val_engines = set(val_df["engine_id"].unique())

    assert len(train_engines.intersection(val_engines)) == 0
    assert len(train_engines) + len(val_engines) == 10
    assert len(train_df) + len(val_df) == len(df)


def test_rolling_features_no_cross_engine() -> None:
    """Confirm rolling calculations do not leak state across engine boundaries."""
    # Engine 1 has high sensor values, Engine 2 has low sensor values
    rows = [
        {"engine_id": 1, "cycle": 1, "sensor_2": 1000.0},
        {"engine_id": 1, "cycle": 2, "sensor_2": 1000.0},
        {"engine_id": 2, "cycle": 1, "sensor_2": 50.0},
        {"engine_id": 2, "cycle": 2, "sensor_2": 50.0},
    ]
    df = pd.DataFrame(rows)

    fe = FeatureEngineer(
        selected_sensors=["sensor_2"],
        rolling_windows=[2],
        include_op_settings=False,
        trend_window=1,
    )
    res = fe.build_features(df)

    # Engine 2 cycle 1 rolling mean must be 50.0, NOT (1000 + 50) / 2 = 525.0
    eng2_c1 = res[(res["engine_id"] == 2) & (res["cycle"] == 1)].iloc[0]
    assert eng2_c1["sensor_2_mean_2"] == 50.0
    assert eng2_c1["sensor_2_std_2"] == 0.0

    eng1_c2 = res[(res["engine_id"] == 1) & (res["cycle"] == 2)].iloc[0]
    assert eng1_c2["sensor_2_mean_2"] == 1000.0


def test_preprocessor_fit_transform() -> None:
    """Verify DataPreprocessor imputes missing values and scales features without NaNs."""
    df = pd.DataFrame(
        {
            "sensor_2": [640.0, np.nan, 642.0, 643.0],
            "sensor_3": [1580.0, 1585.0, 1590.0, 1595.0],
        }
    )
    preprocessor = DataPreprocessor(feature_cols=["sensor_2", "sensor_3"])
    scaled = preprocessor.fit_transform(df)

    assert isinstance(scaled, np.ndarray)
    assert scaled.shape == (4, 2)
    assert not np.isnan(scaled).any()


def test_validator_catches_missing() -> None:
    """Verify data quality validator flags injected NaN values."""
    df = pd.DataFrame(
        {
            "engine_id": [1, 1],
            "cycle": [1, 2],
            "sensor_1": [518.67, np.nan],
        }
    )
    quality = validate_quality(df)
    assert quality["valid"] is False
    assert any("sensor_1" in issue for issue in quality["issues"])
