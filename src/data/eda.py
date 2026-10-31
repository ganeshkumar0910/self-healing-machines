"""Exploratory Data Analysis (EDA) helper module for computing and exporting summary statistics."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Union
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _to_serializable(val: Any) -> Any:
    """Recursively convert NumPy scalars/arrays to native Python types for JSON."""
    if isinstance(val, (np.floating, float)):
        return float(val) if not np.isnan(val) else None
    elif isinstance(val, (np.integer, int)):
        return int(val)
    elif isinstance(val, (np.ndarray, list)):
        return [_to_serializable(item) for item in val]
    elif isinstance(val, dict):
        return {str(k): _to_serializable(v) for k, v in val.items()}
    return val


def compute_eda_stats(
    df: pd.DataFrame, sensor_cols: List[str]
) -> Dict[str, Any]:
    """Compute exploratory summary statistics for CMAPSS telemetry data.

    Calculates descriptive metrics across engine lifecycles, per-sensor distributional
    moments (mean, std, min, max, skewness), and missing data diagnostics.

    Args:
        df: Input DataFrame containing telemetry features.
        sensor_cols: List of sensor column names to analyze.

    Returns:
        dict: Aggregated summary statistics.
    """
    logger.info("Computing EDA statistics for %d sensors across %d rows.", len(sensor_cols), len(df))

    # Sensor moments
    sensor_stats = {}
    for col in sensor_cols:
        if col in df.columns:
            series = df[col].dropna()
            sensor_stats[col] = {
                "mean": float(series.mean()),
                "std": float(series.std()),
                "min": float(series.min()),
                "max": float(series.max()),
                "skewness": float(series.skew()) if len(series) > 2 else 0.0,
            }
        else:
            logger.warning("Sensor column %s not found in DataFrame.", col)

    # Missing value counts
    missing_counts = {col: int(df[col].isna().sum()) for col in df.columns}

    # Engine and cycle statistics
    engine_count = int(df["engine_id"].nunique()) if "engine_id" in df.columns else 0
    cycle_stats: Dict[str, Any] = {}

    if "cycle" in df.columns and "engine_id" in df.columns:
        max_cycles_per_engine = df.groupby("engine_id")["cycle"].max()
        cycle_stats = {
            "total_records": len(df),
            "min_cycle_recorded": int(df["cycle"].min()),
            "max_cycle_recorded": int(df["cycle"].max()),
            "mean_engine_lifespan": float(max_cycles_per_engine.mean()),
            "std_engine_lifespan": float(max_cycles_per_engine.std()),
            "min_engine_lifespan": int(max_cycles_per_engine.min()),
            "max_engine_lifespan": int(max_cycles_per_engine.max()),
        }

    stats = {
        "engine_count": engine_count,
        "total_rows": len(df),
        "cycle_stats": cycle_stats,
        "missing_counts": missing_counts,
        "sensor_stats": sensor_stats,
    }
    return stats


def save_eda_report(stats: Dict[str, Any], output_path: Union[str, Path]) -> None:
    """Save summary statistics as formatted JSON report.

    Args:
        stats: Dictionary of EDA statistics.
        output_path: Destination JSON file path.
    """
    dest_path = Path(output_path).resolve()
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    serializable_stats = _to_serializable(stats)
    with open(dest_path, "w", encoding="utf-8") as f:
        json.dump(serializable_stats, f, indent=2)

    logger.info("Saved EDA report to: %s", dest_path)
