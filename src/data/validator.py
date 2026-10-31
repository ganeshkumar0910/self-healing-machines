"""Data validation module for CMAPSS telemetry schema, quality, and cycle integrity."""

import logging
from typing import Any, Dict, List
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def validate_schema(df: pd.DataFrame, expected_cols: List[str]) -> Dict[str, Any]:
    """Validate DataFrame column structure, column names, and numeric data types.

    Args:
        df: Input DataFrame to check.
        expected_cols: Ordered list of required column names.

    Returns:
        dict: {'valid': bool, 'issues': list[str]}
    """
    issues: List[str] = []

    # Check column count
    if len(df.columns) != len(expected_cols):
        issues.append(
            f"Column count mismatch: expected {len(expected_cols)}, got {len(df.columns)}"
        )

    # Check for missing required columns
    missing_cols = [c for c in expected_cols if c not in df.columns]
    if missing_cols:
        issues.append(f"Missing required columns: {missing_cols}")

    # Check for unexpected extra columns
    extra_cols = [c for c in df.columns if c not in expected_cols]
    if extra_cols:
        issues.append(f"Unexpected extra columns: {extra_cols}")

    # Check column data types are numeric
    non_numeric = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        issues.append(f"Non-numeric columns detected: {non_numeric}")

    valid = len(issues) == 0
    logger.debug("Schema validation: valid=%s, issues=%s", valid, issues)
    return {"valid": valid, "issues": issues}


def validate_quality(df: pd.DataFrame) -> Dict[str, Any]:
    """Validate data quality including missing values, infinite values, and duplicate rows.

    Args:
        df: Input DataFrame to evaluate.

    Returns:
        dict: {
            'valid': bool,
            'issues': list[str],
            'stats': {
                'missing_counts': dict,
                'infinite_counts': dict,
                'duplicate_rows': int,
                'total_rows': int
            }
        }
    """
    issues: List[str] = []

    # Check missing values
    missing_per_col = df.isna().sum().to_dict()
    total_missing = int(df.isna().sum().sum())
    if total_missing > 0:
        cols_with_missing = {k: int(v) for k, v in missing_per_col.items() if v > 0}
        issues.append(f"Missing values found in columns: {cols_with_missing}")

    # Check infinite values
    inf_per_col = {}
    total_inf = 0
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            inf_count = int(np.isinf(df[col]).sum())
            if inf_count > 0:
                inf_per_col[col] = inf_count
                total_inf += inf_count

    if total_inf > 0:
        issues.append(f"Infinite values found in columns: {inf_per_col}")

    # Check duplicate rows
    duplicate_rows = int(df.duplicated().sum())
    if duplicate_rows > 0:
        issues.append(f"Found {duplicate_rows} duplicate rows.")

    valid = len(issues) == 0
    stats = {
        "missing_counts": missing_per_col,
        "infinite_counts": inf_per_col,
        "duplicate_rows": duplicate_rows,
        "total_rows": len(df),
    }

    logger.debug("Quality validation: valid=%s, issues=%s", valid, issues)
    return {"valid": valid, "issues": issues, "stats": stats}


def validate_engine_cycles(df: pd.DataFrame) -> Dict[str, Any]:
    """Validate that engine cycles start at 1 and are strictly monotonically increasing.

    Args:
        df: Input DataFrame containing 'engine_id' and 'cycle' columns.

    Returns:
        dict: {'valid': bool, 'issues': list[str]}
    """
    issues: List[str] = []

    if "engine_id" not in df.columns or "cycle" not in df.columns:
        issues.append("Missing required 'engine_id' or 'cycle' columns for cycle validation.")
        return {"valid": False, "issues": issues}

    for engine_id, group in df.groupby("engine_id"):
        cycles = group["cycle"].to_numpy()

        if len(cycles) == 0:
            issues.append(f"Engine {engine_id} has no cycle records.")
            continue

        # Check cycle starts at 1
        if cycles[0] != 1:
            issues.append(
                f"Engine {engine_id} cycles start at {cycles[0]} instead of 1."
            )

        # Check strictly monotonically increasing
        diffs = np.diff(cycles)
        if not np.all(diffs > 0):
            issues.append(
                f"Engine {engine_id} cycles are not strictly monotonically increasing."
            )

    valid = len(issues) == 0
    logger.debug("Engine cycle validation: valid=%s, issues=%s", valid, issues)
    return {"valid": valid, "issues": issues}


def run_full_validation(
    df: pd.DataFrame, expected_cols: List[str]
) -> Dict[str, Any]:
    """Run schema, quality, and cycle validations and aggregate results.

    Args:
        df: Input DataFrame to validate.
        expected_cols: List of expected column names.

    Returns:
        dict: {
            'valid': bool,
            'schema': dict,
            'quality': dict,
            'cycles': dict,
            'issues': list[str]
        }
    """
    schema_res = validate_schema(df, expected_cols)
    quality_res = validate_quality(df)
    cycle_res = validate_engine_cycles(df)

    all_issues = (
        schema_res["issues"] + quality_res["issues"] + cycle_res["issues"]
    )
    overall_valid = schema_res["valid"] and quality_res["valid"] and cycle_res["valid"]

    logger.info(
        "Full validation completed: valid=%s (%d total issues)",
        overall_valid,
        len(all_issues),
    )

    return {
        "valid": overall_valid,
        "schema": schema_res,
        "quality": quality_res,
        "cycles": cycle_res,
        "issues": all_issues,
    }
