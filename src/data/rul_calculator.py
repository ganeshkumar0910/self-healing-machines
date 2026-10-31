"""RUL (Remaining Useful Life) calculation module for training and testing datasets."""

import logging
from typing import Optional
import pandas as pd

logger = logging.getLogger(__name__)


def compute_train_rul(
    df: pd.DataFrame, cap_value: Optional[int] = None
) -> pd.DataFrame:
    """Compute Remaining Useful Life (RUL) for run-to-failure training records.

    For training engines that run until complete failure, the true RUL at cycle c
    is calculated as:
        RUL = max_cycle_for_engine - c
    Where at failure (c == max_cycle), RUL = 0.

    If cap_value is provided, RUL is capped at that value to model piecewise
    linear degradation (standard practice in CMAPSS literature).

    Args:
        df: Training DataFrame containing 'engine_id' and 'cycle'.
        cap_value: Optional upper bound to clip RUL values (e.g. 125).

    Returns:
        pd.DataFrame: Copy of input DataFrame with newly computed 'RUL' column.

    Raises:
        KeyError: If 'engine_id' or 'cycle' columns are missing.
    """
    if "engine_id" not in df.columns or "cycle" not in df.columns:
        raise KeyError("Input DataFrame must contain 'engine_id' and 'cycle' columns.")

    df_out = df.copy()

    # Calculate max cycle for each engine
    max_cycles = df_out.groupby("engine_id")["cycle"].transform("max")
    rul = max_cycles - df_out["cycle"]

    if cap_value is not None:
        logger.info("Applying RUL cap at %d cycles.", cap_value)
        rul = rul.clip(upper=cap_value)

    df_out["RUL"] = rul.astype(float)
    logger.info(
        "Computed train RUL: min=%.1f, max=%.1f, mean=%.1f (cap=%s)",
        df_out["RUL"].min(),
        df_out["RUL"].max(),
        df_out["RUL"].mean(),
        cap_value,
    )
    return df_out


def compute_test_rul(
    df: pd.DataFrame, rul_labels: pd.Series
) -> pd.DataFrame:
    """Compute Remaining Useful Life (RUL) for truncated test sequences.

    For test engines that are truncated prior to failure, the remaining cycles
    from the last recorded cycle is provided in rul_labels.
    For engine i at cycle c:
        RUL = (max_cycle_in_test_for_engine_i - c) + rul_labels[engine_i]
    At the final recorded cycle c == max_cycle, RUL equals rul_labels[engine_i].

    Args:
        df: Test DataFrame containing 'engine_id' and 'cycle'.
        rul_labels: Series of final true RUL values indexed by engine_id.

    Returns:
        pd.DataFrame: Copy of input DataFrame with newly computed 'RUL' column.

    Raises:
        KeyError: If 'engine_id' or 'cycle' columns are missing.
        ValueError: If rul_labels does not contain all engine IDs present in df.
    """
    if "engine_id" not in df.columns or "cycle" not in df.columns:
        raise KeyError("Input DataFrame must contain 'engine_id' and 'cycle' columns.")

    engine_ids_in_df = set(df["engine_id"].unique())
    labels_indices = set(rul_labels.index)

    missing_engines = engine_ids_in_df - labels_indices
    if missing_engines:
        raise ValueError(
            f"rul_labels is missing ground truth for engine IDs: {sorted(list(missing_engines))}"
        )

    df_out = df.copy()

    # Calculate max recorded cycle for each engine in test set
    max_cycles = df_out.groupby("engine_id")["cycle"].transform("max")
    # Map final RUL to each row based on engine_id
    final_ruls = df_out["engine_id"].map(rul_labels)

    df_out["RUL"] = ((max_cycles - df_out["cycle"]) + final_ruls).astype(float)

    logger.info(
        "Computed test RUL: %d records, min=%.1f, max=%.1f, mean=%.1f",
        len(df_out),
        df_out["RUL"].min(),
        df_out["RUL"].max(),
        df_out["RUL"].mean(),
    )
    return df_out
