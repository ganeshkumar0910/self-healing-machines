"""Data splitting module providing engine-wise train/validation partition."""

import logging
from typing import Tuple
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def engine_wise_split(
    df: pd.DataFrame, val_fraction: float = 0.2, random_seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split dataset into training and validation sets strictly by engine_id.

    Ensures zero temporal or sequence leakage by ensuring that all cycles for a given
    engine reside entirely in either the training set or the validation set.

    Args:
        df: Input DataFrame containing an 'engine_id' column.
        val_fraction: Fraction of engines to allocate to validation set (0.0 to 1.0).
        random_seed: Random seed for reproducible engine shuffling.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: (train_df, val_df)

    Raises:
        KeyError: If 'engine_id' column is missing from df.
        ValueError: If val_fraction is outside [0.0, 1.0] or if too few engines exist.
    """
    if "engine_id" not in df.columns:
        raise KeyError("Input DataFrame must contain 'engine_id' for engine-wise split.")

    if not 0.0 < val_fraction < 1.0:
        raise ValueError(
            f"val_fraction must be strictly between 0 and 1, got {val_fraction}"
        )

    unique_engines = np.array(sorted(df["engine_id"].unique()))
    num_engines = len(unique_engines)

    if num_engines < 2:
        raise ValueError(
            f"Cannot perform validation split with only {num_engines} engine(s)."
        )

    # Deterministic shuffle
    rng = np.random.RandomState(random_seed)
    shuffled_engines = rng.permutation(unique_engines)

    n_val = max(1, int(round(num_engines * val_fraction)))
    val_engines = set(shuffled_engines[:n_val])
    train_engines = set(shuffled_engines[n_val:])

    train_df = df[df["engine_id"].isin(train_engines)].copy()
    val_df = df[df["engine_id"].isin(val_engines)].copy()

    logger.info(
        "Engine-wise split completed: %d total engines -> %d train engines (%d rows), %d val engines (%d rows)",
        num_engines,
        len(train_engines),
        len(train_df),
        len(val_engines),
        len(val_df),
    )

    return train_df, val_df
