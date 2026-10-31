"""Feature engineering module for creating rolling and trend degradation indicators."""

import logging
from typing import List
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Computes rolling statistics, degradation trends, and aggregated telemetry features.

    Calculations are partitioned strictly by engine_id to prevent any cross-unit data
    leakage. Operates on time-ordered cycles within each engine sequence.
    """

    def __init__(
        self,
        selected_sensors: List[str],
        rolling_windows: List[int],
        include_op_settings: bool = True,
        trend_window: int = 5,
    ) -> None:
        """Initialize FeatureEngineer.

        Args:
            selected_sensors: List of sensor columns to calculate features for.
            rolling_windows: List of window sizes (in cycles) for rolling mean and std.
            include_op_settings: Whether to include operating condition setting columns.
            trend_window: Lag interval (in cycles) for rate-of-change / trend calculation.
        """
        self.selected_sensors = list(selected_sensors)
        self.rolling_windows = list(rolling_windows)
        self.include_op_settings = include_op_settings
        self.trend_window = trend_window

    def get_feature_columns(self) -> List[str]:
        """Return the complete list of engineered feature column names in deterministic order.

        Returns:
            list[str]: Feature names excluding metadata columns (engine_id, cycle, RUL).
        """
        cols: List[str] = []
        cols.extend(self.selected_sensors)

        if self.include_op_settings:
            cols.extend(["op_setting_1", "op_setting_2", "op_setting_3"])

        for w in self.rolling_windows:
            for sensor in self.selected_sensors:
                cols.append(f"{sensor}_mean_{w}")
                cols.append(f"{sensor}_std_{w}")

        for sensor in self.selected_sensors:
            cols.append(f"{sensor}_trend")

        return cols

    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract and engineer features for all engines in the input DataFrame.

        For each engine separately (grouped by engine_id and sorted by cycle):
        - Compute rolling mean and rolling std for each selected sensor over each window.
        - Compute trend (diff over trend_window) for each selected sensor.
        - Include current sensor readings and op settings.
        - Preserves engine_id, cycle, and RUL columns if present.

        Args:
            df: Input DataFrame containing engine telemetry.

        Returns:
            pd.DataFrame: DataFrame containing all features and identifier/target columns.

        Raises:
            KeyError: If required sensor or identifier columns are missing.
        """
        if "engine_id" not in df.columns or "cycle" not in df.columns:
            raise KeyError("DataFrame must contain 'engine_id' and 'cycle' columns.")

        missing_sensors = [s for s in self.selected_sensors if s not in df.columns]
        if missing_sensors:
            raise KeyError(f"DataFrame is missing required selected sensors: {missing_sensors}")

        logger.info(
            "Engineering features for %d engines across %d rows...",
            df["engine_id"].nunique(),
            len(df),
        )

        # Sort strictly by engine_id and cycle to guarantee temporal ordering
        df_sorted = df.sort_values(["engine_id", "cycle"]).reset_index(drop=True)
        grouped = df_sorted.groupby("engine_id", sort=False)

        feature_cols = self.get_feature_columns()

        # Build feature dictionary to avoid DataFrame fragmentation
        features: dict = {}

        # 1. Base selected sensors
        for sensor in self.selected_sensors:
            features[sensor] = df_sorted[sensor].to_numpy(dtype=float)

        # 2. Operating settings
        if self.include_op_settings:
            for op in ["op_setting_1", "op_setting_2", "op_setting_3"]:
                if op in df_sorted.columns:
                    features[op] = df_sorted[op].to_numpy(dtype=float)
                else:
                    logger.warning("Operating setting %s not found in DataFrame; filling with 0.0.", op)
                    features[op] = np.zeros(len(df_sorted), dtype=float)

        # 3. Rolling window statistics per engine
        for w in self.rolling_windows:
            for sensor in self.selected_sensors:
                # Rolling mean with min_periods=1 to support initial cycles
                mean_series = (
                    grouped[sensor]
                    .rolling(window=w, min_periods=1)
                    .mean()
                    .reset_index(level=0, drop=True)
                )
                features[f"{sensor}_mean_{w}"] = mean_series.to_numpy(dtype=float)

                # Rolling std with min_periods=1 (std of 1 element is NaN, fillna with 0.0)
                std_series = (
                    grouped[sensor]
                    .rolling(window=w, min_periods=1)
                    .std()
                    .reset_index(level=0, drop=True)
                    .fillna(0.0)
                )
                features[f"{sensor}_std_{w}"] = std_series.to_numpy(dtype=float)

        # 4. Degradation trend (lagged difference over trend_window per engine)
        for sensor in self.selected_sensors:
            trend_series = (
                grouped[sensor]
                .diff(periods=self.trend_window)
                .fillna(0.0)
            )
            features[f"{sensor}_trend"] = trend_series.to_numpy(dtype=float)

        # Assemble output DataFrame
        out_dict = {
            "engine_id": df_sorted["engine_id"].to_numpy(dtype=int),
            "cycle": df_sorted["cycle"].to_numpy(dtype=int),
        }
        for col in feature_cols:
            out_dict[col] = features[col]

        if "RUL" in df_sorted.columns:
            out_dict["RUL"] = df_sorted["RUL"].to_numpy(dtype=float)

        out_df = pd.DataFrame(out_dict)
        logger.info(
            "Feature engineering complete: %d rows, %d total features created.",
            len(out_df),
            len(feature_cols),
        )
        return out_df
