"""Enhanced feature engineering module — rolling stats, EMA, trends, sensor diffs.

Improvements over v1:
- Rolling min / max (windows 5, 10, 20)
- Exponential Moving Average (EMA, α=0.1) per sensor
- Sensor difference pairs (sensor_2 - sensor_21, etc.) where physically meaningful
- Cycle percentage feature (degradation stage proxy)
- All calculations strictly engine-wise — no cross-engine leakage
"""

import logging
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Physically meaningful sensor difference pairs for FD001-FD004
# Based on domain knowledge: compressor vs turbine stage sensors
SENSOR_DIFF_PAIRS: List[Tuple[str, str]] = [
    ("sensor_2", "sensor_3"),    # HPC outlet temp vs LPT outlet temp
    ("sensor_4", "sensor_11"),   # HPC outlet pressure vs HPC effl temp
    ("sensor_9", "sensor_14"),   # Physical total pressure ratio pair
    ("sensor_7", "sensor_12"),   # Fan inlet static pressure vs corrected fan speed
]

# EMA smoothing factor
EMA_ALPHA = 0.1


class FeatureEngineer:
    """Computes rolling statistics, EMA, degradation trends, and sensor differences.

    All calculations are partitioned strictly by engine_id to prevent any
    cross-unit data leakage. Operates on time-ordered cycles within each engine.

    Feature count breakdown:
        Base sensors          : N
        Operating settings    : 3 (if include_op_settings)
        Rolling mean          : N * len(windows)
        Rolling std           : N * len(windows)
        Rolling min           : N * len(windows)
        Rolling max           : N * len(windows)
        EMA                   : N
        Trend (slope)         : N
        Sensor diffs          : len(diff_pairs) [if include_sensor_diffs]
        cycle_pct             : 1 (if include_cycle_pct)
    """

    def __init__(
        self,
        selected_sensors: List[str],
        rolling_windows: Optional[List[int]] = None,
        include_op_settings: bool = True,
        trend_window: int = 5,
        include_rolling_minmax: bool = True,
        include_ema: bool = True,
        include_sensor_diffs: bool = True,
        include_cycle_pct: bool = True,
        ema_alpha: float = EMA_ALPHA,
        sensor_diff_pairs: Optional[List[Tuple[str, str]]] = None,
    ) -> None:
        """Initialize FeatureEngineer.

        Args:
            selected_sensors: List of sensor columns to calculate features for.
            rolling_windows: Window sizes (cycles) for rolling statistics.
            include_op_settings: Include operating condition columns.
            trend_window: Lag interval for rate-of-change calculation.
            include_rolling_minmax: Compute rolling min/max (new in v2).
            include_ema: Compute EMA for each sensor (new in v2).
            include_sensor_diffs: Compute sensor difference pairs (new in v2).
            include_cycle_pct: Include cycle / max_cycle ratio (new in v2).
            ema_alpha: EMA smoothing factor (0 < alpha <= 1).
            sensor_diff_pairs: List of (sensor_a, sensor_b) tuples for differences.
        """
        self.selected_sensors = list(selected_sensors)
        self.rolling_windows = list(rolling_windows or [5, 10, 20])
        self.include_op_settings = include_op_settings
        self.trend_window = trend_window
        self.include_rolling_minmax = include_rolling_minmax
        self.include_ema = include_ema
        self.include_sensor_diffs = include_sensor_diffs
        self.include_cycle_pct = include_cycle_pct
        self.ema_alpha = ema_alpha

        if sensor_diff_pairs is not None:
            self.sensor_diff_pairs = [
                (a, b) for (a, b) in sensor_diff_pairs
                if a in self.selected_sensors and b in self.selected_sensors
            ]
        else:
            self.sensor_diff_pairs = [
                (a, b) for (a, b) in SENSOR_DIFF_PAIRS
                if a in self.selected_sensors and b in self.selected_sensors
            ]

    def get_feature_columns(self) -> List[str]:
        """Return complete, deterministic list of engineered feature column names.

        Returns:
            list[str]: Feature names excluding metadata columns (engine_id, cycle, RUL).
        """
        cols: List[str] = []

        # Base sensor readings
        cols.extend(self.selected_sensors)

        # Operating settings
        if self.include_op_settings:
            cols.extend(["op_setting_1", "op_setting_2", "op_setting_3"])

        # Rolling statistics
        for w in self.rolling_windows:
            for sensor in self.selected_sensors:
                cols.append(f"{sensor}_mean_{w}")
                cols.append(f"{sensor}_std_{w}")

        # Rolling min / max (new in v2)
        if self.include_rolling_minmax:
            for w in self.rolling_windows:
                for sensor in self.selected_sensors:
                    cols.append(f"{sensor}_min_{w}")
                    cols.append(f"{sensor}_max_{w}")

        # EMA (new in v2)
        if self.include_ema:
            for sensor in self.selected_sensors:
                cols.append(f"{sensor}_ema")

        # Trend (rate of change)
        for sensor in self.selected_sensors:
            cols.append(f"{sensor}_trend")

        # Sensor differences (new in v2)
        if self.include_sensor_diffs:
            for a, b in self.sensor_diff_pairs:
                cols.append(f"{a}_minus_{b}")

        # Cycle percentage (new in v2)
        if self.include_cycle_pct:
            cols.append("cycle_pct")

        return cols

    def feature_count(self) -> int:
        """Return the total number of engineered features."""
        return len(self.get_feature_columns())

    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract and engineer features for all engines in the input DataFrame.

        Computations are partitioned by engine_id and sorted by cycle to ensure
        correct temporal ordering. No information from future cycles is used.

        Args:
            df: Input DataFrame containing engine telemetry.

        Returns:
            pd.DataFrame: Feature DataFrame with identifier and optional RUL columns.

        Raises:
            KeyError: If required columns are missing.
        """
        if "engine_id" not in df.columns or "cycle" not in df.columns:
            raise KeyError("DataFrame must contain 'engine_id' and 'cycle' columns.")

        missing_sensors = [s for s in self.selected_sensors if s not in df.columns]
        if missing_sensors:
            raise KeyError(f"DataFrame is missing required sensors: {missing_sensors}")

        logger.info(
            "Engineering features for %d engines across %d rows (%d features)...",
            df["engine_id"].nunique(), len(df), self.feature_count(),
        )

        # Sort strictly by engine + cycle for temporal ordering
        df_sorted = df.sort_values(["engine_id", "cycle"]).reset_index(drop=True)
        grouped = df_sorted.groupby("engine_id", sort=False)

        features: Dict[str, np.ndarray] = {}

        # 1. Base sensor readings
        for sensor in self.selected_sensors:
            features[sensor] = df_sorted[sensor].to_numpy(dtype=float)

        # 2. Operating settings
        if self.include_op_settings:
            for op in ["op_setting_1", "op_setting_2", "op_setting_3"]:
                if op in df_sorted.columns:
                    features[op] = df_sorted[op].to_numpy(dtype=float)
                else:
                    logger.warning("Operating setting %s not found — filling with 0.", op)
                    features[op] = np.zeros(len(df_sorted), dtype=float)

        # 3. Rolling mean and std
        for w in self.rolling_windows:
            for sensor in self.selected_sensors:
                mean_s = (
                    grouped[sensor]
                    .rolling(window=w, min_periods=1)
                    .mean()
                    .reset_index(level=0, drop=True)
                )
                features[f"{sensor}_mean_{w}"] = mean_s.to_numpy(dtype=float)

                std_s = (
                    grouped[sensor]
                    .rolling(window=w, min_periods=1)
                    .std()
                    .reset_index(level=0, drop=True)
                    .fillna(0.0)
                )
                features[f"{sensor}_std_{w}"] = std_s.to_numpy(dtype=float)

        # 4. Rolling min / max (new in v2)
        if self.include_rolling_minmax:
            for w in self.rolling_windows:
                for sensor in self.selected_sensors:
                    min_s = (
                        grouped[sensor]
                        .rolling(window=w, min_periods=1)
                        .min()
                        .reset_index(level=0, drop=True)
                    )
                    features[f"{sensor}_min_{w}"] = min_s.to_numpy(dtype=float)

                    max_s = (
                        grouped[sensor]
                        .rolling(window=w, min_periods=1)
                        .max()
                        .reset_index(level=0, drop=True)
                    )
                    features[f"{sensor}_max_{w}"] = max_s.to_numpy(dtype=float)

        # 5. EMA per sensor (new in v2) — computed engine-wise via pandas ewm
        if self.include_ema:
            for sensor in self.selected_sensors:
                ema_s = (
                    grouped[sensor]
                    .transform(
                        lambda x: x.ewm(alpha=self.ema_alpha, adjust=False).mean()
                    )
                )
                features[f"{sensor}_ema"] = ema_s.to_numpy(dtype=float)

        # 6. Trend (lagged difference)
        for sensor in self.selected_sensors:
            trend_s = (
                grouped[sensor]
                .diff(periods=self.trend_window)
                .fillna(0.0)
            )
            features[f"{sensor}_trend"] = trend_s.to_numpy(dtype=float)

        # 7. Sensor differences (new in v2)
        if self.include_sensor_diffs:
            for a, b in self.sensor_diff_pairs:
                features[f"{a}_minus_{b}"] = (
                    df_sorted[a].to_numpy(dtype=float)
                    - df_sorted[b].to_numpy(dtype=float)
                )

        # 8. Cycle percentage — current_cycle / max_cycle_for_engine (new in v2)
        if self.include_cycle_pct:
            max_cycles = grouped["cycle"].transform("max")
            cycle_pct = (df_sorted["cycle"] / max_cycles).fillna(0.0)
            features["cycle_pct"] = cycle_pct.to_numpy(dtype=float)

        # Assemble output DataFrame
        feature_cols = self.get_feature_columns()
        out_dict: Dict[str, np.ndarray] = {
            "engine_id": df_sorted["engine_id"].to_numpy(dtype=int),
            "cycle": df_sorted["cycle"].to_numpy(dtype=int),
        }
        for col in feature_cols:
            out_dict[col] = features[col]

        if "RUL" in df_sorted.columns:
            out_dict["RUL"] = df_sorted["RUL"].to_numpy(dtype=float)

        out_df = pd.DataFrame(out_dict)
        logger.info(
            "Feature engineering complete: %d rows, %d features.",
            len(out_df), len(feature_cols),
        )
        return out_df
