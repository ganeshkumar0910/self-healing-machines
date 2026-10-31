"""Prediction service for real-time single-cycle and batch fleet RUL inference."""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union
import numpy as np
import pandas as pd

from src.models.model_registry import get_active_model_path, load_model_artifact

logger = logging.getLogger(__name__)


class PredictionService:
    """Production inference service serving RUL predictions and anomaly flags."""

    def __init__(
        self,
        artifact_dir: Union[str, Path],
        model_name: Optional[str] = None,
        anomaly_detector: Optional[Any] = None,
    ) -> None:
        """Initialize PredictionService.

        Args:
            artifact_dir: Directory containing model artifacts.
            model_name: Specific model name to load. If None, loads active model.
            anomaly_detector: Optional fitted AnomalyDetector instance.
        """
        self.artifact_dir = Path(artifact_dir).resolve()
        self.model_name = model_name
        self.anomaly_detector = anomaly_detector
        self.model = None
        self.preprocessor = None
        self.feature_cols = []
        self.metadata = {}

        self.load_model()

    def load_model(self) -> None:
        """Load target model and associated preprocessor from artifact repository."""
        if self.model_name is None:
            try:
                self.model_name = get_active_model_path(self.artifact_dir)
                logger.info("Resolved active model: %s", self.model_name)
            except Exception as exc:
                logger.warning("Could not resolve active model: %s. Model loading deferred.", exc)
                return

        bundle = load_model_artifact(self.artifact_dir, self.model_name)
        self.model = bundle["model"]
        self.preprocessor = bundle["preprocessor"]
        self.feature_cols = bundle["feature_cols"]
        self.metadata = bundle["metadata"]
        logger.info(
            "Loaded model '%s' with %d features.",
            self.model_name,
            len(self.feature_cols),
        )

    def predict_single(self, features_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Generate RUL prediction and anomaly score for a single observation.

        Args:
            features_dict: Mapping of feature names to numeric values.

        Returns:
            dict: {
                'predicted_rul': float,
                'anomaly_score': float,
                'is_anomaly': int
            }
        """
        if self.model is None or self.preprocessor is None:
            raise RuntimeError("Model or preprocessor is not loaded.")

        df_single = pd.DataFrame([features_dict])
        missing = [c for c in self.feature_cols if c not in df_single.columns]
        if missing:
            raise KeyError(f"Input features missing required columns: {missing}")

        X_scaled = self.preprocessor.transform(df_single[self.feature_cols])
        predicted_rul = float(self.model.predict(X_scaled)[0])

        anomaly_score = 0.0
        is_anomaly = 0
        if self.anomaly_detector is not None:
            anomaly_score = float(self.anomaly_detector.score_samples(X_scaled)[0])
            is_anomaly = int(self.anomaly_detector.predict(X_scaled)[0])

        return {
            "predicted_rul": predicted_rul,
            "anomaly_score": anomaly_score,
            "is_anomaly": is_anomaly,
        }

    def predict_batch(self, df_features: pd.DataFrame) -> pd.DataFrame:
        """Score a pre-engineered feature DataFrame in batch.

        Args:
            df_features: DataFrame containing engineered features matching self.feature_cols.

        Returns:
            pd.DataFrame: Copy of input with predicted_rul, anomaly_score, is_anomaly columns.
        """
        if self.model is None or self.preprocessor is None:
            raise RuntimeError("Model or preprocessor is not loaded.")

        missing = [c for c in self.feature_cols if c not in df_features.columns]
        if missing:
            raise KeyError(f"Feature matrix is missing required columns: {missing}")

        X_scaled = self.preprocessor.transform(df_features[self.feature_cols])
        preds = self.model.predict(X_scaled)

        df_out = df_features.copy()
        df_out["predicted_rul"] = np.asarray(preds, dtype=float)

        if self.anomaly_detector is not None:
            scores = self.anomaly_detector.score_samples(X_scaled)
            flags = self.anomaly_detector.predict(X_scaled)
            df_out["anomaly_score"] = np.asarray(scores, dtype=float)
            df_out["is_anomaly"] = np.asarray(flags, dtype=int)
        else:
            df_out["anomaly_score"] = 0.0
            df_out["is_anomaly"] = 0

        return df_out

    def predict_fleet(
        self,
        df_raw: pd.DataFrame,
        feature_engineer: Any,
        anomaly_detector: Optional[Any] = None,
    ) -> pd.DataFrame:
        """Ingest raw CMAPSS telemetry, engineer features, and predict across all units.

        Args:
            df_raw: Raw sensor telemetry DataFrame.
            feature_engineer: FeatureEngineer instance.
            anomaly_detector: Optional AnomalyDetector instance to use for anomaly scoring.

        Returns:
            pd.DataFrame: Full DataFrame with features, predictions, and anomaly diagnostics.
        """
        logger.info("Running fleet inference on %d raw records...", len(df_raw))
        df_engineered = feature_engineer.build_features(df_raw)

        prev_detector = self.anomaly_detector
        if anomaly_detector is not None:
            self.anomaly_detector = anomaly_detector

        try:
            df_predictions = self.predict_batch(df_engineered)
        finally:
            self.anomaly_detector = prev_detector

        return df_predictions

    def get_latest_predictions(
        self, df_with_predictions: pd.DataFrame
    ) -> pd.DataFrame:
        """Filter predictions DataFrame to the most recent cycle per engine unit.

        Args:
            df_with_predictions: DataFrame containing engine_id, cycle, and prediction columns.

        Returns:
            pd.DataFrame: Single row per engine corresponding to its highest recorded cycle.
        """
        if "engine_id" not in df_with_predictions.columns or "cycle" not in df_with_predictions.columns:
            raise KeyError("DataFrame must contain 'engine_id' and 'cycle' columns.")

        latest_df = (
            df_with_predictions.sort_values(["engine_id", "cycle"])
            .groupby("engine_id", as_index=False)
            .last()
        )
        logger.info(
            "Extracted latest cycle predictions for %d engines.", len(latest_df)
        )
        return latest_df
