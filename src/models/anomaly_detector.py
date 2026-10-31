"""Anomaly detector module utilizing Isolation Forests for unmonitored sensor drift detection."""

import logging
from pathlib import Path
from typing import Optional, Union
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

logger = logging.getLogger(__name__)


class AnomalyDetector:
    """Detects multi-sensor degradation anomalies via Isolation Forest scoring.

    Attributes:
        threshold_: Calibrated decision threshold. Observations with scores
            strictly below this threshold are flagged as anomalous (1).
    """

    def __init__(
        self,
        contamination: float = 0.05,
        n_estimators: int = 100,
        random_state: int = 42,
    ) -> None:
        """Initialize AnomalyDetector.

        Args:
            contamination: Expected proportion of outliers in the training population.
            n_estimators: Number of isolation trees in the ensemble.
            random_state: Random seed for reproducible partitioning.
        """
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.threshold_: Optional[float] = None
        self.model = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=random_state,
        )
        self.is_fitted: bool = False

    def fit(self, X: Union[np.ndarray, pd.DataFrame]) -> "AnomalyDetector":
        """Fit Isolation Forest estimator on nominal/training telemetry.

        Args:
            X: Input feature array or DataFrame.

        Returns:
            AnomalyDetector: Self fitted instance.
        """
        X_arr = np.asarray(X, dtype=float)
        logger.info("Fitting AnomalyDetector on shape %s", X_arr.shape)
        self.model.fit(X_arr)
        self.is_fitted = True
        return self

    def score_samples(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Compute anomaly score for each observation using sklearn decision_function.

        Lower / more negative scores correspond to higher degree of abnormality.

        Args:
            X: Input feature array or DataFrame.

        Returns:
            np.ndarray: 1D array of decision function scores.

        Raises:
            RuntimeError: If called before fitting.
        """
        if not self.is_fitted:
            raise RuntimeError("AnomalyDetector must be fitted before scoring samples.")

        X_arr = np.asarray(X, dtype=float)
        scores = self.model.decision_function(X_arr)
        return np.asarray(scores, dtype=float)

    def calibrate_threshold(
        self, X_val: Union[np.ndarray, pd.DataFrame], percentile: float = 10.0
    ) -> float:
        """Calibrate anomaly score threshold as a specific percentile of validation scores.

        Args:
            X_val: Validation dataset features.
            percentile: Percentile (0 to 100) below which an observation is deemed anomalous.

        Returns:
            float: Calibrated score threshold.
        """
        scores = self.score_samples(X_val)
        self.threshold_ = float(np.percentile(scores, percentile))
        logger.info(
            "Calibrated anomaly threshold to %.4f at percentile %.1f",
            self.threshold_,
            percentile,
        )
        return self.threshold_

    def predict(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        threshold: Optional[float] = None,
    ) -> np.ndarray:
        """Predict binary anomaly flags (1: anomaly, 0: normal).

        If a threshold is explicitly passed, or self.threshold_ is set,
        scores < threshold are labeled as anomalies (1). Otherwise, falls back
        to sklearn's default IsolationForest decision (-1 -> 1, 1 -> 0).

        Args:
            X: Input feature array or DataFrame.
            threshold: Optional override threshold.

        Returns:
            np.ndarray: Binary array (1 for anomaly, 0 for normal).
        """
        if not self.is_fitted:
            raise RuntimeError("AnomalyDetector must be fitted before predicting.")

        X_arr = np.asarray(X, dtype=float)
        active_threshold = threshold if threshold is not None else self.threshold_

        if active_threshold is not None:
            scores = self.score_samples(X_arr)
            return (scores < active_threshold).astype(int)
        else:
            raw_preds = self.model.predict(X_arr)
            # IsolationForest returns -1 for outlier/anomaly and +1 for inlier
            return (raw_preds == -1).astype(int)

    def save(self, path: Union[str, Path]) -> None:
        """Serialize AnomalyDetector instance to disk via joblib.

        Args:
            path: Destination file path.
        """
        dest_path = Path(path).resolve()
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, dest_path)
        logger.info("Saved AnomalyDetector to %s", dest_path)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "AnomalyDetector":
        """Load serialized AnomalyDetector instance from disk.

        Args:
            path: Source file path.

        Returns:
            AnomalyDetector: Deserialized detector instance.
        """
        src_path = Path(path).resolve()
        if not src_path.is_file():
            raise FileNotFoundError(f"AnomalyDetector artifact not found: {src_path}")

        obj = joblib.load(src_path)
        if not isinstance(obj, cls):
            raise TypeError(f"Loaded object is of type {type(obj)}, expected {cls.__name__}")

        logger.info("Loaded AnomalyDetector from %s", src_path)
        return obj
