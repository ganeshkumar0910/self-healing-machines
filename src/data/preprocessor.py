"""Data preprocessor module handling feature imputation, scaling, and serialization."""

import logging
from pathlib import Path
from typing import List, Union
import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Preprocesses feature matrices via imputation and standard scaling.

    Fits imputation and scaling parameters solely on training features to prevent
    data leakage, and serializes state for repeatable deployment inference.
    """

    def __init__(
        self,
        feature_cols: List[str],
        impute_strategy: str = "mean",
    ) -> None:
        """Initialize DataPreprocessor.

        Args:
            feature_cols: Ordered list of feature column names to extract and transform.
            impute_strategy: Strategy for missing value imputation ('mean', 'median', etc.).
        """
        self.feature_cols = list(feature_cols)
        self.impute_strategy = impute_strategy
        self.imputer = SimpleImputer(strategy=impute_strategy)
        self.scaler = StandardScaler()
        self.is_fitted: bool = False

    def _extract_features(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Extract array of target features from DataFrame or NumPy array."""
        if isinstance(X, pd.DataFrame):
            missing = [c for c in self.feature_cols if c not in X.columns]
            if missing:
                raise KeyError(f"Input DataFrame is missing required features: {missing}")
            return X[self.feature_cols].to_numpy(dtype=float)
        elif isinstance(X, np.ndarray):
            if X.shape[1] != len(self.feature_cols):
                raise ValueError(
                    f"Array column count {X.shape[1]} does not match expected {len(self.feature_cols)} features."
                )
            return np.asarray(X, dtype=float)
        else:
            raise TypeError(f"Unsupported input type for features: {type(X)}")

    def fit(self, X: Union[pd.DataFrame, np.ndarray]) -> "DataPreprocessor":
        """Fit imputer and scaler on feature data.

        Args:
            X: Training features (pd.DataFrame containing feature_cols, or np.ndarray).

        Returns:
            DataPreprocessor: Self fitted instance.
        """
        logger.info(
            "Fitting DataPreprocessor on %d features using strategy='%s'",
            len(self.feature_cols),
            self.impute_strategy,
        )
        data = self._extract_features(X)
        imputed = self.imputer.fit_transform(data)
        self.scaler.fit(imputed)
        self.is_fitted = True
        return self

    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Apply fitted imputer and scaler to input features.

        Args:
            X: Input features to transform.

        Returns:
            np.ndarray: Imputed and scaled feature matrix.

        Raises:
            RuntimeError: If transform is called prior to fit.
        """
        if not self.is_fitted:
            raise RuntimeError("DataPreprocessor must be fitted before transforming data.")

        data = self._extract_features(X)
        imputed = self.imputer.transform(data)
        scaled = self.scaler.transform(imputed)
        return scaled

    def fit_transform(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Fit on data and transform it in one step.

        Args:
            X: Input features to fit and transform.

        Returns:
            np.ndarray: Scaled feature matrix.
        """
        return self.fit(X).transform(X)

    def save(self, path: Union[str, Path]) -> None:
        """Serialize preprocessor to disk using joblib.

        Args:
            path: Destination file path.
        """
        dest_path = Path(path).resolve()
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, dest_path)
        logger.info("Saved DataPreprocessor to %s", dest_path)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "DataPreprocessor":
        """Load serialized preprocessor from disk.

        Args:
            path: Path to serialized preprocessor joblib file.

        Returns:
            DataPreprocessor: Deserialized preprocessor instance.
        """
        src_path = Path(path).resolve()
        if not src_path.is_file():
            raise FileNotFoundError(f"Preprocessor file not found: {src_path}")

        obj = joblib.load(src_path)
        if not isinstance(obj, cls):
            raise TypeError(f"Loaded object is of type {type(obj)}, expected {cls.__name__}")

        logger.info("Loaded DataPreprocessor from %s", src_path)
        return obj
