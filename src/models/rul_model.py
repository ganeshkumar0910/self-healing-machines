"""RUL regression models and specialized evaluation metrics (NASA asymmetric scoring)."""

from abc import ABC, abstractmethod
import logging
from typing import Any, Dict, Optional, Union
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

logger = logging.getLogger(__name__)


def compute_nasa_score(
    y_true: Union[np.ndarray, pd.Series],
    y_pred: Union[np.ndarray, pd.Series],
) -> float:
    """Calculate the official NASA PHM08 / CMAPSS asymmetric scoring metric.

    The score penalizes late predictions (where predicted RUL > true RUL) more
    heavily than early predictions (predicted RUL < true RUL), reflecting the
    operational reality where late predictions risk catastrophic in-flight failure:
        d = predicted_RUL - true_RUL
        s = exp(-d / 13) - 1  if d < 0 (early prediction)
        s = exp(d / 10) - 1   if d >= 0 (late prediction)
        total_score = sum(s)

    Args:
        y_true: True Remaining Useful Life values.
        y_pred: Predicted Remaining Useful Life values.

    Returns:
        float: Total NASA asymmetric evaluation score.
    """
    y_t = np.asarray(y_true, dtype=float)
    y_p = np.asarray(y_pred, dtype=float)

    if y_t.shape != y_p.shape:
        raise ValueError(
            f"Shape mismatch: y_true shape {y_t.shape} vs y_pred shape {y_p.shape}"
        )

    d = y_p - y_t
    # Vectorized piecewise evaluation
    scores = np.where(d < 0, np.exp(-d / 13.0) - 1.0, np.exp(d / 10.0) - 1.0)
    return float(np.sum(scores))


def evaluate_model(
    y_true: Union[np.ndarray, pd.Series],
    y_pred: Union[np.ndarray, pd.Series],
) -> Dict[str, float]:
    """Compute comprehensive regression evaluation metrics: MAE, RMSE, and NASA score.

    Args:
        y_true: Ground truth target values.
        y_pred: Model predicted target values.

    Returns:
        dict: {'mae': float, 'rmse': float, 'nasa_score': float}
    """
    y_t = np.asarray(y_true, dtype=float)
    y_p = np.asarray(y_pred, dtype=float)

    mae = float(mean_absolute_error(y_t, y_p))
    rmse = float(np.sqrt(mean_squared_error(y_t, y_p)))
    nasa = compute_nasa_score(y_t, y_p)

    logger.info("Evaluation metrics -> MAE: %.3f, RMSE: %.3f, NASA Score: %.3f", mae, rmse, nasa)
    return {
        "mae": mae,
        "rmse": rmse,
        "nasa_score": nasa,
    }


class BaseRULModel(ABC):
    """Abstract base class for Remaining Useful Life regression estimators."""

    @abstractmethod
    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
    ) -> "BaseRULModel":
        """Fit estimator to training features and target RUL."""
        pass

    @abstractmethod
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Predict RUL values for input feature matrix."""
        pass

    @abstractmethod
    def get_params(self) -> Dict[str, Any]:
        """Return estimator hyperparameters as dictionary."""
        pass

    @property
    @abstractmethod
    def model_type(self) -> str:
        """Return standard string identifier for model architecture."""
        pass


class LinearRULModel(BaseRULModel):
    """Ordinary Least Squares Linear Regression baseline for RUL prediction."""

    def __init__(self, fit_intercept: bool = True, **kwargs: Any) -> None:
        self._fit_intercept = fit_intercept
        self._kwargs = kwargs
        self.model = LinearRegression(fit_intercept=fit_intercept, **kwargs)

    @property
    def model_type(self) -> str:
        return "linear_regression"

    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
    ) -> "LinearRULModel":
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y, dtype=float)
        logger.info("Fitting LinearRULModel on shape %s", X_arr.shape)
        self.model.fit(X_arr, y_arr)
        return self

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        X_arr = np.asarray(X, dtype=float)
        preds = self.model.predict(X_arr)
        return np.asarray(preds, dtype=float)

    def get_params(self) -> Dict[str, Any]:
        params = self.model.get_params()
        params["model_type"] = self.model_type
        return params


class RandomForestRULModel(BaseRULModel):
    """Random Forest ensemble regressor for non-linear degradation tracking."""

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = 10,
        min_samples_split: int = 5,
        min_samples_leaf: int = 2,
        random_state: int = 42,
        n_jobs: int = -1,
        **kwargs: Any,
    ) -> None:
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            n_jobs=n_jobs,
            **kwargs,
        )

    @property
    def model_type(self) -> str:
        return "random_forest"

    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
    ) -> "RandomForestRULModel":
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y, dtype=float)
        logger.info("Fitting RandomForestRULModel on shape %s", X_arr.shape)
        self.model.fit(X_arr, y_arr)
        return self

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        X_arr = np.asarray(X, dtype=float)
        preds = self.model.predict(X_arr)
        return np.asarray(preds, dtype=float)

    def get_params(self) -> Dict[str, Any]:
        params = self.model.get_params()
        params["model_type"] = self.model_type
        return params


class GradientBoostingRULModel(BaseRULModel):
    """Gradient Boosted Decision Trees for robust residual minimization."""

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 5,
        learning_rate: float = 0.1,
        random_state: int = 42,
        **kwargs: Any,
    ) -> None:
        self.model = GradientBoostingRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state,
            **kwargs,
        )

    @property
    def model_type(self) -> str:
        return "gradient_boosting"

    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
    ) -> "GradientBoostingRULModel":
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y, dtype=float)
        logger.info("Fitting GradientBoostingRULModel on shape %s", X_arr.shape)
        self.model.fit(X_arr, y_arr)
        return self

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        X_arr = np.asarray(X, dtype=float)
        preds = self.model.predict(X_arr)
        return np.asarray(preds, dtype=float)

    def get_params(self) -> Dict[str, Any]:
        params = self.model.get_params()
        params["model_type"] = self.model_type
        return params
