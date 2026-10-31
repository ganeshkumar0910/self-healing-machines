"""RUL regression models, evaluation metrics, and advanced model types — CPU-only.

Includes:
- NASA asymmetric scoring function
- Row-level and endpoint-level evaluation protocols
- Linear, RandomForest, GradientBoosting, ExtraTrees, HistGradientBoosting, XGBoost
- All models CPU-compatible; n_jobs=-1 where applicable
"""

from abc import ABC, abstractmethod
import logging
import time
from typing import Any, Dict, Optional, Union
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

logger = logging.getLogger(__name__)


# ── Evaluation metrics ────────────────────────────────────────────────────────

def compute_nasa_score(
    y_true: Union[np.ndarray, pd.Series],
    y_pred: Union[np.ndarray, pd.Series],
) -> float:
    """Calculate the official NASA PHM08 / CMAPSS asymmetric scoring metric.

    The score penalises late predictions (predicted RUL > true RUL) more heavily
    than early predictions, reflecting operational risk of missed failures:
        d = predicted_RUL - true_RUL
        s = exp(-d / 13) - 1  if d < 0  (early prediction)
        s = exp( d / 10) - 1  if d >= 0 (late prediction)
        total_score = sum(s)

    Args:
        y_true: True Remaining Useful Life values.
        y_pred: Predicted Remaining Useful Life values.

    Returns:
        float: Total NASA asymmetric evaluation score (lower is better).
    """
    y_t = np.asarray(y_true, dtype=float)
    y_p = np.asarray(y_pred, dtype=float)

    if y_t.shape != y_p.shape:
        raise ValueError(
            f"Shape mismatch: y_true shape {y_t.shape} vs y_pred shape {y_p.shape}"
        )

    d = y_p - y_t
    scores = np.where(d < 0, np.exp(-d / 13.0) - 1.0, np.exp(d / 10.0) - 1.0)
    return float(np.sum(scores))


def evaluate_model(
    y_true: Union[np.ndarray, pd.Series],
    y_pred: Union[np.ndarray, pd.Series],
) -> Dict[str, float]:
    """Compute MAE, RMSE, and NASA score for a set of predictions.

    This is the ROW-LEVEL evaluation (all cycles). Use for training monitoring.
    For operational accuracy, use evaluate_endpoint() instead.

    Args:
        y_true: Ground truth RUL values.
        y_pred: Predicted RUL values.

    Returns:
        dict: {'mae': float, 'rmse': float, 'nasa_score': float}
    """
    y_t = np.asarray(y_true, dtype=float)
    y_p = np.asarray(y_pred, dtype=float)

    mae = float(mean_absolute_error(y_t, y_p))
    rmse = float(np.sqrt(mean_squared_error(y_t, y_p)))
    nasa = compute_nasa_score(y_t, y_p)

    logger.info(
        "Row-level evaluation -> MAE: %.3f, RMSE: %.3f, NASA: %.3f "
        "(n=%d rows)",
        mae, rmse, nasa, len(y_t),
    )
    return {"mae": mae, "rmse": rmse, "nasa_score": nasa}


def evaluate_endpoint(
    df: pd.DataFrame,
    y_pred_col: str = "predicted_rul",
    y_true_col: str = "RUL",
    engine_col: str = "engine_id",
    cycle_col: str = "cycle",
    rul_cap: Optional[float] = None,
) -> Dict[str, float]:
    """Endpoint-level evaluation matching the official C-MAPSS test protocol.

    The C-MAPSS test set is evaluated by predicting RUL at the LAST recorded
    cycle for each engine (one prediction per engine). This mimics operational
    reality where a maintenance decision is made at the current moment.

    Row-level validation (all cycles) inflates performance because early cycles
    with high RUL are trivially easy. Endpoint evaluation is the honest metric.

    Args:
        df: DataFrame containing engine_id, cycle, predicted_rul, and RUL columns.
        y_pred_col: Column name for model predictions.
        y_true_col: Column name for true RUL.
        engine_col: Column name for engine identifier.
        cycle_col: Column name for cycle counter.
        rul_cap: If provided, cap true RUL at this value before evaluation
                 (set to match training cap of 125 for fair comparison).

    Returns:
        dict: {'mae': float, 'rmse': float, 'nasa_score': float,
               'n_engines': int, 'late_pct': float, 'early_pct': float}
    """
    required = [engine_col, cycle_col, y_pred_col, y_true_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"evaluate_endpoint: missing columns {missing}")

    # Select last cycle per engine
    endpoint_df = (
        df.sort_values([engine_col, cycle_col])
        .groupby(engine_col, as_index=False)
        .last()
    )

    y_true = endpoint_df[y_true_col].to_numpy(dtype=float)
    y_pred = endpoint_df[y_pred_col].to_numpy(dtype=float)

    if rul_cap is not None:
        y_true = np.clip(y_true, 0, rul_cap)

    # Clamp predictions to non-negative (no negative RUL)
    y_pred = np.clip(y_pred, 0.0, None)

    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    nasa = compute_nasa_score(y_true, y_pred)

    d = y_pred - y_true
    late_pct = float(np.mean(d > 0)) * 100.0
    early_pct = float(np.mean(d < 0)) * 100.0
    mean_signed_error = float(np.mean(d))

    logger.info(
        "Endpoint evaluation -> MAE: %.3f, RMSE: %.3f, NASA: %.3f "
        "(n=%d engines, late=%.1f%%, early=%.1f%%)",
        mae, rmse, nasa, len(endpoint_df), late_pct, early_pct,
    )
    return {
        "mae": mae,
        "rmse": rmse,
        "nasa_score": nasa,
        "n_engines": int(len(endpoint_df)),
        "late_pct": late_pct,
        "early_pct": early_pct,
        "mean_signed_error": mean_signed_error,
    }


def evaluate_by_rul_range(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    bins: Optional[list] = None,
) -> Dict[str, Dict[str, float]]:
    """Compute MAE broken down by true RUL range (diagnostic tool).

    Args:
        y_true: True RUL values.
        y_pred: Predicted RUL values.
        bins: RUL range boundaries. Defaults to [0, 30, 60, 90, 125, inf].

    Returns:
        dict: RUL range label -> {'mae': float, 'rmse': float, 'count': int}
    """
    if bins is None:
        bins = [0, 30, 60, 90, 125, float("inf")]

    y_t = np.asarray(y_true, dtype=float)
    y_p = np.asarray(y_pred, dtype=float)
    results = {}

    for i in range(len(bins) - 1):
        lo, hi = bins[i], bins[i + 1]
        mask = (y_t >= lo) & (y_t < hi)
        count = int(mask.sum())
        label = f"RUL [{lo:.0f}, {hi:.0f})"

        if count == 0:
            results[label] = {"mae": float("nan"), "rmse": float("nan"), "count": 0}
            continue

        mae = float(mean_absolute_error(y_t[mask], y_p[mask]))
        rmse = float(np.sqrt(mean_squared_error(y_t[mask], y_p[mask])))
        results[label] = {"mae": mae, "rmse": rmse, "count": count}

    return results


# ── Abstract base class ───────────────────────────────────────────────────────

class BaseRULModel(ABC):
    """Abstract base class for Remaining Useful Life regression estimators."""

    @abstractmethod
    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
    ) -> "BaseRULModel":
        """Fit estimator to training features and target RUL."""

    @abstractmethod
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Predict RUL values for input feature matrix."""

    @abstractmethod
    def get_params(self) -> Dict[str, Any]:
        """Return estimator hyperparameters as dictionary."""

    @property
    @abstractmethod
    def model_type(self) -> str:
        """Return standard string identifier for model architecture."""

    def fit_timed(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
    ) -> float:
        """Fit model and return training duration in seconds."""
        t0 = time.time()
        self.fit(X, y)
        elapsed = time.time() - t0
        logger.info("Trained %s in %.2f seconds.", self.model_type, elapsed)
        return elapsed


# ── Model implementations ─────────────────────────────────────────────────────

class LinearRULModel(BaseRULModel):
    """Ordinary Least Squares Linear Regression baseline for RUL prediction."""

    def __init__(self, fit_intercept: bool = True, **kwargs: Any) -> None:
        self._fit_intercept = fit_intercept
        self._kwargs = kwargs
        self.model = LinearRegression(fit_intercept=fit_intercept, **kwargs)

    @property
    def model_type(self) -> str:
        return "linear_regression"

    def fit(self, X, y) -> "LinearRULModel":
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y, dtype=float)
        logger.info("Fitting LinearRULModel on shape %s", X_arr.shape)
        self.model.fit(X_arr, y_arr)
        return self

    def predict(self, X) -> np.ndarray:
        preds = self.model.predict(np.asarray(X, dtype=float))
        return np.clip(np.asarray(preds, dtype=float), 0.0, None)

    def get_params(self) -> Dict[str, Any]:
        params = self.model.get_params()
        params["model_type"] = self.model_type
        return params


class RandomForestRULModel(BaseRULModel):
    """Random Forest ensemble regressor — CPU-parallel with n_jobs=-1."""

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

    def fit(self, X, y) -> "RandomForestRULModel":
        X_arr, y_arr = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
        logger.info("Fitting RandomForestRULModel on shape %s", X_arr.shape)
        self.model.fit(X_arr, y_arr)
        return self

    def predict(self, X) -> np.ndarray:
        preds = self.model.predict(np.asarray(X, dtype=float))
        return np.clip(np.asarray(preds, dtype=float), 0.0, None)

    def get_params(self) -> Dict[str, Any]:
        params = self.model.get_params()
        params["model_type"] = self.model_type
        return params

    def feature_importances(self) -> np.ndarray:
        return self.model.feature_importances_


class GradientBoostingRULModel(BaseRULModel):
    """Gradient Boosted Decision Trees — sequential residual minimisation (CPU)."""

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 4,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        min_samples_leaf: int = 10,
        random_state: int = 42,
        **kwargs: Any,
    ) -> None:
        self.model = GradientBoostingRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            **kwargs,
        )

    @property
    def model_type(self) -> str:
        return "gradient_boosting"

    def fit(self, X, y) -> "GradientBoostingRULModel":
        X_arr, y_arr = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
        logger.info("Fitting GradientBoostingRULModel on shape %s", X_arr.shape)
        self.model.fit(X_arr, y_arr)
        return self

    def predict(self, X) -> np.ndarray:
        preds = self.model.predict(np.asarray(X, dtype=float))
        return np.clip(np.asarray(preds, dtype=float), 0.0, None)

    def get_params(self) -> Dict[str, Any]:
        params = self.model.get_params()
        params["model_type"] = self.model_type
        return params

    def feature_importances(self) -> np.ndarray:
        return self.model.feature_importances_


class ExtraTreesRULModel(BaseRULModel):
    """Extra Randomised Trees — faster training than RF, CPU-parallel."""

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: Optional[int] = 12,
        min_samples_split: int = 4,
        min_samples_leaf: int = 2,
        random_state: int = 42,
        n_jobs: int = -1,
        **kwargs: Any,
    ) -> None:
        self.model = ExtraTreesRegressor(
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
        return "extra_trees"

    def fit(self, X, y) -> "ExtraTreesRULModel":
        X_arr, y_arr = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
        logger.info("Fitting ExtraTreesRULModel on shape %s", X_arr.shape)
        self.model.fit(X_arr, y_arr)
        return self

    def predict(self, X) -> np.ndarray:
        preds = self.model.predict(np.asarray(X, dtype=float))
        return np.clip(np.asarray(preds, dtype=float), 0.0, None)

    def get_params(self) -> Dict[str, Any]:
        params = self.model.get_params()
        params["model_type"] = self.model_type
        return params

    def feature_importances(self) -> np.ndarray:
        return self.model.feature_importances_


class HistGradientBoostingRULModel(BaseRULModel):
    """Histogram-based Gradient Boosting — fastest GB on CPU, handles NaN natively."""

    def __init__(
        self,
        max_iter: int = 300,
        max_depth: Optional[int] = 6,
        learning_rate: float = 0.05,
        min_samples_leaf: int = 20,
        l2_regularization: float = 0.1,
        random_state: int = 42,
        **kwargs: Any,
    ) -> None:
        self.model = HistGradientBoostingRegressor(
            max_iter=max_iter,
            max_depth=max_depth,
            learning_rate=learning_rate,
            min_samples_leaf=min_samples_leaf,
            l2_regularization=l2_regularization,
            random_state=random_state,
            **kwargs,
        )

    @property
    def model_type(self) -> str:
        return "hist_gradient_boosting"

    def fit(self, X, y) -> "HistGradientBoostingRULModel":
        X_arr, y_arr = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
        logger.info("Fitting HistGradientBoostingRULModel on shape %s", X_arr.shape)
        self.model.fit(X_arr, y_arr)
        return self

    def predict(self, X) -> np.ndarray:
        preds = self.model.predict(np.asarray(X, dtype=float))
        return np.clip(np.asarray(preds, dtype=float), 0.0, None)

    def get_params(self) -> Dict[str, Any]:
        params = self.model.get_params()
        params["model_type"] = self.model_type
        return params


class XGBoostRULModel(BaseRULModel):
    """XGBoost regressor — CPU-only via tree_method='hist'. Falls back gracefully if unavailable."""

    def __init__(
        self,
        n_estimators: int = 300,
        max_depth: int = 5,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_alpha: float = 0.1,
        reg_lambda: float = 1.0,
        random_state: int = 42,
        **kwargs: Any,
    ) -> None:
        try:
            import xgboost as xgb  # type: ignore
            self.model = xgb.XGBRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                subsample=subsample,
                colsample_bytree=colsample_bytree,
                reg_alpha=reg_alpha,
                reg_lambda=reg_lambda,
                random_state=random_state,
                tree_method="hist",     # CPU histogram method
                device="cpu",           # Explicit CPU
                verbosity=0,
                **kwargs,
            )
            self._available = True
            logger.info("XGBoost available — using CPU hist method.")
        except ImportError:
            self._available = False
            logger.warning("XGBoost not installed. XGBoostRULModel will raise on fit/predict.")

    @property
    def model_type(self) -> str:
        return "xgboost"

    def fit(self, X, y) -> "XGBoostRULModel":
        if not self._available:
            raise RuntimeError("XGBoost is not installed. Install with: pip install xgboost")
        X_arr, y_arr = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
        logger.info("Fitting XGBoostRULModel on shape %s", X_arr.shape)
        self.model.fit(X_arr, y_arr)
        return self

    def predict(self, X) -> np.ndarray:
        if not self._available:
            raise RuntimeError("XGBoost is not installed.")
        preds = self.model.predict(np.asarray(X, dtype=float))
        return np.clip(np.asarray(preds, dtype=float), 0.0, None)

    def get_params(self) -> Dict[str, Any]:
        if not self._available:
            return {"model_type": self.model_type, "available": False}
        params = self.model.get_params()
        params["model_type"] = self.model_type
        return params

    def feature_importances(self) -> np.ndarray:
        return self.model.feature_importances_


# ── Ensemble uncertainty ──────────────────────────────────────────────────────

def compute_ensemble_uncertainty(
    models: list,
    X: Union[np.ndarray, pd.DataFrame],
    clip_min: float = 0.0,
) -> Dict[str, np.ndarray]:
    """Compute ensemble mean prediction and uncertainty (std) across multiple models.

    Uses prediction variance across the ensemble as a proxy for uncertainty.
    This is a CPU-efficient alternative to bootstrap or MC-Dropout.

    Args:
        models: List of fitted BaseRULModel instances.
        X: Feature matrix.
        clip_min: Minimum value to clip predictions (default 0 — no negative RUL).

    Returns:
        dict: {
            'mean': np.ndarray — ensemble mean prediction,
            'std': np.ndarray — prediction std (uncertainty proxy),
            'lower': np.ndarray — mean - 1.64*std (90% lower bound estimate),
            'upper': np.ndarray — mean + 1.64*std (90% upper bound estimate),
        }
    """
    X_arr = np.asarray(X, dtype=float)
    all_preds = []

    for model in models:
        preds = np.clip(model.predict(X_arr), clip_min, None)
        all_preds.append(preds)

    stack = np.vstack(all_preds)   # shape: (n_models, n_samples)
    mean_pred = np.mean(stack, axis=0)
    std_pred = np.std(stack, axis=0)

    # 1.64-sigma approximates 90% coverage for a normal-like distribution
    lower = np.clip(mean_pred - 1.64 * std_pred, clip_min, None)
    upper = mean_pred + 1.64 * std_pred

    return {
        "mean": mean_pred,
        "std": std_pred,
        "lower": lower,
        "upper": upper,
    }
