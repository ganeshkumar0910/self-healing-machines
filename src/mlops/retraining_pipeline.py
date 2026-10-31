"""Automated continuous retraining pipeline with model challenger-champion promotion."""

from datetime import datetime, timezone
import logging
from pathlib import Path
import shutil
from typing import Any, Dict, Optional, Union
import numpy as np

from src.data.loader import get_project_root, load_raw_cmapss
from src.data.preprocessor import DataPreprocessor
from src.data.rul_calculator import compute_train_rul
from src.data.splitter import engine_wise_split
from src.data.validator import run_full_validation
from src.features.engineer import FeatureEngineer
from src.mlops.experiment_tracker import ExperimentTracker
from src.models.model_registry import (
    get_active_model_path,
    load_model_artifact,
    save_model_artifact,
    set_active_model,
)
from src.models.rul_model import RandomForestRULModel, evaluate_model

logger = logging.getLogger(__name__)


class RetrainingPipeline:
    """Orchestrates end-to-end retraining, validation, benchmarking, and conditional promotion."""

    def __init__(
        self,
        config: Dict[str, Any],
        artifact_dir: Union[str, Path],
        tracker: ExperimentTracker,
    ) -> None:
        """Initialize RetrainingPipeline.

        Args:
            config: Full application settings dictionary.
            artifact_dir: Directory path where model artifacts are housed.
            tracker: ExperimentTracker instance for MLflow audit logs.
        """
        self.config = config
        self.artifact_dir = Path(artifact_dir).resolve()
        self.tracker = tracker
        self.project_root = get_project_root()

    def _should_promote(
        self, candidate_mae: float, active_mae: float, min_improvement: float
    ) -> bool:
        """Determine whether candidate model improves enough over champion to warrant promotion.

        Args:
            candidate_mae: Mean Absolute Error of candidate model on validation set.
            active_mae: Mean Absolute Error of currently deployed champion model.
            min_improvement: Minimum relative fractional improvement required (e.g. 0.01 = 1%).

        Returns:
            bool: True if candidate beats champion by >= min_improvement.
        """
        if active_mae <= 0.0:
            return False
        relative_improvement = (active_mae - candidate_mae) / active_mae
        logger.info(
            "Challenger MAE: %.3f | Champion MAE: %.3f | Relative Improvement: %.2f%% (required: %.2f%%)",
            candidate_mae,
            active_mae,
            relative_improvement * 100.0,
            min_improvement * 100.0,
        )
        return relative_improvement >= min_improvement

    def _backup_active_model(self, artifact_dir: Union[str, Path]) -> Optional[str]:
        """Create a timestamped backup copy of current active production model directory.

        Args:
            artifact_dir: Base directory for model artifacts.

        Returns:
            str or None: Path to created backup directory.
        """
        base_dir = Path(artifact_dir).resolve()
        try:
            active_name = get_active_model_path(base_dir)
        except Exception as exc:
            logger.info("No active model to back up: %s", exc)
            return None

        active_path = base_dir / active_name
        if not active_path.is_dir():
            logger.warning("Active model path %s does not exist on disk.", active_path)
            return None

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        backup_name = f"{active_name}_backup_{timestamp}"
        backup_path = base_dir / backup_name

        shutil.copytree(active_path, backup_path)
        logger.info("Active model backed up from %s to %s", active_path, backup_path)
        return str(backup_path)

    def run(self, run_name: Optional[str] = None) -> Dict[str, Any]:
        """Execute end-to-end retraining, benchmarking, logging, and conditional promotion.

        1. Ingest raw FD001 training data
        2. Validate data schema and quality
        3. Compute target RUL with piecewise cap
        4. Engineer rolling & trend degradation features
        5. Engine-wise split into train and validation sets
        6. Fit preprocessor on train partition
        7. Train candidate RandomForestRULModel
        8. Evaluate candidate on validation partition
        9. Benchmark against deployed champion model if exists
        10. Promote and register candidate if criteria met or champion absent
        11. Log metrics and artifacts to MLflow

        Args:
            run_name: Optional name for the MLflow run.

        Returns:
            dict: Comprehensive pipeline execution report.
        """
        logger.info("=== Starting Automated Retraining Pipeline ===")
        start_time = datetime.now(timezone.utc)

        # 1. Load data
        raw_rel = self.config.get("paths", {}).get("raw_data_dir", "../CMAPSSData")
        raw_file = (self.project_root / raw_rel / "train_FD001.txt").resolve()
        if not raw_file.is_file():
            # Fallback direct check
            raw_file = (self.project_root.parent / "CMAPSSData" / "train_FD001.txt").resolve()

        raw_df = load_raw_cmapss(raw_file, dataset_id="FD001")

        # 2. Validate
        expected_cols = self.config.get("dataset", {}).get(
            "column_names",
            ["engine_id", "cycle", "op_setting_1", "op_setting_2", "op_setting_3"]
            + [f"sensor_{i}" for i in range(1, 22)],
        )
        val_res = run_full_validation(raw_df, expected_cols)
        if not val_res["valid"]:
            logger.warning("Validation warnings detected: %s", val_res["issues"])

        # 3. Compute RUL
        cap_val = (
            self.config.get("rul", {}).get("cap_value", 125)
            if self.config.get("rul", {}).get("use_capped", True)
            else None
        )
        df_rul = compute_train_rul(raw_df, cap_value=cap_val)

        # 4. Feature engineering
        fe_cfg = self.config.get("features", {})
        fe = FeatureEngineer(
            selected_sensors=fe_cfg.get(
                "selected_sensors",
                ["sensor_2", "sensor_3", "sensor_4", "sensor_7", "sensor_8", "sensor_9",
                 "sensor_11", "sensor_12", "sensor_13", "sensor_14", "sensor_15", "sensor_17",
                 "sensor_20", "sensor_21"]
            ),
            rolling_windows=fe_cfg.get("rolling_windows", [5, 10, 20]),
            include_op_settings=fe_cfg.get("include_op_settings", True),
            trend_window=fe_cfg.get("trend_window", 5),
        )
        df_features = fe.build_features(df_rul)

        # 5. Split train/validation
        split_cfg = self.config.get("split", {})
        val_frac = float(split_cfg.get("validation_fraction", 0.2))
        seed = int(split_cfg.get("random_seed", 42))
        train_df, val_df = engine_wise_split(
            df_features, val_fraction=val_frac, random_seed=seed
        )

        # 6. Fit preprocessor
        feature_cols = fe.get_feature_columns()
        preprocessor = DataPreprocessor(feature_cols=feature_cols)
        X_train = preprocessor.fit_transform(train_df)
        y_train = train_df["RUL"].to_numpy(dtype=float)

        X_val = preprocessor.transform(val_df)
        y_val = val_df["RUL"].to_numpy(dtype=float)

        # 7. Train candidate model
        rf_cfg = self.config.get("models", {}).get("random_forest", {})
        candidate = RandomForestRULModel(
            n_estimators=int(rf_cfg.get("n_estimators", 100)),
            max_depth=rf_cfg.get("max_depth", 10),
            min_samples_split=int(rf_cfg.get("min_samples_split", 5)),
            min_samples_leaf=int(rf_cfg.get("min_samples_leaf", 2)),
            random_state=int(rf_cfg.get("random_state", 42)),
            n_jobs=int(rf_cfg.get("n_jobs", -1)),
        )
        candidate.fit(X_train, y_train)

        # 8. Evaluate candidate
        cand_preds = candidate.predict(X_val)
        candidate_metrics = evaluate_model(y_val, cand_preds)

        # 9. Check champion model
        active_model_name: Optional[str] = None
        active_metrics: Optional[Dict[str, float]] = None
        try:
            active_model_name = get_active_model_path(self.artifact_dir)
            active_bundle = load_model_artifact(self.artifact_dir, active_model_name)
            active_model = active_bundle["model"]
            active_preprocessor = active_bundle["preprocessor"]

            X_val_active = active_preprocessor.transform(val_df)
            active_preds = active_model.predict(X_val_active)
            active_metrics = evaluate_model(y_val, active_preds)
            logger.info("Champion model '%s' evaluated on validation split.", active_model_name)
        except Exception as exc:
            logger.info("No active champion model found or failed to evaluate: %s", exc)

        # 10. Promotion decision
        min_improvement = float(
            self.config.get("promotion", {}).get("min_mae_improvement_fraction", 0.01)
        )
        if active_metrics is None:
            should_promote = True
            promotion_reason = "No active champion model exists; auto-promoting candidate."
        else:
            should_promote = self._should_promote(
                candidate_metrics["mae"], active_metrics["mae"], min_improvement
            )
            promotion_reason = (
                f"Candidate MAE ({candidate_metrics['mae']:.2f}) vs Champion MAE ({active_metrics['mae']:.2f}) — "
                f"Promote: {should_promote}"
            )

        # Save and register model if promoted
        promoted_model_name = f"random_forest_{start_time.strftime('%Y%m%d_%H%M%S')}"
        saved_path = ""
        backup_path = None

        if should_promote:
            backup_path = self._backup_active_model(self.artifact_dir)
            saved_path = save_model_artifact(
                model=candidate,
                preprocessor=preprocessor,
                feature_cols=feature_cols,
                metadata={
                    "model_type": candidate.model_type,
                    "metrics": candidate_metrics,
                    "champion_metrics": active_metrics,
                    "promoted": True,
                    "promotion_reason": promotion_reason,
                    "created_at": start_time.isoformat(),
                },
                artifact_dir=self.artifact_dir,
                model_name=promoted_model_name,
            )
            set_active_model(self.artifact_dir, promoted_model_name)
            logger.info("Promoted candidate model as active: %s", promoted_model_name)
        else:
            logger.info("Candidate model did not outperform champion. Not promoted.")

        # 11. Log to MLflow
        if run_name is None:
            run_name = f"retraining_{start_time.strftime('%Y%m%d_%H%M%S')}"

        self.tracker.start_run(run_name=run_name)
        try:
            self.tracker.log_params(
                {
                    "model_type": candidate.model_type,
                    "promoted": should_promote,
                    "min_mae_improvement_threshold": min_improvement,
                    "val_engines_count": val_df["engine_id"].nunique(),
                    "train_engines_count": train_df["engine_id"].nunique(),
                }
            )
            self.tracker.log_metrics(
                {
                    "candidate_mae": candidate_metrics["mae"],
                    "candidate_rmse": candidate_metrics["rmse"],
                    "candidate_nasa_score": candidate_metrics["nasa_score"],
                }
            )
            if active_metrics:
                self.tracker.log_metrics(
                    {
                        "champion_mae": active_metrics["mae"],
                        "champion_rmse": active_metrics["rmse"],
                        "champion_nasa_score": active_metrics["nasa_score"],
                    }
                )
            if saved_path:
                self.tracker.log_model_artifact(saved_path)
        finally:
            self.tracker.end_run()

        result = {
            "status": "success",
            "candidate_metrics": candidate_metrics,
            "champion_metrics": active_metrics,
            "promoted": should_promote,
            "promoted_model_name": promoted_model_name if should_promote else None,
            "promotion_reason": promotion_reason,
            "model_path": saved_path if should_promote else None,
            "backup_path": backup_path,
        }
        logger.info("=== Retraining Pipeline Complete: %s ===", result)
        return result
