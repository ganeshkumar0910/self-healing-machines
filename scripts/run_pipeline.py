"""Command Line Interface (CLI) for executing the predictive maintenance pipeline end-to-end."""

import argparse
from datetime import datetime, timezone
import logging
from pathlib import Path
import sys
import time
from typing import Any, Dict, Optional

# Add project root to path for src package imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.agent.triage_agent import TriageAgent, TriageDecision
from src.data.loader import get_project_root, load_config, load_raw_cmapss, load_rul_labels
from src.data.preprocessor import DataPreprocessor
from src.data.rul_calculator import compute_test_rul, compute_train_rul
from src.data.splitter import engine_wise_split
from src.data.validator import run_full_validation
from src.database.db_manager import DatabaseManager
from src.features.engineer import FeatureEngineer
from src.maintenance.work_order_service import WorkOrderService
from src.mlops.experiment_tracker import ExperimentTracker
from src.models.anomaly_detector import AnomalyDetector
from src.models.model_registry import (
    get_active_model_path,
    save_model_artifact,
    set_active_model,
)
from src.models.prediction_service import PredictionService
from src.models.rul_model import (
    GradientBoostingRULModel,
    LinearRULModel,
    RandomForestRULModel,
    evaluate_model,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("run_pipeline")


def resolve_cmapss_file(root: Path, relative_dir: str, filename: str) -> Path:
    """Resolve file path inside raw CMAPSS data directory."""
    path = (root / relative_dir / filename).resolve()
    if path.is_file():
        return path
    # Try looking in sibling directory
    alt_path = (root.parent / "CMAPSSData" / filename).resolve()
    if alt_path.is_file():
        return alt_path
    raise FileNotFoundError(f"Could not locate CMAPSS file: {filename} at {path} or {alt_path}")


def step_preprocess(config: Dict[str, Any], dataset_id: str = "FD001") -> Dict[str, Path]:
    """Execute data ingestion, schema validation, RUL labeling, and feature engineering."""
    logger.info("=== STEP 1: PREPROCESS DATA [%s] ===", dataset_id)
    t0 = time.time()
    root = get_project_root()

    raw_dir_rel = config.get("paths", {}).get("raw_data_dir", "../CMAPSSData")
    train_file = resolve_cmapss_file(root, raw_dir_rel, f"train_{dataset_id}.txt")
    test_file = resolve_cmapss_file(root, raw_dir_rel, f"test_{dataset_id}.txt")
    rul_file = resolve_cmapss_file(root, raw_dir_rel, f"RUL_{dataset_id}.txt")

    # Ingest raw data
    df_train_raw = load_raw_cmapss(train_file, dataset_id=dataset_id)
    df_test_raw = load_raw_cmapss(test_file, dataset_id=dataset_id)
    rul_labels = load_rul_labels(rul_file)

    # Validate training data
    expected_cols = config.get("dataset", {}).get("column_names")
    val_summary = run_full_validation(df_train_raw, expected_cols)
    if not val_summary["valid"]:
        logger.warning("Data validation warnings: %s", val_summary["issues"])

    # Compute RUL targets
    cap_val = (
        config.get("rul", {}).get("cap_value", 125)
        if config.get("rul", {}).get("use_capped", True)
        else None
    )
    df_train_rul = compute_train_rul(df_train_raw, cap_value=cap_val)
    df_test_rul = compute_test_rul(df_test_raw, rul_labels=rul_labels)

    # Engineer rolling and degradation trend features
    fe_cfg = config.get("features", {})
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

    df_train_features = fe.build_features(df_train_rul)
    df_test_features = fe.build_features(df_test_rul)

    # Fit preprocessor on training feature columns only
    feature_cols = fe.get_feature_columns()
    preprocessor = DataPreprocessor(feature_cols=feature_cols)
    preprocessor.fit(df_train_features)

    # Save processed datasets
    proc_rel = config.get("paths", {}).get("processed_data_dir", "data/processed")
    proc_dir = (root / proc_rel).resolve()
    proc_dir.mkdir(parents=True, exist_ok=True)

    train_out = proc_dir / f"train_{dataset_id}_features.csv"
    test_out = proc_dir / f"test_{dataset_id}_features.csv"
    preproc_out = proc_dir / f"preprocessor_{dataset_id}.joblib"

    df_train_features.to_csv(train_out, index=False)
    df_test_features.to_csv(test_out, index=False)
    preprocessor.save(preproc_out)

    logger.info(
        "Preprocess completed in %.2fs. Outputs saved to %s",
        time.time() - t0,
        proc_dir,
    )
    return {
        "train_features": train_out,
        "test_features": test_out,
        "preprocessor": preproc_out,
    }


def step_train(config: Dict[str, Any], dataset_id: str = "FD001") -> Dict[str, Any]:
    """Train Linear, Random Forest, and Gradient Boosting models, evaluate, and save artifacts."""
    logger.info("=== STEP 2: TRAIN MODELS [%s] ===", dataset_id)
    t0 = time.time()
    root = get_project_root()
    proc_rel = config.get("paths", {}).get("processed_data_dir", "data/processed")
    proc_dir = (root / proc_rel).resolve()

    train_path = proc_dir / f"train_{dataset_id}_features.csv"
    test_path = proc_dir / f"test_{dataset_id}_features.csv"
    preproc_path = proc_dir / f"preprocessor_{dataset_id}.joblib"

    if not train_path.is_file() or not test_path.is_file() or not preproc_path.is_file():
        logger.info("Processed features missing. Executing preprocessing step first...")
        step_preprocess(config, dataset_id=dataset_id)

    df_train_features = pd.read_csv(train_path)
    df_test_features = pd.read_csv(test_path)
    preprocessor = DataPreprocessor.load(preproc_path)
    feature_cols = preprocessor.feature_cols

    # Split train features engine-wise for validation
    split_cfg = config.get("split", {})
    val_frac = float(split_cfg.get("validation_fraction", 0.2))
    seed = int(split_cfg.get("random_seed", 42))
    train_split, val_split = engine_wise_split(
        df_train_features, val_fraction=val_frac, random_seed=seed
    )

    X_train = preprocessor.transform(train_split)
    y_train = train_split["RUL"].to_numpy(dtype=float)

    X_val = preprocessor.transform(val_split)
    y_val = val_split["RUL"].to_numpy(dtype=float)

    X_test = preprocessor.transform(df_test_features)
    y_test = df_test_features["RUL"].to_numpy(dtype=float)

    # Train Isolation Forest Anomaly Detector on nominal training telemetry
    anom_cfg = config.get("anomaly", {})
    anomaly_detector = AnomalyDetector(
        contamination=float(anom_cfg.get("contamination", 0.05)),
        n_estimators=int(anom_cfg.get("n_estimators", 100)),
        random_state=int(anom_cfg.get("random_state", 42)),
    )
    anomaly_detector.fit(X_train)
    anomaly_detector.calibrate_threshold(
        X_val, percentile=float(anom_cfg.get("threshold_percentile", 10))
    )
    anom_path = proc_dir / f"anomaly_detector_{dataset_id}.joblib"
    anomaly_detector.save(anom_path)

    # Setup MLflow Tracker
    mlflow_cfg = config.get("mlflow", {})
    tracker_uri = root / config.get("paths", {}).get("mlflow_tracking_uri", "mlruns")
    exp_tracker = ExperimentTracker(
        tracking_uri=tracker_uri,
        experiment_name=mlflow_cfg.get("experiment_name", "self_healing_machines"),
    )

    models_cfg = config.get("models", {})
    candidates = [
        LinearRULModel(**models_cfg.get("linear_regression", {})),
        RandomForestRULModel(**models_cfg.get("random_forest", {})),
        GradientBoostingRULModel(**models_cfg.get("gradient_boosting", {})),
    ]

    results: Dict[str, Any] = {}
    best_model = None
    best_val_mae = float("inf")
    best_name = ""

    artifact_rel = config.get("paths", {}).get("artifacts_dir", "artifacts/models")
    artifact_dir = (root / artifact_rel).resolve()

    logger.info("Evaluating %d candidate models...", len(candidates))
    for model in candidates:
        m_name = model.model_type
        logger.info("--- Training: %s ---", m_name)
        model.fit(X_train, y_train)

        val_preds = model.predict(X_val)
        val_metrics = evaluate_model(y_val, val_preds)

        test_preds = model.predict(X_test)
        test_metrics = evaluate_model(y_test, test_preds)

        results[m_name] = {
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
        }

        # Log run to MLflow
        with exp_tracker:
            exp_tracker.start_run(run_name=f"{m_name}_{dataset_id}")
            exp_tracker.log_params(model.get_params())
            exp_tracker.log_metrics({f"val_{k}": v for k, v in val_metrics.items()})
            exp_tracker.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})

        if val_metrics["mae"] < best_val_mae:
            best_val_mae = val_metrics["mae"]
            best_model = model
            best_name = m_name

    logger.info("Best model selected: %s (Validation MAE: %.3f)", best_name, best_val_mae)

    # Save active best model artifact
    version_id = f"{best_name}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    saved_path = save_model_artifact(
        model=best_model,
        preprocessor=preprocessor,
        feature_cols=feature_cols,
        metadata={
            "model_type": best_name,
            "validation_metrics": results[best_name]["val_metrics"],
            "test_metrics": results[best_name]["test_metrics"],
            "dataset_id": dataset_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
        artifact_dir=artifact_dir,
        model_name=version_id,
    )
    set_active_model(artifact_dir, version_id)

    # Print summary metrics table
    print("\n" + "=" * 70)
    print(f" MODEL EVALUATION SUMMARY — DATASET {dataset_id}")
    print("=" * 70)
    print(f"{'Model':<22} | {'Val MAE':<10} | {'Val RMSE':<10} | {'Test MAE':<10} | {'Test RMSE':<10}")
    print("-" * 70)
    for name, res in results.items():
        v = res["val_metrics"]
        t = res["test_metrics"]
        print(f"{name:<22} | {v['mae']:<10.3f} | {v['rmse']:<10.3f} | {t['mae']:<10.3f} | {t['rmse']:<10.3f}")
    print("=" * 70)
    print(f"Active Production Model: {version_id} at {saved_path}\n")

    logger.info("Training step completed in %.2fs", time.time() - t0)
    return {
        "best_model_name": best_name,
        "version_id": version_id,
        "artifact_path": saved_path,
        "results": results,
    }


def step_predict(config: Dict[str, Any], dataset_id: str = "FD001") -> Path:
    """Generate RUL predictions and anomaly flags across the fleet."""
    logger.info("=== STEP 3: PREDICT FLEET [%s] ===", dataset_id)
    t0 = time.time()
    root = get_project_root()
    proc_rel = config.get("paths", {}).get("processed_data_dir", "data/processed")
    proc_dir = (root / proc_rel).resolve()
    artifact_rel = config.get("paths", {}).get("artifacts_dir", "artifacts/models")
    artifact_dir = (root / artifact_rel).resolve()

    test_path = proc_dir / f"test_{dataset_id}_features.csv"
    if not test_path.is_file():
        step_preprocess(config, dataset_id=dataset_id)

    df_test_features = pd.read_csv(test_path)

    # Load anomaly detector if available
    anom_path = proc_dir / f"anomaly_detector_{dataset_id}.joblib"
    anomaly_detector = None
    if anom_path.is_file():
        anomaly_detector = AnomalyDetector.load(anom_path)

    prediction_service = PredictionService(
        artifact_dir=artifact_dir,
        anomaly_detector=anomaly_detector,
    )

    df_predictions = prediction_service.predict_batch(df_test_features)
    pred_out = proc_dir / "predictions.csv"
    df_predictions.to_csv(pred_out, index=False)

    logger.info(
        "Predict completed in %.2fs. %d predictions saved to %s",
        time.time() - t0,
        len(df_predictions),
        pred_out,
    )
    return pred_out


def step_triage(config: Dict[str, Any], dataset_id: str = "FD001") -> Path:
    """Run autonomous triage agent on latest observations and output operational actions."""
    logger.info("=== STEP 4: AUTONOMOUS TRIAGE AGENT [%s] ===", dataset_id)
    t0 = time.time()
    root = get_project_root()
    proc_rel = config.get("paths", {}).get("processed_data_dir", "data/processed")
    proc_dir = (root / proc_rel).resolve()
    pred_path = proc_dir / "predictions.csv"

    if not pred_path.is_file():
        step_predict(config, dataset_id=dataset_id)

    df_predictions = pd.read_csv(pred_path)

    # Filter to most recent observation cycle per engine
    latest_df = (
        df_predictions.sort_values(["engine_id", "cycle"])
        .groupby("engine_id", as_index=False)
        .last()
    )

    artifact_rel = config.get("paths", {}).get("artifacts_dir", "artifacts/models")
    artifact_dir = (root / artifact_rel).resolve()
    try:
        active_ver = get_active_model_path(artifact_dir)
    except Exception:
        active_ver = "unknown"

    triage_cfg = config.get("triage", {})
    agent = TriageAgent(
        critical_rul=int(triage_cfg.get("critical_rul_threshold", 15)),
        warning_rul=int(triage_cfg.get("warning_rul_threshold", 40)),
        anomaly_threshold=float(triage_cfg.get("anomaly_score_threshold", -0.05)),
        model_version=active_ver,
    )

    decisions = agent.evaluate_batch(latest_df)
    decisions_df = pd.DataFrame([d.to_dict() for d in decisions])

    triage_out = proc_dir / "triage_decisions.csv"
    decisions_df.to_csv(triage_out, index=False)

    # Print summary counts by severity
    severity_counts = decisions_df["severity"].value_counts().to_dict()
    logger.info(
        "Triage complete in %.2fs. Fleet health distribution: %s",
        time.time() - t0,
        severity_counts,
    )
    return triage_out


def step_work_orders(config: Dict[str, Any]) -> Dict[str, Any]:
    """Ingest triage decisions and generate/audit maintenance work orders in SQLite DB."""
    logger.info("=== STEP 5: WORK ORDER GENERATION & LIFECYCLE ===")
    t0 = time.time()
    root = get_project_root()
    proc_rel = config.get("paths", {}).get("processed_data_dir", "data/processed")
    proc_dir = (root / proc_rel).resolve()
    triage_path = proc_dir / "triage_decisions.csv"

    if not triage_path.is_file():
        step_triage(config)

    decisions_df = pd.read_csv(triage_path)

    db_rel = config.get("paths", {}).get("database", "data/maintenance.db")
    db_path = (root / db_rel).resolve()
    db_manager = DatabaseManager(db_path)
    wo_service = WorkOrderService(db_manager)

    decisions = []
    for _, row in decisions_df.iterrows():
        decisions.append(
            TriageDecision(
                engine_id=int(row["engine_id"]),
                cycle=int(row["cycle"]),
                predicted_rul=float(row["predicted_rul"]),
                anomaly_score=float(row["anomaly_score"]),
                anomaly_threshold=float(row["anomaly_threshold"]),
                severity=str(row["severity"]),
                recommended_action=str(row["recommended_action"]),
                triggered_rule=str(row["triggered_rule"]),
                explanation=str(row["explanation"]),
                model_version=str(row["model_version"]),
                timestamp=str(row["timestamp"]),
            )
        )

    results = wo_service.process_triage_decisions(decisions)
    created_count = sum(1 for r in results if r["created"])
    fleet_summary = wo_service.get_fleet_summary()

    print("\n" + "=" * 50)
    print(" FLEET MAINTENANCE WORK ORDER SUMMARY")
    print("=" * 50)
    print(f"Total Work Orders Recorded : {fleet_summary['total']}")
    print(f"Open Orders                : {fleet_summary['open']}")
    print(f"In-Progress Orders         : {fleet_summary['in_progress']}")
    print(f"Closed Orders              : {fleet_summary['closed']}")
    print("Breakdown by Severity:")
    for sev, cnt in fleet_summary["by_severity"].items():
        print(f"  • {sev:<15}: {cnt}")
    print("=" * 50 + "\n")

    logger.info(
        "Work orders completed in %.2fs. Newly created: %d",
        time.time() - t0,
        created_count,
    )
    return fleet_summary


def step_all(config: Dict[str, Any], dataset_id: str = "FD001") -> None:
    """Execute complete end-to-end pipeline."""
    total_start = time.time()
    logger.info(">>> STARTING FULL PREDICTIVE MAINTENANCE PIPELINE [%s] <<<", dataset_id)

    step_preprocess(config, dataset_id=dataset_id)
    step_train(config, dataset_id=dataset_id)
    step_predict(config, dataset_id=dataset_id)
    step_triage(config, dataset_id=dataset_id)
    step_work_orders(config)

    logger.info(
        ">>> FULL PIPELINE EXECUTED SUCCESSFULLY IN %.2fs <<<",
        time.time() - total_start,
    )


def main() -> None:
    """CLI Entry Point."""
    parser = argparse.ArgumentParser(
        description="Self-Healing Machines Predictive Maintenance Pipeline CLI"
    )
    parser.add_argument(
        "--step",
        choices=["preprocess", "train", "predict", "triage", "work_orders", "all"],
        default="all",
        help="Pipeline step to execute (default: all)",
    )
    parser.add_argument(
        "--dataset",
        default="FD001",
        help="CMAPSS dataset subset ID (default: FD001)",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to custom settings.yaml configuration file",
    )

    args = parser.parse_args()
    config = load_config(args.config)

    if args.step == "preprocess":
        step_preprocess(config, dataset_id=args.dataset)
    elif args.step == "train":
        step_train(config, dataset_id=args.dataset)
    elif args.step == "predict":
        step_predict(config, dataset_id=args.dataset)
    elif args.step == "triage":
        step_triage(config, dataset_id=args.dataset)
    elif args.step == "work_orders":
        step_work_orders(config)
    elif args.step == "all":
        step_all(config, dataset_id=args.dataset)


if __name__ == "__main__":
    main()
