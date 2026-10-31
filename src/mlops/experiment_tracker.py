"""MLflow experiment tracking wrapper for logging parameters, metrics, and model artifacts."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import mlflow
from mlflow.entities import Run
from mlflow.tracking import MlflowClient

logger = logging.getLogger(__name__)


class ExperimentTracker:
    """Wrapper around MLflow for standardized lifecycle tracking and artifact auditing."""

    def __init__(
        self,
        tracking_uri: Union[str, Path] = "mlruns",
        experiment_name: str = "self_healing_machines",
    ) -> None:
        """Initialize ExperimentTracker.

        Args:
            tracking_uri: Local filesystem path, SQLite URI, or remote URI.
                          Plain paths (e.g. 'mlruns') are auto-converted to a
                          SQLite URI because MLflow 3.x retired the file-store
                          backend. Use ``sqlite:///mlflow.db`` format directly
                          or supply any supported backend URI.
            experiment_name: Name of the MLflow experiment.
        """
        uri_str = str(tracking_uri)

        # MLflow 3.x deprecated the file-based tracking store.
        # Convert a plain directory path to a SQLite URI automatically.
        if not uri_str.startswith(("sqlite://", "mysql://", "postgresql://",
                                   "http://", "https://", "databricks")):
            # Place the SQLite DB alongside the supplied path (parent dir)
            base_dir = Path(uri_str).resolve().parent
            base_dir.mkdir(parents=True, exist_ok=True)
            db_path = base_dir / "mlflow.db"
            self.tracking_uri = f"sqlite:///{db_path.as_posix()}"
            logger.info("MLflow tracking URI converted to SQLite: %s", self.tracking_uri)
        else:
            self.tracking_uri = uri_str

        self.experiment_name = experiment_name

        mlflow.set_tracking_uri(self.tracking_uri)
        self.client = MlflowClient(tracking_uri=self.tracking_uri)

        exp = self.client.get_experiment_by_name(experiment_name)
        if exp is None:
            self.experiment_id = self.client.create_experiment(experiment_name)
            logger.info("Created new MLflow experiment '%s' (ID: %s)", experiment_name, self.experiment_id)
        else:
            self.experiment_id = exp.experiment_id
            logger.info("Using existing MLflow experiment '%s' (ID: %s)", experiment_name, self.experiment_id)

        mlflow.set_experiment(experiment_name)
        self.active_run: Optional[Run] = None

    def __enter__(self) -> "ExperimentTracker":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.end_run()

    def start_run(self, run_name: Optional[str] = None) -> Any:
        """Start a new MLflow tracking run.

        Args:
            run_name: Human-readable label for the run.

        Returns:
            mlflow.ActiveRun: Active MLflow run instance.
        """
        # Close any lingering run
        if mlflow.active_run() is not None:
            mlflow.end_run()

        self.active_run = mlflow.start_run(
            run_name=run_name,
            experiment_id=self.experiment_id,
        )
        logger.info("Started MLflow run '%s' (ID: %s)", run_name, self.active_run.info.run_id)
        return self.active_run

    def log_params(self, params: Dict[str, Any]) -> None:
        """Log hyperparameter dictionary to active run.

        Args:
            params: Dictionary of parameters (coerced to strings/primitives).
        """
        flat_params: Dict[str, Any] = {}
        for k, v in params.items():
            if isinstance(v, (int, float, str, bool)):
                flat_params[k] = v
            else:
                flat_params[k] = str(v)
        mlflow.log_params(flat_params)

    def log_metrics(self, metrics: Dict[str, float]) -> None:
        """Log scalar performance metrics to active run.

        Args:
            metrics: Dictionary of metric names and numeric values.
        """
        float_metrics = {k: float(v) for k, v in metrics.items() if v is not None}
        mlflow.log_metrics(float_metrics)

    def log_artifact(self, local_path: Union[str, Path]) -> None:
        """Upload a local file artifact to the MLflow run store.

        Args:
            local_path: Path to the local file to log.
        """
        file_p = Path(local_path).resolve()
        if not file_p.is_file():
            logger.warning("Artifact file not found: %s", file_p)
            return
        mlflow.log_artifact(str(file_p))
        logger.info("Logged artifact to MLflow: %s", file_p.name)

    def log_model_artifact(self, model_dir: Union[str, Path]) -> None:
        """Upload an entire directory containing model artifacts.

        Args:
            model_dir: Directory containing serialized model artifacts.
        """
        dir_p = Path(model_dir).resolve()
        if not dir_p.is_dir():
            logger.warning("Model directory not found: %s", dir_p)
            return
        mlflow.log_artifacts(str(dir_p), artifact_path="model_package")
        logger.info("Logged model artifact directory: %s", dir_p)

    def end_run(self) -> None:
        """Terminate the active MLflow run."""
        if mlflow.active_run() is not None:
            run_id = mlflow.active_run().info.run_id
            mlflow.end_run()
            logger.info("Terminated MLflow run: %s", run_id)
        self.active_run = None

    def get_experiment_runs(self) -> List[Dict[str, Any]]:
        """Retrieve summaries of all runs recorded in this experiment.

        Returns:
            list[dict]: List of run summaries containing run_id, run_name,
                status, metrics, params, and start_time.
        """
        runs = self.client.search_runs(experiment_ids=[self.experiment_id])
        run_summaries = []
        for r in runs:
            run_summaries.append(
                {
                    "run_id": r.info.run_id,
                    "run_name": r.data.tags.get("mlflow.runName", ""),
                    "status": r.info.status,
                    "metrics": r.data.metrics,
                    "params": r.data.params,
                    "start_time": r.info.start_time,
                }
            )
        return run_summaries

    def get_run_by_id(self, run_id: str) -> Dict[str, Any]:
        """Fetch metadata, metrics, and parameters for a specific run ID.

        Args:
            run_id: MLflow run identifier.

        Returns:
            dict: Run summary details.
        """
        r = self.client.get_run(run_id)
        return {
            "run_id": r.info.run_id,
            "run_name": r.data.tags.get("mlflow.runName", ""),
            "status": r.info.status,
            "metrics": r.data.metrics,
            "params": r.data.params,
            "start_time": r.info.start_time,
        }
