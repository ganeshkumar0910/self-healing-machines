"""Model registry for packaging, versioning, inspecting, and activating ML artifacts."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Union
import joblib

logger = logging.getLogger(__name__)


def save_model_artifact(
    model: Any,
    preprocessor: Any,
    feature_cols: List[str],
    metadata: Dict[str, Any],
    artifact_dir: Union[str, Path],
    model_name: str,
) -> str:
    """Save complete model packaging artifact bundle to versioned directory.

    Creates directory {artifact_dir}/{model_name}/ containing:
    - model.pkl: Serialized model estimator
    - preprocessor.pkl: Serialized preprocessor pipeline
    - feature_cols.json: Ordered list of feature names
    - metadata.json: Training metrics, timestamps, and parameters

    Args:
        model: Trained model instance.
        preprocessor: Fitted preprocessor instance.
        feature_cols: List of feature names expected by the model.
        metadata: Dictionary containing performance metrics, git hash, etc.
        artifact_dir: Base directory for model artifacts.
        model_name: Unique version or model identifier name.

    Returns:
        str: Absolute path to the saved model directory.
    """
    target_dir = Path(artifact_dir).resolve() / model_name
    target_dir.mkdir(parents=True, exist_ok=True)

    # 1. Save model
    model_path = target_dir / "model.pkl"
    joblib.dump(model, model_path)

    # 2. Save preprocessor
    preproc_path = target_dir / "preprocessor.pkl"
    if hasattr(preprocessor, "save") and callable(getattr(preprocessor, "save")):
        preprocessor.save(preproc_path)
    else:
        joblib.dump(preprocessor, preproc_path)

    # 3. Save feature columns
    cols_path = target_dir / "feature_cols.json"
    with open(cols_path, "w", encoding="utf-8") as f:
        json.dump(list(feature_cols), f, indent=2)

    # 4. Save metadata
    meta_path = target_dir / "metadata.json"
    meta_to_write = dict(metadata)
    meta_to_write["model_name"] = model_name
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta_to_write, f, indent=2)

    logger.info("Successfully saved model artifact bundle to: %s", target_dir)
    return str(target_dir)


def load_model_artifact(
    artifact_dir: Union[str, Path], model_name: str
) -> Dict[str, Any]:
    """Load model artifact bundle from disk.

    Args:
        artifact_dir: Base artifact directory.
        model_name: Name/version of the model artifact to load.

    Returns:
        dict: {
            'model': loaded_model,
            'preprocessor': loaded_preprocessor,
            'feature_cols': list[str],
            'metadata': dict
        }

    Raises:
        FileNotFoundError: If the target artifact bundle does not exist.
    """
    target_dir = Path(artifact_dir).resolve() / model_name
    if not target_dir.is_dir():
        raise FileNotFoundError(f"Model artifact directory not found: {target_dir}")

    model_path = target_dir / "model.pkl"
    preproc_path = target_dir / "preprocessor.pkl"
    cols_path = target_dir / "feature_cols.json"
    meta_path = target_dir / "metadata.json"

    if not model_path.is_file():
        raise FileNotFoundError(f"Missing model.pkl in {target_dir}")
    if not preproc_path.is_file():
        raise FileNotFoundError(f"Missing preprocessor.pkl in {target_dir}")
    if not cols_path.is_file():
        raise FileNotFoundError(f"Missing feature_cols.json in {target_dir}")

    model = joblib.load(model_path)
    preprocessor = joblib.load(preproc_path)

    with open(cols_path, "r", encoding="utf-8") as f:
        feature_cols = json.load(f)

    metadata: Dict[str, Any] = {}
    if meta_path.is_file():
        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

    logger.info("Successfully loaded model artifact [%s] from %s", model_name, target_dir)
    return {
        "model": model,
        "preprocessor": preprocessor,
        "feature_cols": feature_cols,
        "metadata": metadata,
    }


def list_model_versions(artifact_dir: Union[str, Path]) -> List[Dict[str, Any]]:
    """Enumerate all available model versions in the artifact repository.

    Args:
        artifact_dir: Base directory housing model artifacts.

    Returns:
        list[dict]: List of metadata dictionaries for each available model version.
    """
    base_dir = Path(artifact_dir).resolve()
    if not base_dir.is_dir():
        logger.warning("Artifact directory does not exist: %s", base_dir)
        return []

    versions = []
    for item in sorted(base_dir.iterdir()):
        if item.is_dir():
            meta_path = item / "metadata.json"
            if meta_path.is_file():
                try:
                    with open(meta_path, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                    if "model_name" not in meta:
                        meta["model_name"] = item.name
                    versions.append(meta)
                except Exception as exc:
                    logger.warning("Could not read metadata from %s: %s", meta_path, exc)
    return versions


def get_active_model_path(artifact_dir: Union[str, Path]) -> str:
    """Retrieve the name of the currently active model from active_model.txt.

    Args:
        artifact_dir: Base artifact directory.

    Returns:
        str: Active model name.

    Raises:
        FileNotFoundError: If active_model.txt does not exist or is empty.
    """
    base_dir = Path(artifact_dir).resolve()
    active_file = base_dir / "active_model.txt"

    if not active_file.is_file():
        raise FileNotFoundError(f"Active model pointer file not found: {active_file}")

    with open(active_file, "r", encoding="utf-8") as f:
        active_name = f.read().strip()

    if not active_name:
        raise ValueError(f"Active model file {active_file} is empty.")

    return active_name


def set_active_model(artifact_dir: Union[str, Path], model_name: str) -> None:
    """Set the specified model as the active production model in active_model.txt.

    Args:
        artifact_dir: Base artifact directory.
        model_name: Model identifier to activate.
    """
    base_dir = Path(artifact_dir).resolve()
    base_dir.mkdir(parents=True, exist_ok=True)
    active_file = base_dir / "active_model.txt"

    with open(active_file, "w", encoding="utf-8") as f:
        f.write(model_name.strip() + "\n")

    logger.info("Active model updated to '%s' in %s", model_name, active_file)
