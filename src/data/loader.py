"""Data loader module for CMAPSS turbofan degradation datasets and project configuration."""

import logging
from pathlib import Path
from typing import Optional, Union
import pandas as pd
import yaml

logger = logging.getLogger(__name__)

DEFAULT_COLUMN_NAMES = [
    "engine_id",
    "cycle",
    "op_setting_1",
    "op_setting_2",
    "op_setting_3",
] + [f"sensor_{i}" for i in range(1, 22)]


def get_project_root() -> Path:
    """Find and return the root directory of the self-healing-machines project.

    Searches upward from the current file and current working directory for the
    directory containing config/settings.yaml.

    Returns:
        Path: Absolute path to the project root directory.

    Raises:
        FileNotFoundError: If config/settings.yaml cannot be located in any parent.
    """
    candidates = [Path(__file__).resolve().parent, Path.cwd().resolve()]
    for base in candidates:
        for parent in [base] + list(base.parents):
            if (parent / "config" / "settings.yaml").is_file():
                return parent
            if (parent / "self-healing-machines" / "config" / "settings.yaml").is_file():
                return parent / "self-healing-machines"

    raise FileNotFoundError(
        "Could not find project root containing config/settings.yaml."
    )


def load_config(config_path: Optional[Union[str, Path]] = None) -> dict:
    """Load configuration dictionary from YAML file.

    Args:
        config_path: Optional path to YAML configuration file. If None,
            loads config/settings.yaml relative to project root.

    Returns:
        dict: Parsed configuration dictionary.

    Raises:
        FileNotFoundError: If the config file does not exist.
        yaml.YAMLError: If YAML parsing fails.
    """
    if config_path is None:
        root_dir = get_project_root()
        config_file = root_dir / "config" / "settings.yaml"
    else:
        config_file = Path(config_path).resolve()

    if not config_file.is_file():
        raise FileNotFoundError(f"Config file not found at: {config_file}")

    logger.info("Loading configuration from: %s", config_file)
    with open(config_file, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValueError(f"Config file {config_file} did not parse to a dictionary.")

    return config


def load_raw_cmapss(
    filepath: Union[str, Path],
    dataset_id: str = "FD001"
) -> pd.DataFrame:
    """Read space-separated CMAPSS text file and format it into a DataFrame.

    Reads whitespace-delimited CMAPSS run-to-failure records without header,
    assigns the 26 standard column names from config or defaults, drops any
    extraneous trailing empty columns, and sets appropriate integer data types
    for engine_id and cycle.

    Args:
        filepath: Path to the raw CMAPSS text file (e.g. train_FD001.txt).
        dataset_id: Identifier of the dataset subset (e.g. 'FD001').

    Returns:
        pd.DataFrame: Formatted DataFrame with 26 named columns.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        ValueError: If file content does not match expected column count.
    """
    file_p = Path(filepath).resolve()
    if not file_p.is_file():
        raise FileNotFoundError(f"Raw CMAPSS file not found: {file_p}")

    logger.info("Loading CMAPSS dataset [%s] from %s", dataset_id, file_p)

    # Resolve column names from config if available, else default list
    try:
        cfg = load_config()
        column_names = cfg.get("dataset", {}).get("column_names", DEFAULT_COLUMN_NAMES)
    except Exception as exc:
        logger.warning(
            "Could not load column names from config (%s). Using default columns.",
            exc,
        )
        column_names = DEFAULT_COLUMN_NAMES

    # CMAPSS text files are separated by whitespace and often contain trailing spaces
    df = pd.read_csv(file_p, sep=r"\s+", header=None)

    # Drop columns that are completely NaN (often trailing whitespace parsed as extra col)
    df = df.dropna(how="all", axis=1)

    if df.shape[1] < len(column_names):
        raise ValueError(
            f"File {file_p} has {df.shape[1]} columns, expected at least {len(column_names)}"
        )

    # Keep only the first 26 columns if extra columns exist
    df = df.iloc[:, : len(column_names)]
    df.columns = list(column_names)

    # Convert identifier columns to int
    df["engine_id"] = df["engine_id"].astype(int)
    df["cycle"] = df["cycle"].astype(int)

    logger.info(
        "Loaded dataset [%s]: %d rows, %d engines, %d columns",
        dataset_id,
        len(df),
        df["engine_id"].nunique(),
        df.shape[1],
    )
    return df


def load_rul_labels(filepath: Union[str, Path]) -> pd.Series:
    """Read CMAPSS ground truth RUL label file (RUL_FDxxx.txt).

    Returns a pandas Series of remaining useful life values indexed from
    1 to N, corresponding to engine_id.

    Args:
        filepath: Path to the RUL_FDxxx.txt file.

    Returns:
        pd.Series: RUL values indexed by engine_id (1..N).

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If file is empty or improperly formatted.
    """
    file_p = Path(filepath).resolve()
    if not file_p.is_file():
        raise FileNotFoundError(f"RUL labels file not found: {file_p}")

    logger.info("Loading RUL labels from: %s", file_p)
    df = pd.read_csv(file_p, sep=r"\s+", header=None)
    df = df.dropna(how="all", axis=1)

    if df.empty or df.shape[1] < 1:
        raise ValueError(f"RUL file {file_p} is empty or invalid.")

    rul_series = df.iloc[:, 0].astype(float)
    # Reindex starting from 1 to N to match engine_id convention
    rul_series.index = range(1, len(rul_series) + 1)
    rul_series.index.name = "engine_id"
    rul_series.name = "RUL"

    logger.info("Loaded RUL labels for %d engines.", len(rul_series))
    return rul_series
