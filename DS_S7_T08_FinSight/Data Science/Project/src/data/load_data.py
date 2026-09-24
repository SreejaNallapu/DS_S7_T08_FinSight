"""Day 1 public API: obtain data, load its schema, and extract the raw target."""

from pathlib import Path

import pandas as pd

from src.data.acquire import download_dataset
from src.data.load import load_german_credit
from src.data.schema import COLUMN_NAMES, TARGET, TARGET_LABELS

DEFAULT_RAW_PATH = Path(__file__).resolve().parents[2] / "data/raw/german.data"


def load_data(path: str | Path = DEFAULT_RAW_PATH, *, force_download: bool = False) -> pd.DataFrame:
    """Obtain the checksum-verified official file, reusing a valid local cache."""
    return load_german_credit(download_dataset(Path(path), force=force_download))


def extract_features_target(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return independent copies; retain original target codes 1=good, 2=bad."""
    if list(frame.columns) != COLUMN_NAMES:
        raise ValueError("Expected the complete ordered German Credit schema")
    if frame[TARGET].isna().any() or not frame[TARGET].isin(TARGET_LABELS).all():
        raise ValueError("Target must contain only non-missing codes 1 and 2")
    return frame.drop(columns=TARGET).copy(), frame[TARGET].copy()
