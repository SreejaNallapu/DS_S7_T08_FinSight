"""Reusable loading functions for FinSight."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.schema import CATEGORICAL_COLUMNS, COLUMN_NAMES, NUMERIC_COLUMNS, TARGET


def load_german_credit(path: str | Path) -> pd.DataFrame:
    """Load the whitespace-delimited symbolic UCI file using its fixed schema."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"German Credit data file not found: {path}")
    frame = pd.read_csv(path, sep=r"\s+", header=None)
    if frame.shape[1] != len(COLUMN_NAMES):
        raise ValueError(
            f"Expected {len(COLUMN_NAMES)} whitespace-delimited columns, "
            f"found {frame.shape[1]} in {path}"
        )
    frame.columns = COLUMN_NAMES
    for column in NUMERIC_COLUMNS + [TARGET]:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    for column in CATEGORICAL_COLUMNS:
        frame[column] = frame[column].astype("category")
    return frame
