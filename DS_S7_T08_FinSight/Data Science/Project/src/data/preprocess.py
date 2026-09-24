"""Day 2 preprocessing and a separate, unscaled analytics export. No model fitting."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.data.load_data import DEFAULT_RAW_PATH, load_data
from src.data.schema import TARGET, TARGET_LABELS
from src.data.validate import validate_dataset
from src.features.preprocessing import (
    DataSplit, build_preprocessor, make_train_test_split, split_features_target,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ANALYTICS_PATH = PROJECT_ROOT / "data/processed/german_credit_analytics.csv"


@dataclass
class PreparedData:
    split: DataSplit
    pipeline: Pipeline
    X_train: np.ndarray
    X_test: np.ndarray


def prepare_data(frame: pd.DataFrame, *, scale_numeric: bool = False,
                 test_size: float = 0.2, random_state: int = 42) -> PreparedData:
    """Fit only preprocessing on training rows; use pipeline.transform at inference.

    Scaling is opt-in for later linear/distance models. Tree models need no scaling.
    The target and all analytics-derived columns are excluded by the raw schema.
    """
    split = make_train_test_split(frame, test_size=test_size, random_state=random_state)
    pipeline = Pipeline([("preprocessor", build_preprocessor(scale_numeric=scale_numeric))])
    train = pipeline.fit_transform(split.X_train)
    test = pipeline.transform(split.X_test)
    return PreparedData(split, pipeline, train, test)


def build_analytics_dataset(frame: pd.DataFrame) -> pd.DataFrame:
    """Preserve all validated raw rows and codes; add explicit analytics fields.

    applicant_id is the 1-based source row number, not a real person identifier.
    This table is for descriptive SQL, never an already-transformed ML matrix.
    """
    validate_dataset(frame).raise_for_errors()
    _, target = split_features_target(frame)
    analytics = frame.copy()
    analytics.insert(0, "applicant_id", np.arange(1, len(frame) + 1))
    analytics["credit_risk_label"] = frame[TARGET].map(TARGET_LABELS)
    analytics["is_bad_risk"] = target
    return analytics


def write_analytics_dataset(frame: pd.DataFrame, destination: str | Path = DEFAULT_ANALYTICS_PATH) -> Path:
    """Write UTF-8 CSV with one header and no dataframe index, inside data/processed."""
    destination = Path(destination).resolve()
    processed_root = (PROJECT_ROOT / "data/processed").resolve()
    if not destination.is_relative_to(processed_root) or destination.suffix.lower() != ".csv":
        raise ValueError("Analytics CSV must be written under data/processed/; raw files are protected")
    analytics = build_analytics_dataset(frame)
    destination.parent.mkdir(parents=True, exist_ok=True)
    analytics.to_csv(destination, index=False, encoding="utf-8", lineterminator="\n")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-path", type=Path, default=DEFAULT_RAW_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_ANALYTICS_PATH)
    parser.add_argument("--scale-numeric", action="store_true")
    args = parser.parse_args()
    frame = load_data(args.raw_path)
    output = write_analytics_dataset(frame, args.output)
    prepared = prepare_data(frame, scale_numeric=args.scale_numeric)
    print(f"Analytics CSV: {output} ({len(frame)} rows, 24 columns)")
    print(f"Preprocessing only: train={prepared.X_train.shape}, test={prepared.X_test.shape}; random_state=42")
    print("No ML model trained. Reuse prepared.pipeline.transform(raw_predictors) at inference.")


if __name__ == "__main__":
    main()
