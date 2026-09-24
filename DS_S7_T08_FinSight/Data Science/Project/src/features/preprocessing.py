"""Leakage-safe feature/target splitting and sklearn preprocessing."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data.schema import CATEGORICAL_COLUMNS, COLUMN_NAMES, NUMERIC_COLUMNS, TARGET

BAD_RISK_TARGET = {1: 0, 2: 1}
NON_PREDICTIVE_COLUMNS = {
    "applicant_id",
    TARGET,
    "credit_risk_label",
    "is_bad_risk",
}


@dataclass(frozen=True)
class DataSplit:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


def split_features_target(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return raw predictors and binary target (1 means bad credit risk)."""
    if TARGET not in frame:
        raise ValueError(f"Required target column is missing: {TARGET}")
    if list(frame.columns) != COLUMN_NAMES:
        raise ValueError("Expected the ordered raw schema; exclude analytics IDs and target-derived fields")
    if frame[TARGET].isna().any():
        raise ValueError("Target contains missing values")
    observed = set(frame[TARGET].dropna().unique())
    if observed != set(BAD_RISK_TARGET):
        raise ValueError(f"Expected target codes {set(BAD_RISK_TARGET)}, found {observed}")
    X = frame.drop(columns=TARGET).copy()
    leaked = NON_PREDICTIVE_COLUMNS.intersection(X.columns)
    if leaked:
        raise ValueError(f"Target-derived/helper columns cannot be predictors: {sorted(leaked)}")
    y = frame[TARGET].map(BAD_RISK_TARGET).astype("int8").rename("is_bad_risk")
    return X, y


def make_train_test_split(
    frame: pd.DataFrame, test_size: float = 0.2, random_state: int = 42
) -> DataSplit:
    """Stratify before fitting any transformer to prevent train/test leakage."""
    X, y = split_features_target(frame)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return DataSplit(X_train, X_test, y_train, y_test)


def build_preprocessor(scale_numeric: bool = True) -> ColumnTransformer:
    """Create an unfitted transformer; fit it on training predictors only.

    Scaling is appropriate for linear, distance-based, and neural models. Pass
    ``scale_numeric=False`` for tree-based estimators, which do not need it.
    """
    numeric_steps: list[tuple[str, object]] = [
        ("imputer", SimpleImputer(strategy="median"))
    ]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    numeric_pipeline = Pipeline(numeric_steps)
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", numeric_pipeline, NUMERIC_COLUMNS),
            ("categorical", categorical_pipeline, CATEGORICAL_COLUMNS),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_model_pipeline(
    estimator: BaseEstimator, scale_numeric: bool = True
) -> Pipeline:
    """Bundle preprocessing and estimator so inference repeats training transforms."""
    return Pipeline(
        [
            ("preprocessor", build_preprocessor(scale_numeric=scale_numeric)),
            ("model", estimator),
        ]
    )
