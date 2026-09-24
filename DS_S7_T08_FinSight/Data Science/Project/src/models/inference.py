"""Stable loading and inference helpers for persisted FinSight artifacts."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.features.preprocessing import NON_PREDICTIVE_COLUMNS


def load_model_artifact(path: str | Path) -> dict:
    artifact = joblib.load(path)
    required = {"pipeline", "threshold", "model_name", "feature_columns"}
    missing = required - set(artifact) if isinstance(artifact, dict) else required
    if missing:
        raise ValueError(f"Invalid FinSight model artifact; missing: {sorted(missing)}")
    leaked = NON_PREDICTIVE_COLUMNS.intersection(artifact["feature_columns"])
    if leaked:
        raise ValueError(f"Unsafe FinSight model artifact; leaked predictors: {sorted(leaked)}")
    classes = artifact["pipeline"].named_steps["model"].classes_
    if 1 not in classes:
        raise ValueError("Invalid FinSight model artifact; bad-risk class 1 is missing")
    return artifact


def predict_credit_risk(applicants: pd.DataFrame, artifact: dict) -> pd.DataFrame:
    missing = set(artifact["feature_columns"]) - set(applicants.columns)
    if missing:
        raise ValueError(f"Missing required applicant fields: {sorted(missing)}")
    probabilities = bad_risk_probabilities(applicants, artifact)
    predictions = (probabilities >= artifact["threshold"]).astype(int)
    return pd.DataFrame(
        {
            "bad_risk_probability": probabilities,
            "predicted_bad_risk": predictions,
            "decision_label": ["bad" if value else "good" for value in predictions],
        },
        index=applicants.index,
    )


def bad_risk_probabilities(applicants: pd.DataFrame, artifact: dict) -> np.ndarray:
    """Return predict_proba values for the fitted training class 1 (bad risk)."""
    missing = set(artifact["feature_columns"]) - set(applicants.columns)
    if missing:
        raise ValueError(f"Missing required applicant fields: {sorted(missing)}")
    ordered = applicants.loc[:, artifact["feature_columns"]]
    model = artifact["pipeline"].named_steps["model"]
    matches = np.flatnonzero(model.classes_ == 1)
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one bad-risk class label 1, found {model.classes_.tolist()}")
    bad_risk_index = int(matches[0])
    return np.asarray(artifact["pipeline"].predict_proba(ordered)[:, bad_risk_index])
