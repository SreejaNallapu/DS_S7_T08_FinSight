"""SHAP explanations for the saved Gradient Boosting inference pipeline."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from src.data.schema import CATEGORY_LABELS, CATEGORICAL_COLUMNS


def _bad_class_index(model) -> int:
    classes = np.asarray(model.classes_)
    matches = np.flatnonzero(classes == 1)
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one bad-risk class label 1, found {classes.tolist()}")
    return int(matches[0])


def _normalise_shap_values(raw_values, bad_index: int, rows: int, features: int) -> np.ndarray:
    if isinstance(raw_values, list):
        values = np.asarray(raw_values[bad_index])
    else:
        values = np.asarray(raw_values)
        if values.ndim == 3:
            if values.shape[:2] == (rows, features):
                values = values[:, :, bad_index]
            elif values.shape[1:] == (rows, features):
                values = values[bad_index]
            else:
                raise ValueError(f"Unsupported SHAP output shape: {values.shape}")
    if values.shape != (rows, features):
        raise ValueError(f"Expected SHAP shape {(rows, features)}, found {values.shape}")
    return values.astype(float, copy=False)


def _expected_value(explainer, bad_index: int) -> float:
    values = np.asarray(explainer.expected_value).reshape(-1)
    if len(values) == 1:
        return float(values[0])
    if bad_index >= len(values):
        raise ValueError("SHAP expected value does not contain the bad-risk class")
    return float(values[bad_index])


def _source_feature(transformed_name: str) -> str:
    for column in CATEGORICAL_COLUMNS:
        if transformed_name.startswith(f"{column}_"):
            return column
    return transformed_name


def _human_feature(transformed_name: str, transformed_value: float, applicant: pd.Series) -> str:
    source = _source_feature(transformed_name)
    if source in CATEGORICAL_COLUMNS:
        code = transformed_name[len(source) + 1 :]
        meaning = CATEGORY_LABELS.get(source, {}).get(code, code)
        operator = "=" if transformed_value >= 0.5 else "is not"
        return f"{source.replace('_', ' ')} {operator} {meaning} ({code})"
    value = applicant[source]
    if isinstance(value, (np.integer, int)):
        value = int(value)
    elif isinstance(value, (np.floating, float)):
        value = round(float(value), 4)
    return f"{source.replace('_', ' ')} = {value}"


def compute_shap_values(applicants: pd.DataFrame, artifact: dict) -> dict:
    """Compute and direction-check bad-risk SHAP values on transformed features."""
    if applicants.empty:
        raise ValueError("At least one applicant is required")
    pipeline = artifact["pipeline"]
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]
    bad_index = _bad_class_index(model)
    ordered = applicants.loc[:, artifact["feature_columns"]]
    transformed = np.asarray(preprocessor.transform(ordered))
    feature_names = np.asarray(preprocessor.get_feature_names_out(), dtype=object)
    explainer = shap.TreeExplainer(model)
    values = _normalise_shap_values(
        explainer.shap_values(transformed), bad_index, len(ordered), transformed.shape[1]
    )
    base_value = _expected_value(explainer, bad_index)

    # Gradient Boosting SHAP values are in raw decision-margin units. Confirm
    # additivity against this exact fitted model before assigning direction.
    raw_from_shap = base_value + values.sum(axis=1)
    raw_from_model = np.asarray(model.decision_function(transformed)).reshape(-1)
    if not np.allclose(raw_from_shap, raw_from_model, rtol=1e-5, atol=1e-6):
        raise ValueError("SHAP values do not add up to the fitted model's bad-risk margin")
    probabilities = np.asarray(model.predict_proba(transformed))[:, bad_index]
    logistic_probability = 1.0 / (1.0 + np.exp(-raw_from_shap))
    if not np.allclose(logistic_probability, probabilities, rtol=1e-5, atol=1e-6):
        raise ValueError("Positive SHAP direction could not be verified for bad-risk probability")

    return {
        "values": values,
        "base_value": base_value,
        "transformed": transformed,
        "feature_names": feature_names,
        "bad_class_index": bad_index,
        "direction": "positive SHAP values increase the raw margin and probability of training class 1 (bad risk)",
    }


def explain_applicant(applicant: pd.DataFrame, artifact: dict, top_n: int = 5) -> dict:
    """Return understandable local factors increasing and decreasing bad risk."""
    if len(applicant) != 1:
        raise ValueError("Applicant-level explanation requires exactly one record")
    result = compute_shap_values(applicant, artifact)
    row_values = result["values"][0]
    transformed = result["transformed"][0]
    raw_applicant = applicant.iloc[0]
    factors = [
        {
            "feature": _human_feature(str(name), float(value), raw_applicant),
            "source_feature": _source_feature(str(name)),
            "shap_value": float(shap_value),
        }
        for name, value, shap_value in zip(result["feature_names"], transformed, row_values)
        if not np.isclose(shap_value, 0.0)
    ]
    increasing = sorted(
        (factor for factor in factors if factor["shap_value"] > 0),
        key=lambda factor: factor["shap_value"],
        reverse=True,
    )[:top_n]
    protective = sorted(
        (factor for factor in factors if factor["shap_value"] < 0),
        key=lambda factor: factor["shap_value"],
    )[:top_n]
    return {
        "explanation_space": "raw model margin (log-odds for bad risk)",
        "direction_verified": True,
        "direction": result["direction"],
        "base_value": result["base_value"],
        "top_risk_factors": increasing,
        "top_protective_factors": protective,
    }


def create_global_shap_plot(
    applicants: pd.DataFrame, artifact: dict, destination: str | Path, top_n: int = 15
) -> list[dict]:
    """Create a global mean-|SHAP| plot aggregated to original features."""
    result = compute_shap_values(applicants, artifact)
    totals: dict[str, float] = {}
    mean_absolute = np.abs(result["values"]).mean(axis=0)
    for name, importance in zip(result["feature_names"], mean_absolute):
        source = _source_feature(str(name))
        totals[source] = totals.get(source, 0.0) + float(importance)
    ranking = sorted(totals.items(), key=lambda item: item[1], reverse=True)
    rows = [
        {"feature": name, "mean_absolute_shap": value}
        for name, value in ranking[:top_n]
    ]
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    labels = [row["feature"].replace("_", " ") for row in reversed(rows)]
    values = [row["mean_absolute_shap"] for row in reversed(rows)]
    plt.figure(figsize=(9, 7))
    plt.barh(labels, values, color="#7b61a8")
    plt.xlabel("Mean absolute SHAP value (bad-risk margin)")
    plt.title("Global SHAP Feature Importance")
    plt.tight_layout()
    plt.savefig(destination, dpi=160, bbox_inches="tight")
    plt.close()
    return rows
