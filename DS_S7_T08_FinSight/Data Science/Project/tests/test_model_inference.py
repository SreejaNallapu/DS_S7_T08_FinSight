from pathlib import Path

import joblib
import json
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from src.data.load import load_german_credit
from src.data.preprocess import build_analytics_dataset
from src.features.preprocessing import (
    NON_PREDICTIVE_COLUMNS,
    build_model_pipeline,
    make_train_test_split,
    split_features_target,
)
from src.models.inference import load_model_artifact, predict_credit_risk

RAW_PATH = Path("data/raw/german.data")
FINAL_MODEL_PATH = Path("models/finsight_pipeline.joblib")
METADATA_PATH = Path("models/model_metadata.json")
METRICS_PATH = Path("outputs/metrics/model_metrics.json")


def _temporary_artifact(tmp_path):
    frame = load_german_credit(RAW_PATH)
    split = make_train_test_split(frame)
    pipeline = build_model_pipeline(LogisticRegression(max_iter=1000))
    pipeline.fit(split.X_train, split.y_train)
    artifact = {
        "pipeline": pipeline,
        "threshold": 0.4,
        "model_name": "test_logistic",
        "feature_columns": list(split.X_train.columns),
    }
    path = tmp_path / "model.joblib"
    joblib.dump(artifact, path)
    return path, split.X_test


def test_model_round_trip_and_prediction(tmp_path):
    path, applicants = _temporary_artifact(tmp_path)
    artifact = load_model_artifact(path)
    results = predict_credit_risk(applicants.iloc[:4], artifact)
    assert list(results.columns) == [
        "bad_risk_probability", "predicted_bad_risk", "decision_label"
    ]
    assert len(results) == 4
    assert results["bad_risk_probability"].between(0, 1).all()
    assert set(results["predicted_bad_risk"]).issubset({0, 1})
    assert set(results["decision_label"]).issubset({"good", "bad"})


def test_inference_rejects_missing_fields(tmp_path):
    path, applicants = _temporary_artifact(tmp_path)
    artifact = load_model_artifact(path)
    with pytest.raises(ValueError, match="Missing required applicant fields"):
        predict_credit_risk(applicants.drop(columns="purpose").iloc[:1], artifact)


def test_target_and_helper_columns_never_enter_predictors():
    frame = load_german_credit(RAW_PATH)
    X, y = split_features_target(frame)
    assert set(y.unique()) == {0, 1}
    assert y.name == "is_bad_risk"
    assert NON_PREDICTIVE_COLUMNS.isdisjoint(X.columns)
    with pytest.raises(ValueError, match="ordered raw schema"):
        split_features_target(build_analytics_dataset(frame))


def test_saved_final_model_loads_and_is_deterministic():
    assert FINAL_MODEL_PATH.is_file(), "Run: python -m src.models.train"
    assert METADATA_PATH.is_file()
    assert METRICS_PATH.is_file()
    frame = load_german_credit(RAW_PATH)
    applicants = frame.drop(columns="credit_risk").iloc[:3]
    artifact = load_model_artifact(FINAL_MODEL_PATH)
    assert artifact["pipeline"].named_steps["model"].classes_.tolist() == [0, 1]
    assert artifact["probability_class"] == 1
    assert NON_PREDICTIVE_COLUMNS.isdisjoint(artifact["feature_columns"])
    direct_probability = artifact["pipeline"].predict_proba(
        applicants.loc[:, artifact["feature_columns"]]
    )[:, 1]
    first = predict_credit_risk(applicants, artifact)
    second = predict_credit_risk(applicants, artifact)
    assert np.allclose(first["bad_risk_probability"], direct_probability)
    assert np.allclose(first["bad_risk_probability"], second["bad_risk_probability"])
    assert first["bad_risk_probability"].between(0, 1).all()
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    assert metadata["target_mapping"]["training"] == {
        "0": "good credit risk", "1": "bad credit risk"
    }
