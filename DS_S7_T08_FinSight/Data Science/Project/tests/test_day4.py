import json
from pathlib import Path

import numpy as np
import pytest

from src.data.load import load_german_credit
from src.models.explain import explain_applicant
from src.models.inference import bad_risk_probabilities, load_model_artifact
from src.models.predict import predict_applicant
from src.models.risk_score import probability_to_risk_score, risk_band, suggested_action

MODEL_PATH = Path("models/finsight_pipeline.joblib")
METADATA_PATH = Path("models/model_metadata.json")
RAW_PATH = Path("data/raw/german.data")


@pytest.fixture(scope="module")
def artifact():
    return load_model_artifact(MODEL_PATH)


@pytest.fixture(scope="module")
def applicant(artifact):
    frame = load_german_credit(RAW_PATH)
    return frame.loc[:, artifact["feature_columns"]].iloc[[0]]


def test_bad_risk_probability_uses_fitted_class_one(artifact, applicant):
    model = artifact["pipeline"].named_steps["model"]
    classes = np.asarray(model.classes_)
    assert classes.tolist() == [0, 1]
    bad_index = int(np.flatnonzero(classes == 1)[0])
    expected = artifact["pipeline"].predict_proba(applicant)[:, bad_index]
    actual = bad_risk_probabilities(applicant, artifact)
    assert np.allclose(actual, expected)
    assert np.logical_and(actual >= 0, actual <= 1).all()


@pytest.mark.parametrize(
    "score, expected",
    [(0.0, "LOW"), (29.999, "LOW"), (30.0, "MODERATE"),
     (59.999, "MODERATE"), (60.0, "HIGH"), (100.0, "HIGH")],
)
def test_risk_band_boundaries(score, expected):
    assert risk_band(score) == expected


def test_risk_score_formula_and_validation():
    probability = 0.314159
    score = probability_to_risk_score(probability)
    assert score == pytest.approx(probability * 100)
    assert 0 <= score <= 100
    with pytest.raises(ValueError):
        probability_to_risk_score(1.01)
    with pytest.raises(ValueError):
        risk_band(-0.01)


def test_structured_prediction_response_and_saved_threshold(artifact, applicant):
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    result = predict_applicant(applicant, artifact, top_n=3)
    assert result["decision_threshold"] == artifact["threshold"] == metadata["selected_threshold"]
    assert result["predicted_class"] in {"good", "bad"}
    assert result["predicted_bad_risk"] in {0, 1}
    assert 0 <= result["bad_risk_probability"] <= 1
    assert 0 <= result["risk_score"] <= 100
    assert result["risk_score"] == pytest.approx(result["bad_risk_probability"] * 100)
    assert result["risk_band"] in {"LOW", "MODERATE", "HIGH"}
    assert result["suggested_action"] == suggested_action(result["risk_band"])
    assert len(result["top_risk_factors"]) <= 3
    assert len(result["top_protective_factors"]) <= 3


def test_shap_direction_and_human_readable_factors(artifact, applicant):
    explanation = explain_applicant(applicant, artifact, top_n=5)
    assert explanation["direction_verified"] is True
    assert "bad risk" in explanation["direction"]
    factors = explanation["top_risk_factors"] + explanation["top_protective_factors"]
    assert factors
    assert all("onehot__" not in factor["feature"] for factor in factors)
    assert all("x3_" not in factor["feature"] for factor in factors)
    assert all(factor["shap_value"] > 0 for factor in explanation["top_risk_factors"])
    assert all(factor["shap_value"] < 0 for factor in explanation["top_protective_factors"])


def test_prediction_does_not_fit_model(monkeypatch, artifact, applicant):
    def fail_fit(*args, **kwargs):
        raise AssertionError("Inference must not fit or retrain model state")

    monkeypatch.setattr(artifact["pipeline"], "fit", fail_fit)
    monkeypatch.setattr(artifact["pipeline"].named_steps["preprocessor"], "fit", fail_fit)
    monkeypatch.setattr(artifact["pipeline"].named_steps["model"], "fit", fail_fit)
    result = predict_applicant(applicant, artifact, top_n=2)
    assert 0 <= result["bad_risk_probability"] <= 1
