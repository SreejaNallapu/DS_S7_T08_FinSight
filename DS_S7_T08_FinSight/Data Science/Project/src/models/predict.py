"""Standard FinSight prediction response built from the saved inference artifact."""

from __future__ import annotations

import pandas as pd

from src.models.explain import explain_applicant
from src.models.inference import bad_risk_probabilities
from src.models.risk_score import probability_to_risk_score, risk_band, suggested_action


def predict_applicant(applicant: pd.DataFrame, artifact: dict, *, top_n: int = 5) -> dict:
    """Predict and explain exactly one applicant without fitting any model state."""
    if len(applicant) != 1:
        raise ValueError("Prediction response requires exactly one applicant record")
    probability = float(bad_risk_probabilities(applicant, artifact)[0])
    threshold = float(artifact["threshold"])
    predicted_bad = int(probability >= threshold)
    score = probability_to_risk_score(probability)
    band = risk_band(score)
    explanation = explain_applicant(applicant, artifact, top_n=top_n)
    return {
        "predicted_class": "bad" if predicted_bad else "good",
        "predicted_bad_risk": predicted_bad,
        "bad_risk_probability": probability,
        "decision_threshold": threshold,
        "risk_score": score,
        "risk_band": band,
        "suggested_action": suggested_action(band),
        "top_risk_factors": explanation["top_risk_factors"],
        "top_protective_factors": explanation["top_protective_factors"],
        "explanation_direction": explanation["direction"],
    }
