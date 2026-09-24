"""Transparent presentation-only risk score, bands, and educational actions."""

from __future__ import annotations

import math

LOW_MAX_EXCLUSIVE = 30.0
MODERATE_MAX_EXCLUSIVE = 60.0

RISK_BAND_THRESHOLDS = {
    "LOW": {"minimum": 0.0, "maximum_exclusive": LOW_MAX_EXCLUSIVE},
    "MODERATE": {
        "minimum": LOW_MAX_EXCLUSIVE,
        "maximum_exclusive": MODERATE_MAX_EXCLUSIVE,
    },
    "HIGH": {"minimum": MODERATE_MAX_EXCLUSIVE, "maximum_inclusive": 100.0},
}

SUGGESTED_ACTIONS = {
    "LOW": "Lower Predicted Risk",
    "MODERATE": "Manual Review Suggested",
    "HIGH": "Higher Predicted Risk",
}


def probability_to_risk_score(bad_risk_probability: float) -> float:
    """Return probability × 100 without presentation rounding."""
    probability = float(bad_risk_probability)
    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError("Bad-risk probability must be finite and between 0 and 1")
    return probability * 100.0


def risk_band(risk_score: float) -> str:
    """Assign project-defined display bands; these are not regulatory standards."""
    score = float(risk_score)
    if not math.isfinite(score) or not 0.0 <= score <= 100.0:
        raise ValueError("Risk score must be finite and between 0 and 100")
    if score < LOW_MAX_EXCLUSIVE:
        return "LOW"
    if score < MODERATE_MAX_EXCLUSIVE:
        return "MODERATE"
    return "HIGH"


def suggested_action(band: str) -> str:
    """Return neutral educational wording, never an approval/denial decision."""
    try:
        return SUGGESTED_ACTIONS[band]
    except KeyError as exc:
        raise ValueError(f"Unknown risk band: {band}") from exc
