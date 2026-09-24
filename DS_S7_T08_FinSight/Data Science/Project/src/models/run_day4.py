"""Generate verified Day 4 predictions, SHAP plot, and explainability report."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.load_data import load_data
from src.data.run_day1 import PROJECT_ROOT
from src.models.explain import create_global_shap_plot
from src.models.inference import bad_risk_probabilities, load_model_artifact
from src.models.predict import predict_applicant
from src.models.risk_score import RISK_BAND_THRESHOLDS, SUGGESTED_ACTIONS

MODEL_PATH = PROJECT_ROOT / "models/finsight_pipeline.joblib"
METADATA_PATH = PROJECT_ROOT / "models/model_metadata.json"
PREDICTIONS_PATH = PROJECT_ROOT / "outputs/predictions/example_predictions.json"
SHAP_PLOT_PATH = PROJECT_ROOT / "outputs/plots/shap_summary.png"
REPORT_PATH = PROJECT_ROOT / "reports/explainability_report.md"


def _plain_record(row: pd.Series) -> dict:
    result = {}
    for key, value in row.items():
        if isinstance(value, np.generic):
            value = value.item()
        result[str(key)] = value
    return result


def _choose_examples(features: pd.DataFrame, probabilities: np.ndarray) -> list[int]:
    targets = {"LOW": 0.05, "MODERATE": 0.45, "HIGH": 0.75}
    ranges = {
        "LOW": (0.0, 0.30),
        "MODERATE": (0.30, 0.60),
        "HIGH": (0.60, 1.01),
    }
    selected = []
    for band in ("LOW", "MODERATE", "HIGH"):
        lower, upper = ranges[band]
        candidates = np.flatnonzero((probabilities >= lower) & (probabilities < upper))
        if not len(candidates):
            raise ValueError(f"No real dataset applicant falls in the configured {band} risk band")
        local = int(np.argmin(np.abs(probabilities[candidates] - targets[band])))
        selected.append(int(candidates[local]))
    return selected


def _factor_lines(factors: list[dict]) -> list[str]:
    if not factors:
        return ["  - No non-zero factor in the displayed direction."]
    return [
        f"  - {factor['feature']} (SHAP {factor['shap_value']:+.4f})"
        for factor in factors
    ]


def write_report(payload: dict, global_features: list[dict], destination: Path) -> None:
    artifact = payload["model"]
    lines = [
        "# Day 4 Explainability and Decision Support", "",
        "This report is generated from the saved Day 3 inference pipeline. No model is fitted or retrained during Day 4 prediction or explanation.", "",
        "## Model and target", "",
        f"- Final model: **{artifact['name']}**",
        f"- Decision threshold: **{artifact['decision_threshold']:.2f}**",
        "- Original target: `1 = good credit risk`, `2 = bad credit risk`.",
        "- Training target: `0 = good credit risk`, `1 = bad credit risk`.",
        "- `bad_risk_probability` is the saved classifier's probability for fitted class label `1`, verified from `classes_`.", "",
        "## Risk score and display bands", "",
        "`risk_score = bad_risk_probability × 100`", "",
        "- LOW: `0 ≤ score < 30` — Lower Predicted Risk",
        "- MODERATE: `30 ≤ score < 60` — Manual Review Suggested",
        "- HIGH: `60 ≤ score ≤ 100` — Higher Predicted Risk", "",
        "The model's **0.13 classification threshold** and the **30/60 display-band boundaries** serve different purposes. The former converts model probability into its cost-sensitive predicted class. The latter organizes a 0–100 display for educational decision support. The bands are project-defined, not banking or regulatory standards.", "",
        "## SHAP interpretation", "",
        "SHAP is calculated after the saved `ColumnTransformer` transforms the original applicant fields. The final Gradient Boosting explainer produces additive contributions in raw bad-risk margin (log-odds) units. For every explanation, additivity is checked against `decision_function`, and the logistic transform is checked against the fitted class-1 probability. Therefore, positive values are labelled risk-increasing only after confirming that they increase bad-risk probability; negative values are protective for that prediction.", "",
        "Categorical one-hot features are converted to readable labels such as `checking account status = < 0 DM (A11)` or `checking account status is not no checking account (A14)`.", "",
        "## Global SHAP importance", "",
        "Mean absolute SHAP values are aggregated from transformed columns back to the 20 original applicant features. They measure influence magnitude, not causal effect.", "",
        "| Rank | Feature | Mean absolute SHAP |", "|---:|---|---:|",
    ]
    for rank, row in enumerate(global_features, start=1):
        lines.append(f"| {rank} | {row['feature']} | {row['mean_absolute_shap']:.6f} |")
    lines += ["", "![Global SHAP summary](../outputs/plots/shap_summary.png)", "", "## Real applicant examples", ""]
    for number, example in enumerate(payload["examples"], start=1):
        prediction = example["prediction"]
        lines += [
            f"### Example {number}: {prediction['risk_band']}", "",
            f"- Source row: {example['source_row_number']}",
            f"- Bad-risk probability: **{prediction['bad_risk_probability']:.6f}**",
            f"- Classification: **{prediction['predicted_class']}** at threshold `{prediction['decision_threshold']:.2f}`",
            f"- Risk score: **{prediction['risk_score']:.2f}**",
            f"- Suggested action: **{prediction['suggested_action']}**", "",
            "Risk-increasing factors:",
        ]
        lines += _factor_lines(prediction["top_risk_factors"])
        lines += ["", "Protective factors:"]
        lines += _factor_lines(prediction["top_protective_factors"])
        lines.append("")
    lines += [
        "## Limitations", "",
        "- The score is model probability scaled to 0–100; it is not an official credit score.",
        "- Risk bands and suggested actions are educational project conventions, not approval or denial rules.",
        "- SHAP explains this fitted model's behavior, not causality, applicant intent, or fairness.",
        "- One-hot SHAP contributions can describe both the presence and absence of categories; correlated features can share or redistribute importance.",
        "- The German Credit dataset is small, historical, and includes sensitive or proxy attributes. Human review and fairness analysis are required before any real-world use.",
        "- FinSight must not be used as an automated lending decision system.", "",
    ]
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    artifact = load_model_artifact(MODEL_PATH)
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    if metadata["selected_threshold"] != artifact["threshold"]:
        raise ValueError("Model artifact and metadata decision thresholds disagree")
    if metadata["final_model_name"] != artifact["model_name"]:
        raise ValueError("Model artifact and metadata model names disagree")

    frame = load_data()
    features = frame.loc[:, artifact["feature_columns"]]
    probabilities = bad_risk_probabilities(features, artifact)
    example_indices = _choose_examples(features, probabilities)
    examples = []
    for index in example_indices:
        applicant = features.iloc[[index]]
        examples.append(
            {
                "source_row_number": index + 1,
                "applicant": _plain_record(applicant.iloc[0]),
                "prediction": predict_applicant(applicant, artifact),
            }
        )

    global_features = create_global_shap_plot(features, artifact, SHAP_PLOT_PATH)
    payload = {
        "model": {
            "name": artifact["model_name"],
            "artifact": "models/finsight_pipeline.joblib",
            "decision_threshold": artifact["threshold"],
            "classifier_classes": artifact["pipeline"].named_steps["model"].classes_.tolist(),
            "bad_risk_class": 1,
            "target_mapping": artifact["target_mapping"],
        },
        "risk_score_formula": "bad_risk_probability * 100",
        "risk_band_thresholds": RISK_BAND_THRESHOLDS,
        "suggested_actions": SUGGESTED_ACTIONS,
        "global_shap_features": global_features,
        "examples": examples,
    }
    PREDICTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    PREDICTIONS_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    write_report(payload, global_features, REPORT_PATH)

    print(f"Model: {artifact['model_name']}; threshold={artifact['threshold']:.2f}")
    for number, example in enumerate(examples, start=1):
        prediction = example["prediction"]
        print(
            f"Example {number}: probability={prediction['bad_risk_probability']:.6f}, "
            f"score={prediction['risk_score']:.2f}, band={prediction['risk_band']}, "
            f"prediction={prediction['predicted_class']}"
        )
    print(f"Predictions: {PREDICTIONS_PATH}")
    print(f"SHAP plot: {SHAP_PLOT_PATH}")
    print(f"Report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
