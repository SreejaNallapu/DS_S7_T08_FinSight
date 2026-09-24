"""Thin Flask interface for FinSight analytics and saved-model inference."""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

import pandas as pd
import plotly.express as px
from flask import (
    Flask,
    flash,
    render_template,
    request,
    send_from_directory,
    session,
)

from src.data.schema import CATEGORY_LABELS, CATEGORICAL_COLUMNS, NUMERIC_COLUMNS
from src.models.inference import load_model_artifact
from src.models.predict import predict_applicant

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models/finsight_pipeline.joblib"
METADATA_PATH = PROJECT_ROOT / "models/model_metadata.json"
METRICS_PATH = PROJECT_ROOT / "outputs/metrics/model_metrics.json"
ANALYTICS_PATH = PROJECT_ROOT / "data/processed/german_credit_analytics.csv"
DAY4_PATH = PROJECT_ROOT / "outputs/predictions/example_predictions.json"
PLOTS_DIR = PROJECT_ROOT / "outputs/plots"

FORBIDDEN_INPUTS = {"applicant_id", "credit_risk", "credit_risk_label", "is_bad_risk"}
NUMERIC_RULES = {
    "duration_months": (1, 120),
    "credit_amount": (1, 1_000_000),
    "installment_rate": (1, 4),
    "residence_duration": (1, 4),
    "age_years": (18, 100),
    "existing_credits": (1, 20),
    "dependents": (1, 20),
}
FIELD_LABELS = {
    "checking_account_status": "Checking account status",
    "duration_months": "Loan duration (months)",
    "credit_history": "Credit history",
    "purpose": "Loan purpose",
    "credit_amount": "Credit amount (historical DM)",
    "savings_status": "Savings status",
    "employment_duration": "Employment duration",
    "installment_rate": "Installment rate category (1–4)",
    "personal_status_sex": "Personal status and sex",
    "other_debtors": "Other debtors / guarantors",
    "residence_duration": "Residence duration category (1–4)",
    "property": "Property category",
    "age_years": "Age (years)",
    "other_installment_plans": "Other installment plans",
    "housing": "Housing",
    "existing_credits": "Existing credits",
    "job": "Job category",
    "dependents": "Dependents",
    "telephone": "Registered telephone",
    "foreign_worker": "Foreign-worker status",
}
FIELD_HELPERS = {
    "credit_amount": "DM = Deutsche Mark, the currency used in the original German Credit dataset.",
    "installment_rate": "Category used by the original German Credit dataset.",
    "residence_duration": "Category used by the original German Credit dataset.",
}
PLOT_ALLOWLIST = {
    "shap_summary.png",
    "model_comparison.png",
    "model_roc_curves.png",
    "final_model_confusion_matrix.png",
}


@lru_cache(maxsize=1)
def get_artifact() -> dict:
    """Load the persisted pipeline once per process; never fit it."""
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model artifact is unavailable: {MODEL_PATH.name}")
    return load_model_artifact(MODEL_PATH)


@lru_cache(maxsize=1)
def get_metadata() -> dict:
    if not METADATA_PATH.is_file():
        raise FileNotFoundError(f"Model metadata is unavailable: {METADATA_PATH.name}")
    return json.loads(METADATA_PATH.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def get_metrics() -> dict:
    if not METRICS_PATH.is_file():
        raise FileNotFoundError(f"Model metrics are unavailable: {METRICS_PATH.name}")
    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def get_analytics() -> pd.DataFrame:
    if not ANALYTICS_PATH.is_file():
        raise FileNotFoundError(f"Processed analytics data is unavailable: {ANALYTICS_PATH.name}")
    frame = pd.read_csv(ANALYTICS_PATH)
    required = {"credit_risk_label", "is_bad_risk", "credit_amount", "duration_months"}
    if not required.issubset(frame.columns):
        raise ValueError("Processed analytics data does not match the expected schema")
    return frame


def _friendly_category_label(meaning: str) -> str:
    replacements = {
        "< 0 DM": "Below 0 DM",
        "0 to < 200 DM": "0 to below 200 DM",
        ">= 200 DM or salary assignment for at least 1 year": "200 DM or more, or salary assignment for at least 1 year",
        "< 100 DM": "Below 100 DM",
        "100 to < 500 DM": "100 to below 500 DM",
        "500 to < 1000 DM": "500 to below 1000 DM",
        ">= 1000 DM": "1000 DM or more",
    }
    label = replacements.get(meaning, meaning)
    return label[:1].upper() + label[1:]


def form_fields() -> list[dict]:
    artifact = get_artifact()
    fields = []
    for name in artifact["feature_columns"]:
        field = {
            "name": name,
            "label": FIELD_LABELS[name],
            "helper": FIELD_HELPERS.get(name),
            "categorical": name in CATEGORICAL_COLUMNS,
        }
        if field["categorical"]:
            field["options"] = [
                {"value": code, "label": _friendly_category_label(meaning)}
                for code, meaning in CATEGORY_LABELS[name].items()
            ]
        else:
            field["minimum"], field["maximum"] = NUMERIC_RULES[name]
        fields.append(field)
    return fields


def default_values() -> dict[str, str]:
    try:
        artifact = get_artifact()
        row = get_analytics().loc[0, artifact["feature_columns"]]
        return {name: str(row[name]) for name in artifact["feature_columns"]}
    except (FileNotFoundError, ValueError, KeyError):
        return {}


def parse_applicant(form, prefix: str = "") -> pd.DataFrame:
    artifact = get_artifact()
    forbidden = [name for name in FORBIDDEN_INPUTS if f"{prefix}{name}" in form]
    if forbidden:
        raise ValueError("Identifier and target-derived fields are not accepted")
    values = {}
    errors = []
    for name in artifact["feature_columns"]:
        key = f"{prefix}{name}"
        raw = form.get(key, "").strip()
        if not raw:
            errors.append(f"{FIELD_LABELS[name]} is required")
            continue
        if name in CATEGORICAL_COLUMNS:
            if raw not in CATEGORY_LABELS[name]:
                errors.append(f"{FIELD_LABELS[name]} contains an invalid category")
            else:
                values[name] = raw
        else:
            try:
                numeric = int(raw)
            except ValueError:
                errors.append(f"{FIELD_LABELS[name]} must be a whole number")
                continue
            minimum, maximum = NUMERIC_RULES[name]
            if not minimum <= numeric <= maximum:
                errors.append(f"{FIELD_LABELS[name]} must be between {minimum} and {maximum}")
            else:
                values[name] = numeric
    if errors:
        raise ValueError("; ".join(errors))
    return pd.DataFrame([values], columns=artifact["feature_columns"])


def _plot_html(figure) -> str:
    figure.update_layout(template="plotly_white", margin=dict(l=35, r=20, t=55, b=35))
    return figure.to_html(full_html=False, include_plotlyjs=False, config={"displayModeBar": False})


def _label_categories(frame: pd.DataFrame, column: str) -> pd.Series:
    return frame[column].map(CATEGORY_LABELS[column]).fillna(frame[column])


def create_app(test_config: dict | None = None) -> Flask:
    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.config.from_mapping(
        SECRET_KEY=os.getenv("FINSIGHT_SECRET_KEY", "finsight-local-development"),
        MAX_CONTENT_LENGTH=64 * 1024,
    )
    if test_config:
        app.config.update(test_config)

    @app.get("/")
    def index():
        error = None
        summary = {}
        charts = []
        try:
            data = get_analytics()
            metadata = get_metadata()
            summary = {
                "applicants": len(data),
                "good": int((data["credit_risk_label"] == "good").sum()),
                "bad": int((data["credit_risk_label"] == "bad").sum()),
                "model": metadata["final_model_name"].replace("_", " ").title(),
                "roc_auc": metadata["holdout_metrics"]["roc_auc"],
                "recall": metadata["holdout_metrics"]["recall"],
            }
            counts = data["credit_risk_label"].value_counts().rename_axis("risk").reset_index(name="applicants")
            charts.append(_plot_html(px.pie(counts, names="risk", values="applicants", hole=0.55, title="Portfolio Risk Distribution", color="risk", color_discrete_map={"good": "#1f9d78", "bad": "#c84b5b"})))
            charts.append(_plot_html(px.box(data, x="credit_risk_label", y="credit_amount", color="credit_risk_label", title="Credit Amount by Observed Risk", labels={"credit_risk_label": "Observed risk", "credit_amount": "Credit amount (historical DM)"}, color_discrete_map={"good": "#1f9d78", "bad": "#c84b5b"})))
        except (FileNotFoundError, ValueError, KeyError) as exc:
            error = str(exc)
        return render_template("index.html", summary=summary, charts=charts, error=error)

    @app.get("/portfolio")
    def portfolio():
        error = None
        charts = []
        try:
            data = get_analytics().copy()
            counts = data["credit_risk_label"].value_counts().rename_axis("risk").reset_index(name="applicants")
            charts.append(_plot_html(px.bar(counts, x="risk", y="applicants", color="risk", title="Good and Bad Risk Counts", color_discrete_map={"good": "#1f9d78", "bad": "#c84b5b"})))
            charts.append(_plot_html(px.histogram(data, x="credit_amount", color="credit_risk_label", nbins=35, barmode="overlay", opacity=0.65, title="Credit Amount Distribution by Risk", labels={"credit_amount": "Credit amount (historical DM)", "credit_risk_label": "Observed risk"})))
            charts.append(_plot_html(px.box(data, x="credit_risk_label", y="duration_months", color="credit_risk_label", title="Loan Duration by Risk", labels={"duration_months": "Duration (months)", "credit_risk_label": "Observed risk"})))
            for column, title in [("savings_status", "Bad-Risk Rate by Savings Status"), ("employment_duration", "Bad-Risk Rate by Employment Duration")]:
                grouped = data.assign(label=_label_categories(data, column)).groupby("label", observed=True)["is_bad_risk"].agg(["mean", "count"]).reset_index()
                grouped["bad_risk_percent"] = grouped["mean"] * 100
                charts.append(_plot_html(px.bar(grouped.sort_values("bad_risk_percent"), x="bad_risk_percent", y="label", orientation="h", hover_data=["count"], title=title, labels={"bad_risk_percent": "Bad-risk rate (%)", "label": ""})))
        except (FileNotFoundError, ValueError, KeyError) as exc:
            error = str(exc)
        return render_template("portfolio.html", charts=charts, error=error)

    @app.route("/predict", methods=["GET", "POST"])
    def predict():
        result = None
        values = default_values()
        error = None
        try:
            fields = form_fields()
        except (FileNotFoundError, ValueError, KeyError) as exc:
            fields, error = [], str(exc)
        if request.method == "POST" and not error:
            values.update(request.form.to_dict())
            try:
                applicant = parse_applicant(request.form)
                result = predict_applicant(applicant, get_artifact())
                session["last_prediction"] = result
                flash("Risk assessment completed using the saved FinSight pipeline.", "success")
            except (ValueError, FileNotFoundError, KeyError) as exc:
                error = str(exc)
        return render_template("predict.html", fields=fields, values=values, result=result, error=error)

    @app.get("/explain")
    def explain():
        global_features = []
        error = None
        if DAY4_PATH.is_file():
            try:
                global_features = json.loads(DAY4_PATH.read_text(encoding="utf-8")).get("global_shap_features", [])[:10]
            except (json.JSONDecodeError, OSError) as exc:
                error = f"Explainability data is unavailable: {exc}"
        elif not (PLOTS_DIR / "shap_summary.png").is_file():
            error = "The generated global SHAP artifact is unavailable."
        return render_template("explain.html", local=session.get("last_prediction"), global_features=global_features, error=error, shap_available=(PLOTS_DIR / "shap_summary.png").is_file())

    @app.get("/performance")
    def performance():
        error = None
        metrics = None
        try:
            metrics = get_metrics()
        except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
            error = str(exc)
        return render_template("performance.html", metrics=metrics, error=error)

    @app.route("/whatif", methods=["GET", "POST"])
    def whatif():
        comparison = None
        error = None
        try:
            fields = form_fields()
            defaults = default_values()
        except (FileNotFoundError, ValueError, KeyError) as exc:
            fields, defaults, error = [], {}, str(exc)
        original_values = {f"original_{key}": value for key, value in defaults.items()}
        updated_values = {f"updated_{key}": value for key, value in defaults.items()}
        if request.method == "POST" and not error:
            original_values.update({key: value for key, value in request.form.items() if key.startswith("original_")})
            updated_values.update({key: value for key, value in request.form.items() if key.startswith("updated_")})
            try:
                original = predict_applicant(parse_applicant(request.form, "original_"), get_artifact(), top_n=3)
                updated = predict_applicant(parse_applicant(request.form, "updated_"), get_artifact(), top_n=3)
                comparison = {
                    "original": original,
                    "updated": updated,
                    "probability_change": updated["bad_risk_probability"] - original["bad_risk_probability"],
                    "score_change": updated["risk_score"] - original["risk_score"],
                }
            except (ValueError, FileNotFoundError, KeyError) as exc:
                error = str(exc)
        return render_template("whatif.html", fields=fields, original_values=original_values, updated_values=updated_values, comparison=comparison, error=error)

    @app.get("/artifacts/plots/<name>")
    def artifact_plot(name: str):
        if name not in PLOT_ALLOWLIST or not (PLOTS_DIR / name).is_file():
            return "Plot not found", 404
        return send_from_directory(PLOTS_DIR, name)

    @app.errorhandler(404)
    def not_found(error):
        return render_template("base.html", standalone_error="The requested page was not found."), 404

    @app.errorhandler(500)
    def server_error(error):
        return render_template("base.html", standalone_error="FinSight could not complete this request. Check the application logs."), 500

    return app


app = create_app()


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
