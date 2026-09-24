from pathlib import Path

import pandas as pd
import pytest
from werkzeug.datastructures import MultiDict

from dashboard.app import create_app, get_artifact, parse_applicant
from src.data.load import load_german_credit


@pytest.fixture()
def app():
    return create_app({"TESTING": True, "SECRET_KEY": "dashboard-test-key"})


@pytest.fixture()
def client(app):
    return app.test_client()


@pytest.fixture(scope="module")
def valid_form():
    artifact = get_artifact()
    frame = load_german_credit(Path("data/raw/german.data"))
    row = frame.loc[0, artifact["feature_columns"]]
    return {name: str(value) for name, value in row.items()}


@pytest.mark.parametrize(
    "route, marker",
    [
        ("/", "Portfolio Risk Distribution"),
        ("/portfolio", "Portfolio Analytics"),
        ("/predict", "Applicant Risk Assessment"),
        ("/explain", "SHAP Explainability"),
        ("/performance", "Model Performance"),
        ("/whatif", "What-If Simulator"),
    ],
)
def test_get_routes(client, route, marker):
    response = client.get(route)
    assert response.status_code == 200
    assert marker in response.get_data(as_text=True)


@pytest.mark.parametrize(
    "route, label",
    [
        ("/", "Overview"),
        ("/portfolio", "Portfolio"),
        ("/predict", "Assessment"),
        ("/explain", "Explainability"),
        ("/performance", "Performance"),
        ("/whatif", "What-If"),
    ],
)
def test_active_navigation_matches_current_page(client, route, label):
    html = client.get(route).get_data(as_text=True)
    assert f'class="nav-link active" aria-current="page"' in html
    active_fragment = html.split('class="nav-link active" aria-current="page"', 1)[1]
    assert f">{label}</a>" in active_fragment.split("</li>", 1)[0]


def test_valid_prediction_renders_saved_model_response(client, valid_form):
    response = client.post("/predict", data=valid_form)
    html = response.get_data(as_text=True)
    assert response.status_code == 200
    assert "Assessment result" in html
    assert "Bad-risk probability" in html
    assert "Risk score" in html
    assert "Decision threshold" in html
    assert "Suggested action" in html
    assert "Factors increasing predicted bad risk" in html
    assert "Protective factors" in html
    assert any(band in html for band in ("LOW", "MODERATE", "HIGH"))
    assert "Loan Approved" not in html
    assert "Loan Rejected" not in html


def test_form_labels_are_readable_but_submitted_values_are_unchanged(client, valid_form):
    html = client.get("/predict").get_data(as_text=True)
    assert "Credit amount (historical DM)" in html
    assert "DM = Deutsche Mark, the currency used in the original German Credit dataset." in html
    assert '>Below 0 DM</option>' in html
    assert 'value="A11"' in html
    assert "(A11)" not in html
    assert html.count("Category used by the original German Credit dataset.") >= 2

    parsed = parse_applicant(MultiDict(valid_form))
    expected = load_german_credit(Path("data/raw/german.data")).loc[0, get_artifact()["feature_columns"]]
    pd.testing.assert_series_equal(parsed.iloc[0], expected, check_names=False)


def test_global_footer_removed_but_context_notes_remain(client):
    overview = client.get("/").get_data(as_text=True)
    predict_page = client.get("/predict").get_data(as_text=True)
    whatif_page = client.get("/whatif").get_data(as_text=True)
    assert "Educational prototype disclaimer" not in overview
    assert "site-footer" not in overview
    assert "classification threshold is separate from display bands" in predict_page
    assert "does not establish causal effects" in whatif_page


def test_invalid_prediction_returns_friendly_error(client, valid_form):
    invalid = dict(valid_form)
    invalid["age_years"] = "-5"
    response = client.post("/predict", data=invalid)
    html = response.get_data(as_text=True)
    assert response.status_code == 200
    assert "Age (years) must be between 18 and 100" in html
    assert "Traceback" not in html


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("credit_amount", "", "Credit amount (historical DM) is required"),
        ("credit_amount", "-1", "Credit amount (historical DM) must be between 1 and 1000000"),
        ("duration_months", "-1", "Loan duration (months) must be between 1 and 120"),
        ("age_years", "101", "Age (years) must be between 18 and 100"),
        ("credit_amount", "not-a-number", "Credit amount (historical DM) must be a whole number"),
        ("checking_account_status", "INVALID", "Checking account status contains an invalid category"),
    ],
)
def test_invalid_inputs_are_rejected_without_tracebacks(client, valid_form, field, value, message):
    invalid = dict(valid_form)
    invalid[field] = value
    response = client.post("/predict", data=invalid)
    html = response.get_data(as_text=True)
    assert response.status_code == 200
    assert message in html
    assert "Traceback" not in html


def test_target_fields_are_not_inputs_and_are_rejected(client, valid_form):
    page = client.get("/predict").get_data(as_text=True)
    for forbidden in ("applicant_id", "credit_risk", "credit_risk_label", "is_bad_risk"):
        assert f'name="{forbidden}"' not in page
    injected = dict(valid_form)
    injected["credit_risk"] = "1"
    response = client.post("/predict", data=injected)
    assert response.status_code == 200
    assert "Identifier and target-derived fields are not accepted" in response.get_data(as_text=True)


def test_explain_page_shows_local_result_after_prediction(client, valid_form):
    assert client.post("/predict", data=valid_form).status_code == 200
    response = client.get("/explain")
    html = response.get_data(as_text=True)
    assert response.status_code == 200
    assert "Latest applicant explanation" in html
    assert "Risk-increasing factors" in html
    assert "Protective factors" in html


def test_performance_uses_actual_four_model_metrics(client):
    html = client.get("/performance").get_data(as_text=True)
    for model in ("Logistic Regression", "Decision Tree", "Random Forest", "Gradient Boosting"):
        assert model in html
    assert "0.8005" in html
    assert "Selected" in html


def test_whatif_comparison(client, valid_form):
    form = {}
    for name, value in valid_form.items():
        form[f"original_{name}"] = value
        form[f"updated_{name}"] = value
    form["updated_credit_amount"] = "5000"
    response = client.post("/whatif", data=form)
    html = response.get_data(as_text=True)
    assert response.status_code == 200
    assert "What-if comparison" in html
    assert "Original" in html and "Updated" in html and "Change" in html
    assert "does not establish causal effects" in html


def test_identical_whatif_profiles_have_zero_change(client, valid_form):
    form = {
        f"{prefix}{name}": value
        for prefix in ("original_", "updated_")
        for name, value in valid_form.items()
    }
    response = client.post("/whatif", data=form)
    html = response.get_data(as_text=True)
    assert response.status_code == 200
    assert "+0.00 pp" in html
    assert "+0.00 score points" in html


def test_whatif_rejects_target_derived_fields(client, valid_form):
    form = {
        f"{prefix}{name}": value
        for prefix in ("original_", "updated_")
        for name, value in valid_form.items()
    }
    form["updated_is_bad_risk"] = "1"
    response = client.post("/whatif", data=form)
    assert response.status_code == 200
    assert "Identifier and target-derived fields are not accepted" in response.get_data(as_text=True)


def test_flask_prediction_never_retrains(monkeypatch, client, valid_form):
    artifact = get_artifact()

    def fail_fit(*args, **kwargs):
        raise AssertionError("Flask inference must never call fit")

    monkeypatch.setattr(artifact["pipeline"], "fit", fail_fit)
    monkeypatch.setattr(artifact["pipeline"].named_steps["preprocessor"], "fit", fail_fit)
    monkeypatch.setattr(artifact["pipeline"].named_steps["model"], "fit", fail_fit)
    response = client.post("/predict", data=valid_form)
    assert response.status_code == 200
    assert "Assessment result" in response.get_data(as_text=True)
