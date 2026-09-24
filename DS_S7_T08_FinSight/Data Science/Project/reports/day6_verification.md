# Day 6 Final Technical Verification

FinSight was audited as a feature-complete Flask project. No model training, threshold tuning, AWS mutation, UI redesign, or repository reorganization was performed during this stabilization pass.

## Repository structure verified

The mentor-required top-level folders remain in place: `dashboard/`, `data/`, `models/`, `notebooks/`, `outputs/`, `reports/`, `src/`, and `tests/`. Existing Word documents, PowerPoint files, reports, and generated artifacts were preserved. Active application code contains no Streamlit dependency or machine-specific absolute path. `requirements.txt` contains the active runtime, notebook, AWS, and test dependencies.

## Raw data verified

- File: `data/raw/german.data`
- Shape: 1,000 rows and 21 columns: 20 predictors plus the source target
- Source target: `1 = good`, `2 = bad`
- Counts: 700 good and 300 bad
- SHA-256: `b21f3d81db8071257d5ff1deaeba1fd4303b62712e6fcc9715c7a86202cb5871`
- Integrity: the computed checksum matches `data/raw/german.metadata.json`

The raw file was read-only during Day 6 and was not regenerated or modified.

## Processed data verified

- File: `data/processed/german_credit_analytics.csv`
- Shape: 1,000 rows and 24 columns
- Labels: 700 `good`, 300 `bad`
- Binary target: 700 class `0`, 300 class `1`
- Reproducibility: serializing `build_analytics_dataset(load_data())` with the production CSV settings produced text identical to the checked file

The analytics export remains unscaled and unencoded for S3, Glue, and Athena. It is not used as the model feature matrix.

## Target leakage check

The saved artifact has exactly the 20 raw predictors. `applicant_id`, `credit_risk`, `credit_risk_label`, and `is_bad_risk` are absent from its feature contract. The reusable splitter rejects analytics/helper columns, artifact loading rejects leaked feature contracts, and both Assessment and What-If accept only the saved feature list. Web tests verify that target-derived fields are absent from forms and rejected if submitted.

## Model artifact check

- Artifact: `models/finsight_pipeline.joblib`
- Selected model: `gradient_boosting` / `GradientBoostingClassifier`
- Pipeline: fitted `ColumnTransformer` plus fitted classifier
- Fitted classes: `[0, 1]`
- Explicit bad-risk probability class: `1`
- Threshold: `0.13`, consistent between the artifact, metadata, prediction response, and metrics
- Flask and prediction code call only transform/predict/explain operations; automated guards fail if any `fit` method is called during a request
- Model retrained on Day 6: no

## Risk-score check

`risk_score = bad_risk_probability * 100`. Probability validation enforces the inclusive range 0–1, which gives a score in the inclusive range 0–100. Boundary tests confirm LOW for scores below 30, MODERATE for 30 to below 60, and HIGH for 60 through 100. The `0.13` classification threshold remains separate from the 30/60 display bands.

## SHAP check

The global SHAP plot and saved global rankings exist and are loaded by Flask rather than recalculated per page request. Local explanations use the saved preprocessing and model. Additivity is checked against the fitted Gradient Boosting decision margin and the logistic transform is checked against class-1 probability before direction is assigned. Positive values are risk-increasing; negative values are protective. Factor labels are human-readable. Application text consistently describes association/model sensitivity and makes no causal claim.

## Flask route check

The application was started from the repository root with debug mode disabled. Live HTTP checks returned status 200 for `/`, `/portfolio`, `/predict`, `/explain`, `/performance`, and `/whatif`. Each page marked its matching navigation item active. Referenced local CSS, JavaScript, and allow-listed plot routes were also checked through the application.

## Applicant form check

Multiple valid profiles were submitted through the saved pipeline. Results included model classification, bad-risk probability, score, band, classification threshold, suggested action, risk-increasing factors, and protective factors. Readable category labels still submit original UCI values (for example, `Below 0 DM` submits `A11`). Credit values remain on the original historical DM scale and the UI explains Deutsche Mark.

Missing fields, negative credit amounts, negative durations, out-of-range ages, malformed numbers, and invalid categories produce friendly validation messages with HTTP 200. No raw traceback is exposed.

## What-If check

Identical Original and Updated profiles return `+0.00` percentage-point probability change and `+0.00` score change. Valid changed profiles return independently calculated original and updated probabilities/scores from the same saved pipeline. The page retains the model-sensitivity, non-causal disclaimer.

## Portfolio and performance checks

Portfolio charts are built from the real processed CSV and cover risk distribution, historical-DM credit amount, loan duration, savings, and employment. The Performance page reads `outputs/metrics/model_metrics.json`, showing Logistic Regression, Decision Tree, Random Forest, and Gradient Boosting without training on page load.

The selected Gradient Boosting holdout values remain ROC-AUC `0.8005`, bad-risk recall `0.8667`, and cost `122`. Its confusion matrix is ordered with actual classes as rows and predicted classes as columns, both `[good, bad]`:

| | Predicted good | Predicted bad |
|---|---:|---:|
| True good | 58 | 82 |
| True bad | 8 | 52 |

## AWS documentation check

No AWS resource was accessed or modified. Documentation now consistently records region `eu-north-1` and the implemented architecture:

```text
Amazon S3
→ AWS Glue Data Catalog (table registered using Athena DDL)
→ Amazon Athena
→ SQL analytics
```

Processed data is documented at `s3://finsight-credit-risk/processed/german_credit_analytics.csv`, database `finsight`, table `german_credit_analytics`. A Glue crawler was not used due to account permissions and is only an optional alternative.

## Credential and security check

Repository text was scanned case-insensitively for AWS access-key names and common password, secret, and token terms while excluding binary artifacts, the virtual environment, Git internals, and temporary files. No AWS credential, account number, password, or access token was found. The only `secret` matches are Flask's `SECRET_KEY` configuration name and a test-only key value. `.gitignore` covers `.venv/`, `.pytest_cache/`, `.tmp/`, `__pycache__/`, `*.pyc`, `.env`, `.env.*`, `.aws/`, credential files, PEM files, and private keys without excluding source, dashboard, tests, reports, or README.

## Test results

Complete command: `.venv\Scripts\python.exe -m pytest -q`

- Passed: 81
- Failed: 0
- Warnings: 50 non-blocking upstream deprecation warnings
- Duration: 25.92 seconds

## Known non-blocking warnings

The environment emits upstream SHAP/Matplotlib pending-deprecation warnings and Joblib/NumPy deprecation warnings while loading the existing artifact. They do not affect the verified calculations or routes.

## Known issues

No blocking technical issue was found. Bootstrap and Plotly browser assets are loaded from public CDNs, so full visual styling and interactive charts require browser network access; the Flask routes and local project artifacts remain available without live AWS access.
