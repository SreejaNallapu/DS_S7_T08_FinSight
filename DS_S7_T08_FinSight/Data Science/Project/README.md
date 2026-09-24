# FinSight — Cloud-Enabled Explainable Credit Risk Analytics & Decision Support System

**Explainable Credit Risk Analytics & Loan Decision Support System**

FinSight is a student-friendly, production-style Python project for classifying applicants in the UCI Statlog German Credit dataset as good or bad credit risks. The current implementation provides a reproducible data foundation, exploratory analysis, leakage-safe preprocessing, a trained cost-sensitive inference pipeline, SHAP-based educational decision support, and a complete Flask dashboard.


## Day 1 foundation

Objective: establish a reproducible, validated German Credit data foundation for explainable credit-risk analytics and later cloud-backed decision support.

The repository already contains EDA, preprocessing, training code, saved models, and model reports from earlier work. These are preserved historical work; no models are trained by the Day 1 command. The milestone descriptions and model results below describe that earlier work, not a new Day 1 execution.

Final stack: Python, Pandas, NumPy, Scikit-learn, SHAP, Joblib, Flask, HTML/CSS, Bootstrap, Plotly, Amazon S3, AWS Glue, Amazon Athena, Boto3, and Pytest. The Flask application lives inside `dashboard/`. The Day 2 S3, Glue Data Catalog, and Athena analytics path is deployed and verified. Existing Matplotlib and notebook dependencies support the preserved EDA implementation.

Run from the existing repository in PowerShell (create `.venv` only if absent):

```powershell
# Run these commands from the FinSight repository root.
# Only if .venv is absent: python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m src.data.run_day1
.venv\Scripts\python.exe -m pytest tests/test_data.py tests/test_day1.py -q
```

The command uses `src/data/load_data.py`, reuses the verified raw cache, downloads the official symbolic file if absent, validates it, and generates `reports/data_dictionary.md`, `reports/data_quality_report.md`, and `outputs/data_quality_report.json`. Existing historical JSON and figures in `reports/` remain intact. `--force-download` explicitly refreshes raw data; `--raw-path`, `--reports-dir`, and `--outputs-dir` override destinations.

```python
from src.data.load_data import load_data, extract_features_target
frame = load_data()
X, y = extract_features_target(frame)  # 20 predictors; raw 1=good, 2=bad target
```

Validation checks 1,000 rows, ordered columns, completeness, documented codes, and finite positive integer numeric features. Duplicate rows are reported as warnings, never removed. Numeric positivity/integrality is a basic sanity rule, not a claim that all possible domain anomalies are detectable. Unknown/no-account categorical codes are legitimate values, not missing values.

### Manual AWS setup (optional for Day 1)

1. Install AWS CLI separately and configure an AWS profile outside this repository using `aws configure sso`, then `aws sso login --profile finsight` (use the profile name you configured). Alternatively use your organization's environment or role credentials.
2. Create a private S3 bucket in your chosen region; retain Block Public Access and default encryption. Choose a prefix such as `finsight/raw/`.
3. Grant the profile `s3:PutObject` and `s3:GetObject` on that prefix, and `s3:ListBucket` on the bucket so missing-object checks can return 404. Customer-managed KMS encryption may additionally require KMS permissions.
4. Set the profile and region, then run the example below after replacing the bucket placeholder. No AWS resources are created by the Day 1 data command. Glue/Athena setup was deferred during Day 1 and was subsequently completed in Day 2 using Athena DDL registration.

```powershell
$env:AWS_PROFILE = 'finsight'
$env:AWS_DEFAULT_REGION = 'eu-north-1'
.venv\Scripts\python.exe -c "from src.cloud.s3_utils import upload_file, object_exists, download_file; b='YOUR-BUCKET'; k='finsight/raw/german.data'; upload_file('data/raw/german.data', b, k); print(object_exists(b, k)); download_file(b, k, 'data/raw/german.s3-copy.data')"
```

S3 helpers use [Boto3 credential resolution](https://docs.aws.amazon.com/boto3/latest/guide/credentials.html), accept an optional client for tests, and propagate service errors. A 403 is not treated as a missing object; see [AWS HeadObject documentation](https://docs.aws.amazon.com/boto3/latest/reference/services/s3/client/head_object.html). Local S3 tests use mocks and do not verify live AWS access. Never put credentials in source files or commit credential files.

## Dataset and target

The original symbolic UCI Statlog German Credit file contains 1,000 historical applicants, 20 predictors, and one outcome. Predictors cover checking and savings status, credit history, requested amount and duration, employment, installment burden, housing, property, age, and related application information.

The source target is `credit_risk`: `1 = good risk`, `2 = bad risk`. Preprocessing maps this explicitly to `is_bad_risk`, where `0 = good` and `1 = bad`, so later recall and cost-sensitive evaluation focus on the risky class. The observed class distribution is 700 good and 300 bad applicants.

## Day 1 status

Implemented:

- Reproducible download and local cache of UCI's original symbolic `german.data`
- SHA-256, retrieval timestamp, and source URL provenance metadata
- Pinned official-file checksum verification for cached and fresh downloads
- Explicit 21-column schema and pandas dtypes
- Validation of dimensions, missingness, target codes, numeric fields, and documented categorical codes
- Duplicate-row warning
- Computed Markdown and JSON data-quality reports
- Complete data dictionary with official categorical meanings and target encoding
- Unit/integration tests

Day 1 intentionally did not split or transform the data. Those steps are now implemented in the EDA/preprocessing milestone. Model fitting or tuning, SHAP explanations, and dashboard pages remain later work.

## EDA and preprocessing status

Implemented:

- Reproducible numeric and categorical distributions
- Credit-risk relationships for important account, history, savings, purpose, duration, amount, and age variables
- Completeness, duplicate, invalid-value, class-balance, summary-statistic, and IQR outlier metrics
- Five report-ready PNG figures plus Markdown and JSON EDA reports
- Reproducible 80/20 train/test split with target stratification (`random_state=42`)
- `ColumnTransformer` with median numeric imputation and one-hot categorical encoding
- Optional numeric standardization for scale-sensitive models; tree models can disable scaling
- Unknown-category-safe inference and a single sklearn model pipeline that repeats training transformations

No unusual observation is removed automatically. Potential outliers are descriptive IQR flags and may be legitimate credit values.

## Quick start (PowerShell)

From the project root:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m src.data.run_day1
python -m src.data.run_eda
python -m jupyter nbconvert --to notebook --execute notebooks/01_eda.ipynb --inplace
python -m pytest -q
```

Python 3.10–3.12 is recommended. If PowerShell blocks virtual-environment activation, you can run the interpreter directly instead:

```powershell
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m src.data.run_day1
.venv\Scripts\python.exe -m src.data.run_eda
.venv\Scripts\python.exe -m jupyter nbconvert --to notebook --execute notebooks/01_eda.ipynb --inplace
.venv\Scripts\python.exe -m pytest -q
```

The first pipeline run needs internet access. It caches `data/raw/german.data`; later runs reuse it. To intentionally refresh the raw cache:

```powershell
python -m src.data.run_day1 --force-download
```

Review the generated outputs:

- `reports/data_quality_report.md` — readable inspection results
- `outputs/data_quality_report.json` — detailed machine-readable profile
- `reports/data_dictionary.md` — schema, categorical code meanings, and target definition
- `data/raw/german.metadata.json` — download provenance and checksum
- `reports/eda_report.md` — readable EDA findings and outlier flags
- `reports/eda_summary.json` — detailed EDA metrics and category-level bad-risk rates
- `reports/figures/` — report-ready visualizations
- `notebooks/01_eda.ipynb` — narrative analysis backed by reusable `src/` code
- `data/processed/german_credit_analytics.csv` — unscaled, human-readable analytics table for S3/Glue/Athena

The generated quality report includes dimensions, dtypes, missingness, duplicates, unique counts, observed categorical codes, numeric ranges, and the target distribution. Re-running the command overwrites only the generated reports with freshly computed values.

## Model experimentation and evaluation

Day 3 trains exactly four models: Logistic Regression, Decision Tree, Random Forest, and Gradient Boosting. It uses a reproducible stratified 80/20 split (`random_state=42`), five-fold stratified cross-validation on the 800-row training partition, lightweight eight-candidate randomized searches for the strongest candidate families, and one untouched 200-row holdout set. Every estimator remains inside the preprocessing pipeline so imputation, scaling, and encoding are fitted only on training folds.

The original target is `credit_risk`, where `1 = good` and `2 = bad`. Training maps this to `0 = good` and `1 = bad`. Bad risk is explicitly the positive class for precision, recall, F1, ROC-AUC, and thresholding. `bad_risk_probability` is the fitted classifier's `predict_proba` output for class label `1`; inference verifies the classifier classes rather than assuming an unverified probability column.

The 20 original applicant features are used. `applicant_id`, `credit_risk`, `credit_risk_label`, and `is_bad_risk` are explicitly excluded to prevent identifier or target leakage. The model trains from the validated raw dataset, not the 24-column analytics export.

The German Credit cost convention assigns a cost of 5 to predicting a genuinely bad applicant as good and 1 to predicting a good applicant as bad. FinSight evaluates `5 × false negatives + false positives`. Each threshold is selected using out-of-fold training probabilities, and the final model is selected using training cost, mean CV ROC-AUC, bad-risk recall, F1, CV stability, interpretability, and deployment practicality—not accuracy or holdout performance alone.

Run training:

```powershell
.venv\Scripts\python.exe -m src.models.train
```

Run the complete tests:

```powershell
.venv\Scripts\python.exe -m pytest -q
```

Generated artifacts:

- `models/finsight_pipeline.joblib` — fitted preprocessing and selected classifier, threshold, mapping, and feature contract
- `models/model_metadata.json` — training timestamp, dataset checksum, features, target mapping, split, and selected-model metrics
- `models/final_model.joblib` — refreshed compatibility copy of the same current artifact
- `outputs/metrics/model_metrics.json` — all fold values, holdout metrics, confusion matrices, costs, and tuning results
- `reports/model_evaluation.md` — readable comparison and selection rationale
- `outputs/plots/model_comparison.png`
- `outputs/plots/model_roc_curves.png`
- `outputs/plots/final_model_confusion_matrix.png`
- `outputs/plots/final_model_feature_importance.png`

Latest verified Day 3 run (`random_state=42`, training timestamp `2026-09-17T14:34:49.124090+00:00`):

| Model | Threshold | Accuracy | Precision | Bad-risk recall | F1 | ROC-AUC | CV ROC-AUC mean ± std | Holdout cost |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.18 | 0.6050 | 0.4240 | 0.8833 | 0.5730 | 0.8095 | 0.7838 ± 0.0502 | 107 |
| Decision Tree | 0.10 | 0.6050 | 0.4159 | 0.7833 | 0.5434 | 0.6741 | 0.7016 ± 0.0467 | 131 |
| Random Forest | 0.34 | 0.5300 | 0.3836 | 0.9333 | 0.5437 | 0.7912 | 0.7936 ± 0.0461 | 110 |
| Gradient Boosting | 0.13 | 0.5500 | 0.3881 | 0.8667 | 0.5361 | 0.8005 | 0.7962 ± 0.0367 | 122 |

Gradient Boosting was selected at threshold `0.13`. It had the lowest out-of-fold training cost (`398`, versus Logistic Regression `410`, Random Forest `416`, and Decision Tree `476`), bad-risk recall `0.9125`, F1 `0.5824`, the highest mean CV ROC-AUC (`0.7962`), and the lowest ROC-AUC variability (`0.0367`). The holdout results were not used to reverse or alter this selection. Gradient Boosting remains practical to deploy inside the existing sklearn pipeline, while its feature importance provides useful global inspection; it is less intrinsically transparent than Logistic Regression or a single tree, so later explanations and human review remain required.

Load and predict in Python:

```python
from src.models.inference import load_model_artifact, predict_credit_risk

artifact = load_model_artifact("models/finsight_pipeline.joblib")
results = predict_credit_risk(applicant_dataframe, artifact)
```

`applicant_dataframe` must contain the same 20 raw predictor columns used by the loader. The saved pipeline applies the fitted preprocessing automatically and returns bad-risk probability, binary prediction, and decision label.

## Day 4 risk score and explainability

Day 4 uses the saved Day 3 pipeline without fitting or retraining anything. The classifier's fitted classes are `[0, 1]`, where training class `1` means bad credit risk. The inference code explicitly locates class `1` in `classes_` before returning `bad_risk_probability`; it does not assume an unverified probability-column position.

The presentation score is intentionally transparent:

```text
risk_score = bad_risk_probability × 100
```

It ranges from 0 to 100 and is not an official banking credit score. Project-defined display bands are stored centrally in `src/models/risk_score.py`:

| Display band | Score range | Educational wording |
|---|---:|---|
| LOW | 0 to less than 30 | Lower Predicted Risk |
| MODERATE | 30 to less than 60 | Manual Review Suggested |
| HIGH | 60 to 100 | Higher Predicted Risk |

The classification threshold (`0.13`) and display bands (`30` and `60`) are separate concepts. The threshold is the Day 3 cost-sensitive model decision boundary. The bands organize the probability-derived score for presentation; they are not regulatory standards and do not determine loan approval or rejection.

`src.models.predict.predict_applicant` returns a reusable response containing the predicted class, bad-risk probability, saved decision threshold, score, band, neutral suggested action, risk-increasing factors, and protective factors.

SHAP explanations operate on the saved pipeline's transformed feature matrix. For the Gradient Boosting model, code verifies that SHAP values add to the model's raw class-1 decision margin and that its logistic transform equals the bad-risk probability before assigning direction. Positive SHAP values therefore increase predicted bad risk for this exact artifact; negative values decrease it. One-hot feature names are converted to readable category statements, and global importance is aggregated back to original applicant features.

Generate the three real example predictions, global SHAP plot, and report:

```powershell
.venv\Scripts\python.exe -m src.models.run_day4
```

Generated outputs:

- `outputs/predictions/example_predictions.json`
- `outputs/plots/shap_summary.png`
- `reports/explainability_report.md`

Day 4 limitations: the score is only scaled model probability; bands and actions are project conventions; SHAP explains model behavior rather than causality or fairness; the small historical dataset contains sensitive/proxy attributes; and FinSight must remain human-reviewed decision support, not an automated lending system.

## Day 5 Flask dashboard

The web interface lives entirely inside the mentor-required `dashboard/` folder. `dashboard/app.py` is a thin Flask/Jinja layer: it loads the saved pipeline once per process, reads generated analytics/metrics artifacts, validates web input, and calls the reusable Day 4 functions. It never fits preprocessing, retrains a model, or requires live AWS access.

Architecture:

```text
Browser → Flask/Jinja dashboard → saved FinSight pipeline
                              ↘ processed analytics and generated reports/plots
```

Pages:

- `/` — overview cards and real portfolio charts
- `/portfolio` — descriptive Plotly portfolio analytics
- `/predict` — validated 20-feature applicant assessment, risk score, band, action, and local SHAP factors
- `/explain` — generated global SHAP artifact and the latest session-level applicant explanation
- `/performance` — actual Day 3 four-model metrics and saved evaluation plots
- `/whatif` — original/updated profile sensitivity comparison using the same saved pipeline

Start Flask from the repository root:

```powershell
.venv\Scripts\python.exe -m dashboard.app
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000). The development server binds only to localhost and runs with debug mode disabled.

Prediction workflow:

1. Open `/predict` and enter all 20 original predictors. Categorical controls submit the UCI codes expected by the fitted encoder while displaying readable labels.
2. Flask rejects missing, invalid, or out-of-range fields and never accepts `applicant_id`, `credit_risk`, `credit_risk_label`, or `is_bad_risk` as inputs.
3. The saved pipeline returns class-1 bad-risk probability; Day 4 logic derives the score, display band, neutral action, and SHAP factors.
4. The result is available to `/explain` for the current browser session.

What-if workflow:

1. Enter an original and updated valid profile at `/whatif`.
2. Both are evaluated independently with the same saved model and threshold.
3. The page reports probability and score differences. It describes model sensitivity only—not causality, financial advice, or guaranteed real-world risk reduction.

The explainability page reuses `outputs/plots/shap_summary.png`; it does not regenerate global SHAP calculations on page load. The performance page reads `outputs/metrics/model_metrics.json`; no route retrains or tunes models. The local dashboard uses the processed CSV directly, while the verified S3 → Glue Data Catalog via Athena DDL → Athena architecture remains unchanged and optional for local page loading.

Website disclaimer: FinSight is an educational machine-learning prototype based on a public historical benchmark dataset. Its estimates, bands, actions, and explanations are not intended for actual lending or financial decisions. Historical-data limitations, sensitive/proxy fields, and fairness concerns require human review in real credit systems.

## Target definition

The raw UCI encoding is preserved on disk: `1 = good credit risk`, `2 = bad credit risk`. Model training explicitly maps it to `0 = good`, `1 = bad`. Bad risk is the positive event, and its probability is the classifier probability for class `1`. The 5:1 asymmetric error cost is used for out-of-fold threshold selection and model comparison.

## Project layout

```text
data/raw/          cached source data (ignored by Git)
data/processed/    later clean/model-ready datasets
notebooks/         optional exploration; reusable logic belongs in src/
src/data/          acquisition, schema, loading, validation, reports
src/features/      later feature engineering
src/models/        training, evaluation, persistence, and inference
src/cloud/         reusable Boto3 S3 helpers
src/utils/         shared helpers
dashboard/         Flask application, HTML/CSS, Bootstrap templates
models/            serialized pipeline and model metadata
outputs/           generated plots, predictions, and machine-readable metrics
reports/           written reports and decision log
tests/             automated checks
```

## Preprocessing workflow

1. Validate the raw dataset before analysis.
2. Separate the raw predictors from `credit_risk` and map bad risk to binary class `1`.
3. Stratify the untouched predictors and target into 80% training and 20% testing partitions.
4. Fit all imputers, scaling parameters, and one-hot categories on training data only.
5. Apply the fitted transformer to test or inference rows through the same sklearn `Pipeline`.

Use `build_model_pipeline(estimator, scale_numeric=True)` for logistic regression, SVM, k-nearest-neighbor, or neural models. Use `scale_numeric=False` for decision trees and tree ensembles. One-hot encoding uses `handle_unknown="ignore"`, so a valid new category does not break inference.

## Day 2 analytics dataset and AWS query layer

Generate the analytics CSV and verify the Day 2 implementation without training a model:

```powershell
.venv\Scripts\python.exe -m src.data.run_eda
.venv\Scripts\python.exe -m src.data.preprocess
.venv\Scripts\python.exe -m jupyter nbconvert --to notebook --execute notebooks/01_eda.ipynb --inplace
.venv\Scripts\python.exe -m pytest tests/test_day2.py tests/test_preprocessing.py -q
```

`data/processed/german_credit_analytics.csv` contains all 1,000 validated source rows and the original 21 columns, plus a stable source-row `applicant_id`, readable `credit_risk_label`, and binary `is_bad_risk` (`1 = bad`). It is deliberately unscaled and unencoded so it remains understandable in S3, Glue, and Athena. The raw file is never overwritten. This analytics export must not be used as model input because it contains the original and derived targets.

The reusable ML path is separate: `src.data.preprocess.prepare_data` creates the stratified split and fits its `ColumnTransformer` only on training rows. `src.features.preprocessing.build_model_pipeline` attaches the same preprocessing to an estimator for identical inference-time transformations.

`src/cloud/s3_utils.py` provides credential-neutral upload/download/existence helpers. `src/cloud/athena_queries.py` contains configurable applicant count, risk count, average credit amount, average duration, and grouped risk queries. Override the defaults with `FINSIGHT_ATHENA_DATABASE` and `FINSIGHT_ATHENA_TABLE`, or pass `--database` and `--table` on the command line.

### Implemented S3, Glue Data Catalog, and Athena architecture

The Day 2 AWS deployment is complete and was verified against the live processed data:

```text
Amazon S3
→ AWS Glue Data Catalog (table registered using Athena DDL)
→ Amazon Athena
→ SQL analytics
```

- The raw German Credit dataset is stored in Amazon S3.
- The processed analytics object is `s3://finsight-credit-risk/processed/german_credit_analytics.csv`.
- The Glue Data Catalog database is `finsight`.
- The catalog table is `finsight.german_credit_analytics`.
- The table was registered by running the generated `CREATE EXTERNAL TABLE` statement in Athena. Because Athena uses the Glue Data Catalog for metadata, the DDL registered the table there without a crawler.
- `SELECT * ... LIMIT 10`, applicant retrieval, and grouped credit-risk SQL were successfully executed against the S3 CSV.

The account denied the permissions required to create an AWS Glue crawler. A crawler was therefore not used and is not required by this implementation. It remains only an optional catalog-discovery alternative for an account with suitable permissions.

The detailed implementation record and verification checklist is in `reports/day2_aws_setup.md`. The verified AWS region is `eu-north-1`. The commands used to generate the local CSV and the table DDL are:

```powershell
$env:AWS_PROFILE = 'finsight'
$env:AWS_DEFAULT_REGION = 'eu-north-1'
.venv\Scripts\python.exe -m src.data.preprocess
aws s3 cp data/processed/german_credit_analytics.csv s3://finsight-credit-risk/processed/german_credit_analytics.csv --profile finsight --region eu-north-1
.venv\Scripts\python.exe -m src.cloud.athena_queries --database finsight --table german_credit_analytics --s3-location s3://finsight-credit-risk/processed/
```

The printed DDL was executed in Athena and the printed SELECT queries were used for verification. IAM needs S3 read access to the processed prefix, S3 read/write access to the configured Athena-results prefix, and Glue Data Catalog/Athena permissions. No AWS credentials or account IDs are committed to this repository.

## Data source and limitations

Hofmann, H. (1994), *Statlog (German Credit Data)*, UCI Machine Learning Repository, DOI: [10.24432/C5NC77](https://doi.org/10.24432/C5NC77), licensed CC BY 4.0.

This project uses the original symbolic file because it preserves categorical meanings. UCI's later [South German Credit](https://archive.ics.uci.edu/dataset/573/south+german+credit+update) page warns that the older dataset's coding documentation contains errors and provides corrected background. FinSight therefore reproduces the requested original UCI codebook but records this limitation. The combined `personal_status_sex`, foreign-worker status, and age are sensitive or potentially proxy attributes; future decision support must present model output as advisory, not an automated lending decision.

## Seven-day roadmap

1. Data acquisition, validation, profiling, dictionary (complete)
2. Exploratory analysis and leakage-aware preprocessing (complete)
3. Four-model comparison, cost-sensitive selection, and persistence (complete)
4. Risk scoring, decision-support response, and global/local SHAP explanations (complete)
5. Flask decision-support dashboard (complete)
6. Final technical stabilization and verification (complete)
7. Final reporting and viva preparation
