# Day 3 Model Evaluation

Generated from the current validated dataset and pipeline at `2026-09-17T14:53:49.602679+00:00`. All results below were computed by the training command.

## Dataset and predictors

- Dataset: `data/raw/german.data` (1000 rows, 21 columns)
- SHA-256: `b21f3d81db8071257d5ff1deaeba1fd4303b62712e6fcc9715c7a86202cb5871`
- Predictors used: 20 original applicant features
- Excluded from predictors: `applicant_id, credit_risk, credit_risk_label, is_bad_risk`
- The analytics CSV is not used as a model matrix; identifiers and target-derived analytics fields cannot enter `X`.

## Target and probability convention

- Original target: `credit_risk` (`1 = good credit risk`, `2 = bad credit risk`).
- Training target: `0 = good credit risk`, `1 = bad credit risk`.
- Bad risk is explicitly training class `1` for precision, recall, F1, ROC-AUC, thresholding, and cost calculations.
- `bad_risk_probability` is the `predict_proba` column whose fitted classifier class label is `1`; inference validates this mapping.

## Methodology

- Split: 800 training rows and 200 untouched holdout rows, stratified with `random_state=42`.
- Cross-validation: 5-fold StratifiedKFold on training data.
- All imputation, scaling, one-hot encoding, tuning, and threshold selection are fitted inside sklearn pipelines on training folds only.
- Models: Logistic Regression, Decision Tree, Random Forest, and Gradient Boosting only.
- Logistic Regression, Random Forest, and Gradient Boosting receive an eight-candidate randomized search; the shallow Decision Tree remains an untuned interpretable baseline.
- Cost: `5 * bad predicted good (FN) + 1 * good predicted bad (FP)`.
- Each decision threshold is chosen from out-of-fold training probabilities. The holdout is used once for final reporting, not model or threshold selection.

## Holdout comparison at training-selected cost thresholds

| Model | Threshold | Accuracy | Precision | Recall | F1 | ROC-AUC | Cost |
|---|---:|---:|---:|---:|---:|---:|---:|
| logistic_regression | 0.18 | 0.6050 | 0.4240 | 0.8833 | 0.5730 | 0.8095 | 107 |
| decision_tree | 0.10 | 0.6050 | 0.4159 | 0.7833 | 0.5434 | 0.6741 | 131 |
| random_forest | 0.34 | 0.5300 | 0.3836 | 0.9333 | 0.5437 | 0.7912 | 110 |
| gradient_boosting | 0.13 | 0.5500 | 0.3881 | 0.8667 | 0.5361 | 0.8005 | 122 |

The precision, recall, and F1 values above all treat bad risk as the positive class. Confusion matrices use rows as actual and columns as predicted, ordered `[good, bad]`.

## Five-fold cross-validation summary

Cross-validation accuracy, precision, recall, and F1 use each fitted classifier's default prediction threshold; ROC-AUC is threshold-independent. Cost-sensitive thresholds are selected separately from the complete set of out-of-fold training probabilities.

| Model | Accuracy mean | Precision mean | Recall mean | F1 mean | ROC-AUC mean | ROC-AUC std |
|---|---:|---:|---:|---:|---:|---:|
| logistic_regression | 0.7475 | 0.6148 | 0.3958 | 0.4771 | 0.7838 | 0.0502 |
| decision_tree | 0.6950 | 0.4814 | 0.4458 | 0.4553 | 0.7016 | 0.0467 |
| random_forest | 0.7075 | 0.5083 | 0.7250 | 0.5971 | 0.7936 | 0.0461 |
| gradient_boosting | 0.7588 | 0.6338 | 0.4667 | 0.5361 | 0.7962 | 0.0367 |

## Lightweight tuning

- **logistic_regression**: 8 candidates; best tuning CV ROC-AUC 0.7838; parameters `{"model__C": 0.1, "model__class_weight": null}`.
- **random_forest**: 8 candidates; best tuning CV ROC-AUC 0.7936; parameters `{"model__class_weight": "balanced", "model__max_depth": 5, "model__max_features": "sqrt", "model__min_samples_leaf": 2, "model__n_estimators": 200}`.
- **gradient_boosting**: 8 candidates; best tuning CV ROC-AUC 0.7962; parameters `{"model__learning_rate": 0.05, "model__max_depth": 3, "model__min_samples_leaf": 5, "model__n_estimators": 200, "model__subsample": 0.8}`.

## Final selection

Selected: **gradient_boosting**, threshold **0.13**.

Selected from training evidence using lowest out-of-fold 5:1 misclassification cost, then mean CV ROC-AUC, bad-risk recall, F1, and ROC-AUC stability. Holdout metrics were not used for selection; interpretability and deployment practicality were reviewed in the report.

Its out-of-fold training cost was **398**, with bad-risk recall **0.9125** and F1 **0.5824**. Its untouched holdout cost was **122**.
The 5:1 German Credit error cost makes missing genuinely bad-risk applicants more costly than incorrectly flagging good-risk applicants. Accuracy alone therefore does not determine the selected model.
The complete fitted preprocessing-and-classifier pipeline is persisted, so inference applies exactly the training transformations.

## Limitations

- The dataset has only 1,000 historical applicants; subgroup and holdout estimates can be unstable.
- Threshold and model selection reflect the documented 5:1 cost assumption, which may differ from a real lender's operational costs.
- The data contains sensitive or proxy attributes; predictive performance does not establish fairness, causality, or suitability for automated lending decisions.
- Results are specific to this dataset version, split seed, feature definitions, and library versions.
- FinSight output is decision support and requires human review; it is not an approval or denial decision.

## Generated artifacts

- `models/finsight_pipeline.joblib`
- `models/model_metadata.json`
- `outputs/metrics/model_metrics.json`
- `outputs/plots/model_comparison.png`
- `outputs/plots/model_roc_curves.png`
- `outputs/plots/final_model_confusion_matrix.png`
- `outputs/plots/final_model_feature_importance.png`
