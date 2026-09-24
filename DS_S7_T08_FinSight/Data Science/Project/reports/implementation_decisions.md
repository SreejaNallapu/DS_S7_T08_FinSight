# Implementation Decisions

## Day 1

1. **Use the original symbolic UCI file.** It retains categories such as `A11` instead of using UCI's separately prepared numeric representation. Encoding will later be learned inside a scikit-learn pipeline to avoid leakage.
2. **Keep acquisition separate from loading.** `acquire.py` owns network/cache/provenance behavior; `load.py` is deterministic and can operate offline.
3. **Use a single schema module.** Column names, roles, units, categorical code meanings, and target labels cannot silently diverge across reports, tests, and future features.
4. **Treat contract violations as errors.** Wrong row/column counts, missing values, unknown codes, or unexpected targets stop the pipeline. Exact duplicates are warnings because they may be legitimate identical records and should not be deleted without evidence.
5. **Preserve raw target values.** The source codes remain 1/2 on Day 1. Future model code will explicitly map bad risk to the positive class.
6. **Generate rather than hand-write statistics.** Both report formats are produced from the cached file so no result is fabricated or allowed to drift.
7. **Keep notebooks optional.** Reusable logic lives under `src/`; notebooks can call it for presentation-oriented EDA later.
8. **Do not model yet.** Avoiding premature tuning keeps Day 1 focused on provenance and data understanding.
9. **Pin the official raw-file checksum.** A cached or newly downloaded file is accepted only when its SHA-256 matches the reviewed UCI symbolic file. This makes silent corruption or upstream content drift visible.
10. **Use project-relative CLI defaults.** The Day 1 command resolves default data and report locations from the source tree, so it behaves consistently even when launched outside the repository root.
11. **Keep observed facts separate from documentation.** The quality report contains computed observed codes and ranges; the data dictionary contains UCI's documented meanings, including codes that are valid but absent from this sample.

## EDA and preprocessing

1. **Map bad risk to the positive class explicitly.** Raw code `2` becomes `is_bad_risk=1`; raw code `1` becomes `0`. The original cached data is never modified.
2. **Split raw rows before fitting transformations.** Stratification preserves the 70/30 class balance, while a fixed seed makes the 80/20 split reproducible. Imputer, scaler, and encoder state is learned only from training rows.
3. **One-hot encode nominal codes.** Values such as `A11` are labels, not numeric magnitudes. Unknown categories are ignored at inference rather than causing failure.
4. **Make scaling estimator-dependent.** Standardization is enabled for scale-sensitive estimators and can be disabled for tree models. It is never applied to one-hot categorical columns.
5. **Retain unusual values.** The 1.5×IQR rule is used only to flag observations for review. Credit amount, duration, and age extremes can be legitimate, so EDA does not delete or cap them without domain evidence.
6. **Keep the notebook thin.** The notebook imports validation, EDA, and preprocessing functions from `src/`, preventing a second, inconsistent implementation.

## Day 3 model experimentation

1. **Reserve one stratified holdout.** The 80/20 split from preprocessing remains fixed. Five-fold stratified CV and all searches operate on the 800 training rows only.
2. **Tune compactly.** Logistic Regression, Random Forest, and Gradient Boosting receive eight-candidate randomized searches scored by ROC-AUC. The shallow Decision Tree remains a transparent baseline. XGBoost and other additional model families are omitted to keep the required comparison simple and reproducible.
3. **Encode UCI's asymmetric cost directly.** A bad applicant predicted good is a false negative and costs 5; a good applicant predicted bad is a false positive and costs 1.
4. **Select thresholds without holdout leakage.** Each model's threshold minimizes cost on out-of-fold training probabilities. Ties favor risky-applicant recall and then F1.
5. **Select the final model without consulting holdout outcomes.** Lowest out-of-fold cost wins, followed by mean CV ROC-AUC, bad-risk recall, F1, and ROC-AUC stability. The test set is reported once for honest generalization evidence.
6. **Persist the complete pipeline.** `models/finsight_pipeline.joblib` contains fitted preprocessing, classifier, selected threshold, feature order, dataset checksum, and target/probability definition. `models/model_metadata.json` provides the same operational contract in a readable format. The historical `final_model.joblib` filename is refreshed as a compatibility copy so it cannot silently refer to an older run.
7. **Treat feature importance as descriptive.** Tree impurity importance and absolute linear coefficients help explanation but do not establish causality or fairness.
8. **Enforce the leakage boundary.** Only the 20 original applicant predictors enter `X`; `applicant_id`, `credit_risk`, `credit_risk_label`, and `is_bad_risk` are rejected or excluded.
9. **Resolve bad-risk probability by class label.** Inference locates classifier class `1` in `classes_` and uses that `predict_proba` column, rather than assuming column position without validation.
