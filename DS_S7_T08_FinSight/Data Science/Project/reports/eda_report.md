# Exploratory Data Analysis Summary

All figures and statistics are computed from the validated raw UCI dataset.

## Data quality

- Completeness: **100.00%**
- Exact duplicates: **0**
- Contract validation errors: **0**

## Target balance

- Good risk: **700 (70.00%)**
- Bad risk: **300 (30.00%)**
- The 70/30 imbalance supports stratified splitting and metrics beyond accuracy.

## Numeric associations with bad risk

These are univariate Pearson correlations with `is_bad_risk`; they are descriptive, not causal.

| Feature | Correlation |
|---|---:|
| duration_months | 0.2149 |
| credit_amount | 0.1547 |
| age_years | -0.0911 |
| installment_rate | 0.0724 |
| existing_credits | -0.0457 |
| residence_duration | 0.0030 |
| dependents | -0.0030 |

## Numeric comparisons by risk

| Risk | Age (mean years) | Amount (mean DM) | Duration (mean months) |
|---|---:|---:|---:|
| good | 36.22 | 2985.46 | 19.21 |
| bad | 33.96 | 3938.13 | 24.86 |

## Important categorical relationships

Sample sizes accompany rates: small groups can be unstable. Codes use the official data dictionary. Dashed chart lines show the overall 30% bad-risk rate.

| Feature | Code | Meaning | Applicants | Bad risk (%) |
|---|---|---|---:|---:|
| checking_account_status | A11 | < 0 DM | 274 | 49.27 |
| checking_account_status | A12 | 0 to < 200 DM | 269 | 39.03 |
| checking_account_status | A13 | >= 200 DM or salary assignment for at least 1 year | 63 | 22.22 |
| checking_account_status | A14 | no checking account | 394 | 11.68 |
| savings_status | A61 | < 100 DM | 603 | 35.99 |
| savings_status | A62 | 100 to < 500 DM | 103 | 33.01 |
| savings_status | A65 | unknown or no savings account | 183 | 17.49 |
| savings_status | A63 | 500 to < 1000 DM | 63 | 17.46 |
| savings_status | A64 | >= 1000 DM | 48 | 12.50 |
| employment_duration | A72 | < 1 year | 172 | 40.70 |
| employment_duration | A71 | unemployed | 62 | 37.10 |
| employment_duration | A73 | 1 to < 4 years | 339 | 30.68 |
| employment_duration | A75 | >= 7 years | 253 | 25.30 |
| employment_duration | A74 | 4 to < 7 years | 174 | 22.41 |
| credit_history | A30 | no credits taken or all credits paid back duly | 40 | 62.50 |
| credit_history | A31 | all credits at this bank paid back duly | 49 | 57.14 |
| credit_history | A32 | existing credits paid back duly to date | 530 | 31.89 |
| credit_history | A33 | delay in paying off in the past | 88 | 31.82 |
| credit_history | A34 | critical account or other credits outside this bank | 293 | 17.06 |
| purpose | A46 | education | 50 | 44.00 |
| purpose | A410 | other | 12 | 41.67 |
| purpose | A40 | new car | 234 | 38.03 |
| purpose | A45 | repairs | 22 | 36.36 |
| purpose | A49 | business | 97 | 35.05 |
| purpose | A44 | domestic appliances | 12 | 33.33 |
| purpose | A42 | furniture/equipment | 181 | 32.04 |
| purpose | A43 | radio/television | 280 | 22.14 |
| purpose | A41 | used car | 103 | 16.50 |
| purpose | A48 | retraining | 9 | 11.11 |
| housing | A153 | free | 108 | 40.74 |
| housing | A151 | rent | 179 | 39.11 |
| housing | A152 | own | 713 | 26.09 |

These full-dataset descriptions are exploratory, not causal or independent holdout evaluation. No rows are deleted or thresholds tuned from these plots.

## Potential outliers (1.5 × IQR rule)

Flags are retained for investigation; no rows are removed automatically. The IQR rule is weak for low-cardinality count/ordinal fields (especially `dependents`, whose IQR is zero), so those flags are not evidence of invalid data.

| Feature | Lower fence | Upper fence | Flagged rows |
|---|---:|---:|---:|
| duration_months | -6.00 | 42.00 | 70 |
| credit_amount | -2544.62 | 7882.38 | 72 |
| installment_rate | -1.00 | 7.00 | 0 |
| residence_duration | -1.00 | 7.00 | 0 |
| age_years | 4.50 | 64.50 | 23 |
| existing_credits | -0.50 | 3.50 | 6 |
| dependents | 1.00 | 1.00 | 155 |

## Report figures

- `../outputs/plots/target_distribution.png`
- `../outputs/plots/numeric_distributions.png`
- `../outputs/plots/categorical_distributions.png`
- `../outputs/plots/categorical_risk_relationships.png`
- `../outputs/plots/numeric_risk_relationships.png`
