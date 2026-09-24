# Data Dictionary

Source: [UCI Statlog (German Credit Data)](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data), original symbolic file `german.data`.

> Documentation caveat: UCI's later South German Credit dataset notes errors in the original coding information. FinSight preserves the requested original Statlog codes and labels them from the official Statlog page; sensitive combined attributes should not be treated as clean demographic fields.

| # | FinSight name | Type | Unit | Official meaning |
|---:|---|---|---|---|
| 1 | `checking_account_status` | categorical | — | Status of existing checking account |
| 2 | `duration_months` | integer | months | Credit duration |
| 3 | `credit_history` | categorical | — | Credit history |
| 4 | `purpose` | categorical | — | Loan purpose |
| 5 | `credit_amount` | integer | Deutsche Mark (DM) | Credit amount |
| 6 | `savings_status` | categorical | — | Savings account or bonds |
| 7 | `employment_duration` | categorical | — | Present employment duration |
| 8 | `installment_rate` | integer | ordinal category | Installment rate as percentage of disposable income |
| 9 | `personal_status_sex` | categorical | — | Combined personal status and sex |
| 10 | `other_debtors` | categorical | — | Other debtors or guarantors |
| 11 | `residence_duration` | integer | ordinal category | Time at present residence |
| 12 | `property` | categorical | — | Most valuable property category |
| 13 | `age_years` | integer | years | Applicant age |
| 14 | `other_installment_plans` | categorical | — | Other installment plans |
| 15 | `housing` | categorical | — | Housing arrangement |
| 16 | `existing_credits` | integer | count | Existing credits at this bank |
| 17 | `job` | categorical | — | Job category |
| 18 | `dependents` | integer | count | People liable for maintenance |
| 19 | `telephone` | categorical | — | Telephone registered in applicant name |
| 20 | `foreign_worker` | categorical | — | Foreign-worker status |
| 21 | `credit_risk` | target | — | Observed credit risk: 1=good, 2=bad |

## Original categorical codes

### `checking_account_status`

| Code | Official meaning |
|---|---|
| A11 | < 0 DM |
| A12 | 0 to < 200 DM |
| A13 | >= 200 DM or salary assignment for at least 1 year |
| A14 | no checking account |

### `credit_history`

| Code | Official meaning |
|---|---|
| A30 | no credits taken or all credits paid back duly |
| A31 | all credits at this bank paid back duly |
| A32 | existing credits paid back duly to date |
| A33 | delay in paying off in the past |
| A34 | critical account or other credits outside this bank |

### `purpose`

| Code | Official meaning |
|---|---|
| A40 | new car |
| A41 | used car |
| A42 | furniture/equipment |
| A43 | radio/television |
| A44 | domestic appliances |
| A45 | repairs |
| A46 | education |
| A47 | vacation (documented by UCI as not occurring) |
| A48 | retraining |
| A49 | business |
| A410 | other |

### `savings_status`

| Code | Official meaning |
|---|---|
| A61 | < 100 DM |
| A62 | 100 to < 500 DM |
| A63 | 500 to < 1000 DM |
| A64 | >= 1000 DM |
| A65 | unknown or no savings account |

### `employment_duration`

| Code | Official meaning |
|---|---|
| A71 | unemployed |
| A72 | < 1 year |
| A73 | 1 to < 4 years |
| A74 | 4 to < 7 years |
| A75 | >= 7 years |

### `personal_status_sex`

| Code | Official meaning |
|---|---|
| A91 | male: divorced/separated |
| A92 | female: divorced/separated/married |
| A93 | male: single |
| A94 | male: married/widowed |
| A95 | female: single |

### `other_debtors`

| Code | Official meaning |
|---|---|
| A101 | none |
| A102 | co-applicant |
| A103 | guarantor |

### `property`

| Code | Official meaning |
|---|---|
| A121 | real estate |
| A122 | building-society savings agreement or life insurance |
| A123 | car or other property not in savings |
| A124 | unknown or no property |

### `other_installment_plans`

| Code | Official meaning |
|---|---|
| A141 | bank |
| A142 | stores |
| A143 | none |

### `housing`

| Code | Official meaning |
|---|---|
| A151 | rent |
| A152 | own |
| A153 | free |

### `job`

| Code | Official meaning |
|---|---|
| A171 | unemployed/unskilled non-resident |
| A172 | unskilled resident |
| A173 | skilled employee/official |
| A174 | management/self-employed/highly qualified employee/officer |

### `telephone`

| Code | Official meaning |
|---|---|
| A191 | none |
| A192 | yes, registered in applicant name |

### `foreign_worker`

| Code | Official meaning |
|---|---|
| A201 | yes |
| A202 | no |

## Target encoding

The raw target is retained as `credit_risk`: **1 = good credit risk**, **2 = bad credit risk**. No binary remapping is performed on Day 1. For future modeling, the planned positive event is bad risk (`2`), because recall and cost for risky applicants are decision-relevant; that transformation must occur inside the modeling pipeline.
