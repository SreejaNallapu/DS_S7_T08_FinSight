# Day 1 Data Quality Report

Generated from the cached raw UCI file; values below are computed, not estimates.

## Overview

- Shape: **1,000 rows × 21 columns** (20 predictors + 1 target)
- Missing values: **0**
- Exact duplicate rows: **0**
- Validation passed: **True**

## Target distribution

| Original code | Meaning | Count | Percent |
|---:|---|---:|---:|
| 1 | good | 700 | 70.00% |
| 2 | bad | 300 | 30.00% |

Completeness: **100.0%**. Validation checks schema, row count, missingness, target codes, documented categories, and finite positive integer numeric values.

## Column profile

| Column | dtype | Missing | Unique |
|---|---|---:|---:|
| checking_account_status | category | 0 | 4 |
| duration_months | int64 | 0 | 33 |
| credit_history | category | 0 | 5 |
| purpose | category | 0 | 10 |
| credit_amount | int64 | 0 | 921 |
| savings_status | category | 0 | 5 |
| employment_duration | category | 0 | 5 |
| installment_rate | int64 | 0 | 4 |
| personal_status_sex | category | 0 | 4 |
| other_debtors | category | 0 | 3 |
| residence_duration | int64 | 0 | 4 |
| property | category | 0 | 4 |
| age_years | int64 | 0 | 53 |
| other_installment_plans | category | 0 | 3 |
| housing | category | 0 | 3 |
| existing_credits | int64 | 0 | 4 |
| job | category | 0 | 4 |
| dependents | int64 | 0 | 2 |
| telephone | category | 0 | 2 |
| foreign_worker | category | 0 | 2 |
| credit_risk | int64 | 0 | 2 |

## Observed categorical codes

Codes listed here are observed in this copy; their meanings are in `data_dictionary.md`.

| Column | Observed codes |
|---|---|
| checking_account_status | `A11`, `A12`, `A13`, `A14` |
| credit_history | `A30`, `A31`, `A32`, `A33`, `A34` |
| purpose | `A40`, `A41`, `A410`, `A42`, `A43`, `A44`, `A45`, `A46`, `A48`, `A49` |
| savings_status | `A61`, `A62`, `A63`, `A64`, `A65` |
| employment_duration | `A71`, `A72`, `A73`, `A74`, `A75` |
| personal_status_sex | `A91`, `A92`, `A93`, `A94` |
| other_debtors | `A101`, `A102`, `A103` |
| property | `A121`, `A122`, `A123`, `A124` |
| other_installment_plans | `A141`, `A142`, `A143` |
| housing | `A151`, `A152`, `A153` |
| job | `A171`, `A172`, `A173`, `A174` |
| telephone | `A191`, `A192` |
| foreign_worker | `A201`, `A202` |

## Numeric ranges

| Column | Min | Mean | Median | Max |
|---|---:|---:|---:|---:|
| duration_months | 4 | 20.90 | 18 | 72 |
| credit_amount | 250 | 3271.26 | 2319.5 | 18424 |
| installment_rate | 1 | 2.97 | 3 | 4 |
| residence_duration | 1 | 2.85 | 3 | 4 |
| age_years | 19 | 35.55 | 33 | 75 |
| existing_credits | 1 | 1.41 | 1 | 4 |
| dependents | 1 | 1.16 | 1 | 2 |
