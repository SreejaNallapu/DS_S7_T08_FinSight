"""Single source of truth for the original UCI symbolic dataset schema."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ColumnSpec:
    name: str
    kind: str
    description: str
    unit: str = ""


COLUMNS = (
    ColumnSpec("checking_account_status", "categorical", "Status of existing checking account"),
    ColumnSpec("duration_months", "integer", "Credit duration", "months"),
    ColumnSpec("credit_history", "categorical", "Credit history"),
    ColumnSpec("purpose", "categorical", "Loan purpose"),
    ColumnSpec("credit_amount", "integer", "Credit amount", "Deutsche Mark (DM)"),
    ColumnSpec("savings_status", "categorical", "Savings account or bonds"),
    ColumnSpec("employment_duration", "categorical", "Present employment duration"),
    ColumnSpec("installment_rate", "integer", "Installment rate as percentage of disposable income", "ordinal category"),
    ColumnSpec("personal_status_sex", "categorical", "Combined personal status and sex"),
    ColumnSpec("other_debtors", "categorical", "Other debtors or guarantors"),
    ColumnSpec("residence_duration", "integer", "Time at present residence", "ordinal category"),
    ColumnSpec("property", "categorical", "Most valuable property category"),
    ColumnSpec("age_years", "integer", "Applicant age", "years"),
    ColumnSpec("other_installment_plans", "categorical", "Other installment plans"),
    ColumnSpec("housing", "categorical", "Housing arrangement"),
    ColumnSpec("existing_credits", "integer", "Existing credits at this bank", "count"),
    ColumnSpec("job", "categorical", "Job category"),
    ColumnSpec("dependents", "integer", "People liable for maintenance", "count"),
    ColumnSpec("telephone", "categorical", "Telephone registered in applicant name"),
    ColumnSpec("foreign_worker", "categorical", "Foreign-worker status"),
    ColumnSpec("credit_risk", "target", "Observed credit risk: 1=good, 2=bad"),
)

COLUMN_NAMES = [column.name for column in COLUMNS]
TARGET = "credit_risk"
NUMERIC_COLUMNS = [column.name for column in COLUMNS if column.kind == "integer"]
CATEGORICAL_COLUMNS = [column.name for column in COLUMNS if column.kind == "categorical"]

CATEGORY_LABELS: dict[str, dict[str, str]] = {
    "checking_account_status": {"A11": "< 0 DM", "A12": "0 to < 200 DM", "A13": ">= 200 DM or salary assignment for at least 1 year", "A14": "no checking account"},
    "credit_history": {"A30": "no credits taken or all credits paid back duly", "A31": "all credits at this bank paid back duly", "A32": "existing credits paid back duly to date", "A33": "delay in paying off in the past", "A34": "critical account or other credits outside this bank"},
    "purpose": {"A40": "new car", "A41": "used car", "A42": "furniture/equipment", "A43": "radio/television", "A44": "domestic appliances", "A45": "repairs", "A46": "education", "A47": "vacation (documented by UCI as not occurring)", "A48": "retraining", "A49": "business", "A410": "other"},
    "savings_status": {"A61": "< 100 DM", "A62": "100 to < 500 DM", "A63": "500 to < 1000 DM", "A64": ">= 1000 DM", "A65": "unknown or no savings account"},
    "employment_duration": {"A71": "unemployed", "A72": "< 1 year", "A73": "1 to < 4 years", "A74": "4 to < 7 years", "A75": ">= 7 years"},
    "personal_status_sex": {"A91": "male: divorced/separated", "A92": "female: divorced/separated/married", "A93": "male: single", "A94": "male: married/widowed", "A95": "female: single"},
    "other_debtors": {"A101": "none", "A102": "co-applicant", "A103": "guarantor"},
    "property": {"A121": "real estate", "A122": "building-society savings agreement or life insurance", "A123": "car or other property not in savings", "A124": "unknown or no property"},
    "other_installment_plans": {"A141": "bank", "A142": "stores", "A143": "none"},
    "housing": {"A151": "rent", "A152": "own", "A153": "free"},
    "job": {"A171": "unemployed/unskilled non-resident", "A172": "unskilled resident", "A173": "skilled employee/official", "A174": "management/self-employed/highly qualified employee/officer"},
    "telephone": {"A191": "none", "A192": "yes, registered in applicant name"},
    "foreign_worker": {"A201": "yes", "A202": "no"},
}

TARGET_LABELS = {1: "good", 2: "bad"}

