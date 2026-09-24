"""Generate machine-readable and human-readable Day 1 artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.data.schema import CATEGORY_LABELS, COLUMNS, TARGET, TARGET_LABELS
from src.data.validate import ValidationResult


def build_quality_summary(frame: pd.DataFrame, validation: ValidationResult) -> dict:
    target_counts = frame[TARGET].value_counts().sort_index()
    return {
        "dimensions": {"rows": len(frame), "columns": len(frame.columns), "features": len(frame.columns) - 1},
        "target_distribution": {
            TARGET_LABELS[int(code)]: {"code": int(code), "count": int(count), "percentage": round(100 * int(count) / len(frame), 2)}
            for code, count in target_counts.items()
        },
        "column_dtypes": {name: str(dtype) for name, dtype in frame.dtypes.items()},
        "missing_by_column": {name: int(value) for name, value in frame.isna().sum().items()},
        "total_missing": int(frame.isna().sum().sum()),
        "completeness_percentage": round(100 * (1 - frame.isna().sum().sum() / frame.size), 2) if frame.size else 0.0,
        "duplicate_rows": int(frame.duplicated().sum()),
        "unique_values": {name: int(frame[name].nunique(dropna=True)) for name in frame.columns},
        "observed_values": {
            name: sorted(str(value) for value in frame[name].dropna().unique())
            for name in frame.select_dtypes(include=["category"]).columns
        },
        "numeric_summary": json.loads(frame.describe().round(2).to_json()),
        "validation": {"passed": validation.is_valid, "errors": list(validation.errors), "warnings": list(validation.warnings)},
    }


def write_reports(frame: pd.DataFrame, validation: ValidationResult, reports_dir: Path, outputs_dir: Path | None = None) -> None:
    reports_dir.mkdir(parents=True, exist_ok=True)
    summary = build_quality_summary(frame, validation)
    outputs_dir = outputs_dir if outputs_dir is not None else reports_dir
    outputs_dir.mkdir(parents=True, exist_ok=True)
    (outputs_dir / "data_quality_report.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    lines = ["# Day 1 Data Quality Report", "", "Generated from the cached raw UCI file; values below are computed, not estimates.", "", "## Overview", "", f"- Shape: **{len(frame):,} rows × {len(frame.columns)} columns** (20 predictors + 1 target)", f"- Missing values: **{summary['total_missing']}**", f"- Exact duplicate rows: **{summary['duplicate_rows']}**", f"- Validation passed: **{validation.is_valid}**", "", "## Target distribution", "", "| Original code | Meaning | Count | Percent |", "|---:|---|---:|---:|"]
    for label, values in summary["target_distribution"].items():
        lines.append(f"| {values['code']} | {label} | {values['count']} | {values['percentage']:.2f}% |")
    lines += ["", f"Completeness: **{summary['completeness_percentage']}%**. Validation checks schema, row count, missingness, target codes, documented categories, and finite positive integer numeric values."]
    lines += ["", "## Column profile", "", "| Column | dtype | Missing | Unique |", "|---|---|---:|---:|"]
    for column in frame.columns:
        lines.append(f"| {column} | {frame[column].dtype} | {summary['missing_by_column'][column]} | {summary['unique_values'][column]} |")
    lines += ["", "## Observed categorical codes", "", "Codes listed here are observed in this copy; their meanings are in `data_dictionary.md`.", "", "| Column | Observed codes |", "|---|---|"]
    for column, values in summary["observed_values"].items():
        lines.append(f"| {column} | {', '.join(f'`{value}`' for value in values)} |")
    numeric = frame.select_dtypes(include="number").drop(columns=[TARGET], errors="ignore")
    lines += ["", "## Numeric ranges", "", "| Column | Min | Mean | Median | Max |", "|---|---:|---:|---:|---:|"]
    for column in numeric.columns:
        series = numeric[column]
        lines.append(
            f"| {column} | {series.min():g} | {series.mean():.2f} | "
            f"{series.median():g} | {series.max():g} |"
        )
    if validation.warnings:
        lines += ["", "## Validation warnings", ""] + [f"- {item}" for item in validation.warnings]
    (reports_dir / "data_quality_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    dictionary = ["# Data Dictionary", "", "Source: [UCI Statlog (German Credit Data)](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data), original symbolic file `german.data`.", "", "> Documentation caveat: UCI's later South German Credit dataset notes errors in the original coding information. FinSight preserves the requested original Statlog codes and labels them from the official Statlog page; sensitive combined attributes should not be treated as clean demographic fields.", "", "| # | FinSight name | Type | Unit | Official meaning |", "|---:|---|---|---|---|"]
    for index, spec in enumerate(COLUMNS, 1):
        dictionary.append(f"| {index} | `{spec.name}` | {spec.kind} | {spec.unit or '—'} | {spec.description} |")
    dictionary += ["", "## Original categorical codes", ""]
    for column, labels in CATEGORY_LABELS.items():
        dictionary += [f"### `{column}`", "", "| Code | Official meaning |", "|---|---|"] + [f"| {code} | {meaning} |" for code, meaning in labels.items()] + [""]
    dictionary += ["## Target encoding", "", "The raw target is retained as `credit_risk`: **1 = good credit risk**, **2 = bad credit risk**. No binary remapping is performed on Day 1. For future modeling, the planned positive event is bad risk (`2`), because recall and cost for risky applicants are decision-relevant; that transformation must occur inside the modeling pipeline.", ""]
    (reports_dir / "data_dictionary.md").write_text("\n".join(dictionary), encoding="utf-8")
