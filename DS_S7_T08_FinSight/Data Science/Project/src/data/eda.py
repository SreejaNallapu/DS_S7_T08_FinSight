"""Reusable exploratory statistics for the German Credit dataset."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data.schema import CATEGORY_LABELS, CATEGORICAL_COLUMNS, NUMERIC_COLUMNS, TARGET, TARGET_LABELS
from src.data.validate import validate_dataset

IMPORTANT_CATEGORIES = ["checking_account_status", "savings_status", "employment_duration", "credit_history", "purpose", "housing"]
IMPORTANT_NUMERIC = ["age_years", "credit_amount", "duration_months"]


def iqr_outlier_summary(frame: pd.DataFrame) -> dict[str, dict[str, float | int]]:
    """Flag potential outliers without deleting or altering observations."""
    result = {}
    for column in NUMERIC_COLUMNS:
        q1, q3 = frame[column].quantile([0.25, 0.75])
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        count = int(((frame[column] < lower) | (frame[column] > upper)).sum())
        result[column] = {
            "q1": round(float(q1), 2),
            "q3": round(float(q3), 2),
            "lower_fence": round(float(lower), 2),
            "upper_fence": round(float(upper), 2),
            "potential_outliers": count,
        }
    return result


def build_eda_summary(frame: pd.DataFrame) -> dict:
    validation = validate_dataset(frame)
    validation.raise_for_errors()
    bad = (frame[TARGET] == 2).astype(int)
    target_counts = frame[TARGET].value_counts().sort_index()
    numeric_stats = frame[NUMERIC_COLUMNS].describe().round(2).to_dict()
    numeric_associations = {
        column: round(float(frame[column].corr(bad)), 4) for column in NUMERIC_COLUMNS
    }
    categorical_bad_rates = {}
    for column in CATEGORICAL_COLUMNS:
        grouped = (
            pd.DataFrame({column: frame[column], "is_bad_risk": bad})
            .groupby(column, observed=True)["is_bad_risk"]
            .agg(["count", "mean"])
            .sort_values("mean", ascending=False)
        )
        categorical_bad_rates[column] = {
            str(code): {
                "count": int(row["count"]),
                "bad_rate_percent": round(float(row["mean"] * 100), 2),
            }
            for code, row in grouped.iterrows()
        }
    return {
        "data_quality": {
            "rows": len(frame),
            "columns": len(frame.columns),
            "completeness_percent": round(100 * (1 - frame.isna().sum().sum() / frame.size), 2),
            "missing_values": int(frame.isna().sum().sum()),
            "duplicate_rows": int(frame.duplicated().sum()),
            "invalid_values": len(validation.errors),
            "validation_errors": list(validation.errors),
            "validation_warnings": list(validation.warnings),
        },
        "class_distribution": {
            TARGET_LABELS[int(code)]: {
                "code": int(code),
                "count": int(count),
                "percent": round(100 * int(count) / len(frame), 2),
            }
            for code, count in target_counts.items()
        },
        "numeric_summary": numeric_stats,
        "potential_outliers_iqr": iqr_outlier_summary(frame),
        "numeric_correlation_with_bad_risk": numeric_associations,
        "categorical_bad_rates": categorical_bad_rates,
    }


def _save_figure(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()


def create_eda_figures(frame: pd.DataFrame, output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    created = []
    colors = ["#2f80ed", "#eb5757"]

    counts = frame[TARGET].map(TARGET_LABELS).value_counts().reindex(["good", "bad"])
    ax = counts.plot.bar(color=colors, title="Credit Risk Class Distribution")
    ax.set(xlabel="Credit risk", ylabel="Applicants")
    ax.bar_label(ax.containers[0])
    path = output_dir / "target_distribution.png"
    _save_figure(path)
    created.append(path)

    frame[IMPORTANT_NUMERIC].hist(figsize=(12, 3.8), layout=(1, 3), bins=20, color="#2f80ed", edgecolor="white")
    plt.suptitle("Numeric Feature Distributions", y=1.01, fontsize=14)
    path = output_dir / "numeric_distributions.png"
    _save_figure(path)
    created.append(path)

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, column in zip(axes.flat, IMPORTANT_CATEGORIES):
        frame[column].value_counts().sort_index().plot.bar(ax=ax, color="#56cc9d")
        ax.set_title(column.replace("_", " "))
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=45)
    for ax in axes.flat[len(IMPORTANT_CATEGORIES):]:
        ax.axis("off")
    fig.suptitle("Categorical Feature Distributions", y=1.0, fontsize=14)
    path = output_dir / "categorical_distributions.png"
    _save_figure(path)
    created.append(path)

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for ax, column in zip(axes.flat, IMPORTANT_CATEGORIES):
        grouped = frame.assign(is_bad=frame[TARGET].eq(2)).groupby(column, observed=True)["is_bad"].agg(["mean", "count"]).sort_values("mean")
        grouped["mean"].mul(100).plot.barh(ax=ax, color="#eb5757")
        ax.set_yticklabels([f"{code} (n={int(count)})" for code, count in grouped["count"].items()])
        ax.axvline(30, color="#333333", linestyle="--", linewidth=1)
        ax.set_xlim(0, 100)
        ax.set(title=f"Bad-risk rate by {column.replace('_', ' ')}", xlabel="Bad-risk rate (%)", ylabel="")
    path = output_dir / "categorical_risk_relationships.png"
    _save_figure(path)
    created.append(path)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    labels = frame[TARGET].map(TARGET_LABELS)
    for ax, column in zip(axes, ["duration_months", "credit_amount", "age_years"]):
        groups = [frame.loc[labels == label, column] for label in ["good", "bad"]]
        ax.boxplot(groups, showfliers=True)
        ax.set_xticks([1, 2], ["good", "bad"])
        ax.set(title=column.replace("_", " "), ylabel=column.replace("_", " "))
    fig.suptitle("Important Numeric Features by Credit Risk")
    path = output_dir / "numeric_risk_relationships.png"
    _save_figure(path)
    created.append(path)
    return created


def write_eda_report(frame: pd.DataFrame, reports_dir: Path, outputs_dir: Path | None = None) -> dict:
    reports_dir.mkdir(parents=True, exist_ok=True)
    summary = build_eda_summary(frame)
    outputs_dir = outputs_dir if outputs_dir is not None else reports_dir
    outputs_dir.mkdir(parents=True, exist_ok=True)
    (outputs_dir / "eda_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    quality = summary["data_quality"]
    classes = summary["class_distribution"]
    correlations = sorted(
        summary["numeric_correlation_with_bad_risk"].items(),
        key=lambda item: abs(item[1]), reverse=True,
    )
    outliers = summary["potential_outliers_iqr"]
    lines = [
        "# Exploratory Data Analysis Summary", "",
        "All figures and statistics are computed from the validated raw UCI dataset.", "",
        "## Data quality", "",
        f"- Completeness: **{quality['completeness_percent']:.2f}%**",
        f"- Exact duplicates: **{quality['duplicate_rows']}**",
        f"- Contract validation errors: **{quality['invalid_values']}**", "",
        "## Target balance", "",
        f"- Good risk: **{classes['good']['count']} ({classes['good']['percent']:.2f}%)**",
        f"- Bad risk: **{classes['bad']['count']} ({classes['bad']['percent']:.2f}%)**",
        "- The 70/30 imbalance supports stratified splitting and metrics beyond accuracy.", "",
        "## Numeric associations with bad risk", "",
        "These are univariate Pearson correlations with `is_bad_risk`; they are descriptive, not causal.", "",
        "| Feature | Correlation |", "|---|---:|",
    ]
    lines += [f"| {name} | {value:.4f} |" for name, value in correlations]
    lines += ["", "## Numeric comparisons by risk", "", "| Risk | Age (mean years) | Amount (mean DM) | Duration (mean months) |", "|---|---:|---:|---:|"]
    for code, values in frame.groupby(TARGET)[IMPORTANT_NUMERIC].mean().iterrows():
        lines.append(f"| {TARGET_LABELS[code]} | {values['age_years']:.2f} | {values['credit_amount']:.2f} | {values['duration_months']:.2f} |")
    lines += ["", "## Important categorical relationships", "", "Sample sizes accompany rates: small groups can be unstable. Codes use the official data dictionary. Dashed chart lines show the overall 30% bad-risk rate.", "", "| Feature | Code | Meaning | Applicants | Bad risk (%) |", "|---|---|---|---:|---:|"]
    for column in IMPORTANT_CATEGORIES:
        for code, values in summary['categorical_bad_rates'][column].items():
            lines.append(f"| {column} | {code} | {CATEGORY_LABELS[column][code]} | {values['count']} | {values['bad_rate_percent']:.2f} |")
    lines += ["", "These full-dataset descriptions are exploratory, not causal or independent holdout evaluation. No rows are deleted or thresholds tuned from these plots."]
    lines += ["", "## Potential outliers (1.5 × IQR rule)", "",
              "Flags are retained for investigation; no rows are removed automatically. The IQR rule is weak for low-cardinality count/ordinal fields (especially `dependents`, whose IQR is zero), so those flags are not evidence of invalid data.", "",
              "| Feature | Lower fence | Upper fence | Flagged rows |", "|---|---:|---:|---:|"]
    lines += [f"| {name} | {values['lower_fence']:.2f} | {values['upper_fence']:.2f} | {values['potential_outliers']} |" for name, values in outliers.items()]
    lines += ["", "## Report figures", "",
              "- `../outputs/plots/target_distribution.png`", "- `../outputs/plots/numeric_distributions.png`",
              "- `../outputs/plots/categorical_distributions.png`", "- `../outputs/plots/categorical_risk_relationships.png`",
              "- `../outputs/plots/numeric_risk_relationships.png`", ""]
    (reports_dir / "eda_report.md").write_text("\n".join(lines), encoding="utf-8")
    return summary
