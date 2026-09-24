"""Model experimentation, cost-sensitive evaluation, and reporting."""

from __future__ import annotations

import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    make_scorer,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
)
from sklearn.tree import DecisionTreeClassifier

from src.features.preprocessing import (
    NON_PREDICTIVE_COLUMNS,
    build_model_pipeline,
    make_train_test_split,
)

RANDOM_STATE = 42
FALSE_NEGATIVE_COST = 5
FALSE_POSITIVE_COST = 1
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
SCORING = {
    "accuracy": "accuracy",
    "precision": make_scorer(precision_score, pos_label=1, zero_division=0),
    "recall": make_scorer(recall_score, pos_label=1, zero_division=0),
    "f1": make_scorer(f1_score, pos_label=1, zero_division=0),
    "roc_auc": "roc_auc",
}


def candidate_pipelines() -> dict[str, object]:
    return {
        "logistic_regression": build_model_pipeline(
            LogisticRegression(max_iter=2000, random_state=RANDOM_STATE), True
        ),
        "decision_tree": build_model_pipeline(
            DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE), False
        ),
        "random_forest": build_model_pipeline(
            RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, n_jobs=1), False
        ),
        "gradient_boosting": build_model_pipeline(
            GradientBoostingClassifier(random_state=RANDOM_STATE), False
        ),
    }


def tuning_spaces() -> dict[str, dict[str, list]]:
    return {
        "logistic_regression": {
            "model__C": [0.1, 0.5, 1.0, 2.0],
            "model__class_weight": [None, "balanced"],
        },
        "random_forest": {
            "model__n_estimators": [200, 400],
            "model__max_depth": [None, 5, 8],
            "model__min_samples_leaf": [1, 2, 4],
            "model__max_features": ["sqrt", 0.7],
            "model__class_weight": [None, "balanced"],
        },
        "gradient_boosting": {
            "model__n_estimators": [75, 125, 200],
            "model__learning_rate": [0.03, 0.05, 0.1],
            "model__max_depth": [1, 2, 3],
            "model__min_samples_leaf": [5, 10, 20],
            "model__subsample": [0.8, 1.0],
        },
    }


def misclassification_cost(y_true, y_pred) -> dict[str, int | float]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    total = FALSE_NEGATIVE_COST * int(fn) + FALSE_POSITIVE_COST * int(fp)
    return {
        "false_positives_good_as_bad": int(fp),
        "false_negatives_bad_as_good": int(fn),
        "true_negatives": int(tn),
        "true_positives": int(tp),
        "total_cost": int(total),
        "average_cost_per_applicant": round(total / len(y_true), 4),
    }


def metric_row(y_true, probabilities, threshold: float) -> dict:
    predictions = (np.asarray(probabilities) >= threshold).astype(int)
    matrix = confusion_matrix(y_true, predictions, labels=[0, 1])
    return {
        "threshold": round(float(threshold), 4),
        "accuracy": round(accuracy_score(y_true, predictions), 4),
        "precision": round(precision_score(y_true, predictions, zero_division=0), 4),
        "recall": round(recall_score(y_true, predictions, zero_division=0), 4),
        "f1": round(f1_score(y_true, predictions, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_true, probabilities), 4),
        "confusion_matrix": matrix.tolist(),
        "cost": misclassification_cost(y_true, predictions),
    }


def choose_cost_threshold(y_true, probabilities) -> tuple[float, dict]:
    candidates = np.round(np.arange(0.10, 0.91, 0.01), 2)
    rows = [metric_row(y_true, probabilities, threshold) for threshold in candidates]
    best = min(
        rows,
        key=lambda row: (
            row["cost"]["total_cost"],
            -row["recall"],
            -row["f1"],
        ),
    )
    return float(best["threshold"]), best


def summarize_cv(pipeline, X_train, y_train) -> dict:
    scores = cross_validate(
        pipeline, X_train, y_train, cv=CV, scoring=SCORING, n_jobs=1
    )
    summary = {}
    for metric in SCORING:
        values = scores[f"test_{metric}"]
        summary[metric] = {
            "mean": round(float(values.mean()), 4),
            "std": round(float(values.std()), 4),
            "fold_values": [round(float(value), 4) for value in values],
        }
    return summary


def tune_candidates(pipelines, X_train, y_train) -> tuple[dict, dict]:
    tuned = dict(pipelines)
    tuning_results = {}
    for name, space in tuning_spaces().items():
        search = RandomizedSearchCV(
            estimator=pipelines[name],
            param_distributions=space,
            n_iter=8,
            scoring="roc_auc",
            cv=CV,
            random_state=RANDOM_STATE,
            n_jobs=1,
            refit=True,
            return_train_score=False,
        )
        search.fit(X_train, y_train)
        tuned[name] = search.best_estimator_
        tuning_results[name] = {
            "best_cv_roc_auc": round(float(search.best_score_), 4),
            "best_params": search.best_params_,
            "candidates_evaluated": len(search.cv_results_["params"]),
        }
    return tuned, tuning_results


def feature_importance(pipeline, top_n: int = 20) -> list[dict]:
    names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    model = pipeline.named_steps["model"]
    if hasattr(model, "feature_importances_"):
        values = model.feature_importances_
        importance_type = "impurity_importance"
    elif hasattr(model, "coef_"):
        values = np.abs(model.coef_[0])
        importance_type = "absolute_coefficient"
    else:
        return []
    order = np.argsort(values)[::-1][:top_n]
    return [
        {
            "feature": str(names[index]),
            "importance": round(float(values[index]), 6),
            "type": importance_type,
        }
        for index in order
    ]


def create_plots(evaluations, fitted, split, output_dir: Path, selected_name: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    names = list(evaluations)
    labels = [name.replace("_", " ").title() for name in names]
    metric_names = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    x = np.arange(len(names))
    width = 0.15
    fig, ax = plt.subplots(figsize=(11, 6))
    for index, metric in enumerate(metric_names):
        values = [evaluations[name]["cost_sensitive"][metric] for name in names]
        ax.bar(x + (index - 2) * width, values, width, label=metric.replace("_", " ").title())
    ax.set_xticks(x, labels, rotation=12)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Holdout score")
    ax.set_title("Model Comparison at Training-Selected Cost Thresholds")
    ax.legend(ncols=3, fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / "model_comparison.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 6))
    for name, pipeline in fitted.items():
        probability = pipeline.predict_proba(split.X_test)[:, 1]
        fpr, tpr, _ = roc_curve(split.y_test, probability)
        plt.plot(fpr, tpr, label=f"{name} (AUC={evaluations[name]['default']['roc_auc']:.3f})")
    plt.plot([0, 1], [0, 1], "--", color="grey")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("Holdout ROC Curves")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / "model_roc_curves.png", dpi=160)
    plt.close()

    selected = evaluations[selected_name]["cost_sensitive"]
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(
        confusion_matrix=np.asarray(selected["confusion_matrix"]),
        display_labels=["good", "bad"],
    ).plot(ax=ax, colorbar=False)
    ax.set_title(
        f"Final Model Confusion Matrix: {selected_name.replace('_', ' ').title()}\n"
        f"threshold={selected['threshold']:.2f}"
    )
    plt.tight_layout()
    plt.savefig(output_dir / "final_model_confusion_matrix.png", dpi=160)
    plt.close()

    importance = evaluations[selected_name]["feature_importance"]
    if importance:
        labels = [row["feature"] for row in reversed(importance)]
        values = [row["importance"] for row in reversed(importance)]
        plt.figure(figsize=(9, 7))
        plt.barh(labels, values, color="#2f80ed")
        plt.xlabel(importance[0]["type"].replace("_", " "))
        plt.title(f"Top Features: {selected_name}")
        plt.tight_layout()
        plt.savefig(output_dir / "final_model_feature_importance.png", dpi=160)
        plt.close()


def run_experiments(
    frame: pd.DataFrame,
    reports_dir: Path,
    models_dir: Path,
    outputs_dir: Path,
    raw_path: Path,
) -> dict:
    split = make_train_test_split(frame, test_size=0.2, random_state=RANDOM_STATE)
    leaked = NON_PREDICTIVE_COLUMNS.intersection(split.X_train.columns)
    if leaked:
        raise ValueError(f"Refusing to train with leaked/helper predictors: {sorted(leaked)}")
    pipelines = candidate_pipelines()
    tuned, tuning = tune_candidates(pipelines, split.X_train, split.y_train)
    evaluations, fitted = {}, {}
    for name, pipeline in tuned.items():
        cv_summary = summarize_cv(pipeline, split.X_train, split.y_train)
        oof_probability = cross_val_predict(
            pipeline, split.X_train, split.y_train, cv=CV,
            method="predict_proba", n_jobs=1,
        )[:, 1]
        threshold, training_threshold_result = choose_cost_threshold(
            split.y_train, oof_probability
        )
        model = clone(pipeline).fit(split.X_train, split.y_train)
        classes = model.named_steps["model"].classes_
        if set(classes) != {0, 1}:
            raise ValueError(f"Expected fitted target classes {{0, 1}}, found {set(classes)}")
        bad_risk_index = int(np.flatnonzero(classes == 1)[0])
        probability = model.predict_proba(split.X_test)[:, bad_risk_index]
        evaluations[name] = {
            "cv": cv_summary,
            "default": metric_row(split.y_test, probability, 0.5),
            "cost_sensitive": metric_row(split.y_test, probability, threshold),
            "threshold_selected_from_training_oof": training_threshold_result,
            "feature_importance": feature_importance(model),
        }
        fitted[name] = model

    # Choose from training evidence only; the holdout remains evaluation-only.
    selected_name = min(
        evaluations,
        key=lambda name: (
            evaluations[name]["threshold_selected_from_training_oof"]["cost"]["average_cost_per_applicant"],
            -evaluations[name]["cv"]["roc_auc"]["mean"],
            -evaluations[name]["threshold_selected_from_training_oof"]["recall"],
            -evaluations[name]["threshold_selected_from_training_oof"]["f1"],
            evaluations[name]["cv"]["roc_auc"]["std"],
        ),
    )
    selected_threshold = evaluations[selected_name]["cost_sensitive"]["threshold"]
    trained_at = datetime.now(timezone.utc).isoformat()
    raw_path = Path(raw_path)
    raw_sha256 = hashlib.sha256(raw_path.read_bytes()).hexdigest()
    feature_columns = list(split.X_train.columns)
    target_mapping = {
        "original": {"1": "good credit risk", "2": "bad credit risk"},
        "training": {"0": "good credit risk", "1": "bad credit risk"},
        "positive_class": 1,
        "probability_output": "bad_risk_probability is predict_proba for training class 1",
    }
    models_dir.mkdir(parents=True, exist_ok=True)
    artifact = {
        "pipeline": fitted[selected_name],
        "threshold": selected_threshold,
        "model_name": selected_name,
        "target_definition": "training 1=bad credit risk, training 0=good credit risk",
        "target_mapping": target_mapping,
        "probability_class": 1,
        "feature_columns": feature_columns,
        "trained_at_utc": trained_at,
        "raw_dataset_sha256": raw_sha256,
    }
    pipeline_path = models_dir / "finsight_pipeline.joblib"
    joblib.dump(artifact, pipeline_path)
    # Refresh the historical filename too so no stale artifact remains available.
    joblib.dump(artifact, models_dir / "final_model.joblib")
    joblib.dump(
        fitted[selected_name].named_steps["preprocessor"],
        models_dir / "fitted_preprocessor.joblib",
    )
    report = {
        "experiment_design": {
            "random_state": RANDOM_STATE,
            "train_rows": len(split.X_train),
            "test_rows": len(split.X_test),
            "cross_validation": "5-fold StratifiedKFold on training data",
            "positive_class": "training class 1 = bad credit risk",
            "probability_column": "predict_proba column whose classifier class label is 1",
            "cost_formula": "5 * bad predicted good (FN) + 1 * good predicted bad (FP)",
            "threshold_selection": "minimum out-of-fold training cost; ties favor recall then F1",
            "selection_data": "training cross-validation and out-of-fold predictions only; holdout excluded",
        },
        "dataset": {
            "source": "data/raw/german.data",
            "sha256": raw_sha256,
            "rows": len(frame),
            "raw_columns": len(frame.columns),
        },
        "features": {
            "used": feature_columns,
            "excluded": sorted(NON_PREDICTIVE_COLUMNS),
        },
        "target_mapping": target_mapping,
        "trained_at_utc": trained_at,
        "tuning": tuning,
        "models": evaluations,
        "selected_model": {
            "name": selected_name,
            "threshold": selected_threshold,
            "rationale": "Selected from training evidence using lowest out-of-fold 5:1 misclassification cost, then mean CV ROC-AUC, bad-risk recall, F1, and ROC-AUC stability. Holdout metrics were not used for selection; interpretability and deployment practicality were reviewed in the report.",
        },
    }
    reports_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir = outputs_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = metrics_dir / "model_metrics.json"
    metrics_path.write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    metadata = {
        "final_model_name": selected_name,
        "trained_at_utc": trained_at,
        "artifact": pipeline_path.name,
        "raw_dataset": "data/raw/german.data",
        "raw_dataset_sha256": raw_sha256,
        "feature_columns": feature_columns,
        "excluded_columns": sorted(NON_PREDICTIVE_COLUMNS),
        "target_mapping": target_mapping,
        "split": report["experiment_design"],
        "selected_threshold": selected_threshold,
        "holdout_metrics": evaluations[selected_name]["cost_sensitive"],
        "cross_validation": evaluations[selected_name]["cv"],
    }
    (models_dir / "model_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    create_plots(evaluations, fitted, split, outputs_dir / "plots", selected_name)
    write_model_report(report, reports_dir / "model_evaluation.md")
    return report


def write_model_report(report: dict, path: Path) -> None:
    design = report["experiment_design"]
    dataset = report["dataset"]
    features = report["features"]
    mapping = report["target_mapping"]
    lines = [
        "# Day 3 Model Evaluation", "",
        f"Generated from the current validated dataset and pipeline at `{report['trained_at_utc']}`. All results below were computed by the training command.", "",
        "## Dataset and predictors", "",
        f"- Dataset: `{dataset['source']}` ({dataset['rows']} rows, {dataset['raw_columns']} columns)",
        f"- SHA-256: `{dataset['sha256']}`",
        f"- Predictors used: {len(features['used'])} original applicant features",
        f"- Excluded from predictors: `{', '.join(features['excluded'])}`",
        "- The analytics CSV is not used as a model matrix; identifiers and target-derived analytics fields cannot enter `X`.", "",
        "## Target and probability convention", "",
        f"- Original target: `credit_risk` (`1 = {mapping['original']['1']}`, `2 = {mapping['original']['2']}`).",
        f"- Training target: `0 = {mapping['training']['0']}`, `1 = {mapping['training']['1']}`.",
        "- Bad risk is explicitly training class `1` for precision, recall, F1, ROC-AUC, thresholding, and cost calculations.",
        "- `bad_risk_probability` is the `predict_proba` column whose fitted classifier class label is `1`; inference validates this mapping.", "",
        "## Methodology", "",
        f"- Split: {design['train_rows']} training rows and {design['test_rows']} untouched holdout rows, stratified with `random_state={design['random_state']}`.",
        f"- Cross-validation: {design['cross_validation']}.",
        "- All imputation, scaling, one-hot encoding, tuning, and threshold selection are fitted inside sklearn pipelines on training folds only.",
        "- Models: Logistic Regression, Decision Tree, Random Forest, and Gradient Boosting only.",
        "- Logistic Regression, Random Forest, and Gradient Boosting receive an eight-candidate randomized search; the shallow Decision Tree remains an untuned interpretable baseline.",
        f"- Cost: `{design['cost_formula']}`.",
        "- Each decision threshold is chosen from out-of-fold training probabilities. The holdout is used once for final reporting, not model or threshold selection.", "",
        "## Holdout comparison at training-selected cost thresholds", "",
        "| Model | Threshold | Accuracy | Precision | Recall | F1 | ROC-AUC | Cost |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, values in report["models"].items():
        row = values["cost_sensitive"]
        lines.append(
            f"| {name} | {row['threshold']:.2f} | {row['accuracy']:.4f} | "
            f"{row['precision']:.4f} | {row['recall']:.4f} | {row['f1']:.4f} | "
            f"{row['roc_auc']:.4f} | {row['cost']['total_cost']} |"
        )
    lines += ["", "The precision, recall, and F1 values above all treat bad risk as the positive class. Confusion matrices use rows as actual and columns as predicted, ordered `[good, bad]`.", "",
              "## Five-fold cross-validation summary", "",
              "Cross-validation accuracy, precision, recall, and F1 use each fitted classifier's default prediction threshold; ROC-AUC is threshold-independent. Cost-sensitive thresholds are selected separately from the complete set of out-of-fold training probabilities.", "",
              "| Model | Accuracy mean | Precision mean | Recall mean | F1 mean | ROC-AUC mean | ROC-AUC std |", "|---|---:|---:|---:|---:|---:|---:|"]
    for name, values in report["models"].items():
        cv = values["cv"]
        lines.append(f"| {name} | {cv['accuracy']['mean']:.4f} | {cv['precision']['mean']:.4f} | {cv['recall']['mean']:.4f} | {cv['f1']['mean']:.4f} | {cv['roc_auc']['mean']:.4f} | {cv['roc_auc']['std']:.4f} |")
    lines += ["", "## Lightweight tuning", ""]
    for name, values in report["tuning"].items():
        lines.append(
            f"- **{name}**: {values['candidates_evaluated']} candidates; best tuning CV ROC-AUC "
            f"{values['best_cv_roc_auc']:.4f}; parameters `{json.dumps(values['best_params'], sort_keys=True)}`."
        )
    selected = report["selected_model"]
    selected_values = report["models"][selected["name"]]
    selected_holdout = selected_values["cost_sensitive"]
    selected_oof = selected_values["threshold_selected_from_training_oof"]
    lines += [
        "", "## Final selection", "",
        f"Selected: **{selected['name']}**, threshold **{selected['threshold']:.2f}**.", "",
        selected["rationale"], "",
        f"Its out-of-fold training cost was **{selected_oof['cost']['total_cost']}**, with bad-risk recall **{selected_oof['recall']:.4f}** and F1 **{selected_oof['f1']:.4f}**. Its untouched holdout cost was **{selected_holdout['cost']['total_cost']}**.",
        "The 5:1 German Credit error cost makes missing genuinely bad-risk applicants more costly than incorrectly flagging good-risk applicants. Accuracy alone therefore does not determine the selected model.",
        "The complete fitted preprocessing-and-classifier pipeline is persisted, so inference applies exactly the training transformations.", "",
        "## Limitations", "",
        "- The dataset has only 1,000 historical applicants; subgroup and holdout estimates can be unstable.",
        "- Threshold and model selection reflect the documented 5:1 cost assumption, which may differ from a real lender's operational costs.",
        "- The data contains sensitive or proxy attributes; predictive performance does not establish fairness, causality, or suitability for automated lending decisions.",
        "- Results are specific to this dataset version, split seed, feature definitions, and library versions.",
        "- FinSight output is decision support and requires human review; it is not an approval or denial decision.", "",
        "## Generated artifacts", "",
        "- `models/finsight_pipeline.joblib`",
        "- `models/model_metadata.json`",
        "- `outputs/metrics/model_metrics.json`",
        "- `outputs/plots/model_comparison.png`",
        "- `outputs/plots/model_roc_curves.png`",
        "- `outputs/plots/final_model_confusion_matrix.png`",
        "- `outputs/plots/final_model_feature_importance.png`", "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
