"""Train, compare, select, and persist FinSight credit-risk models."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.data.load import load_german_credit
from src.data.run_day1 import PROJECT_ROOT
from src.data.validate import validate_dataset
from src.models.experiment import run_experiments


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-path", type=Path, default=PROJECT_ROOT / "data/raw/german.data")
    parser.add_argument("--reports-dir", type=Path, default=PROJECT_ROOT / "reports")
    parser.add_argument("--models-dir", type=Path, default=PROJECT_ROOT / "models")
    parser.add_argument("--outputs-dir", type=Path, default=PROJECT_ROOT / "outputs")
    args = parser.parse_args()
    frame = load_german_credit(args.raw_path)
    validation = validate_dataset(frame)
    validation.raise_for_errors()
    report = run_experiments(frame, args.reports_dir, args.models_dir, args.outputs_dir, args.raw_path)
    selected = report["selected_model"]
    result = report["models"][selected["name"]]["cost_sensitive"]
    print(f"Selected model: {selected['name']} (threshold={selected['threshold']:.2f})")
    print(f"Holdout ROC-AUC={result['roc_auc']:.4f}, recall={result['recall']:.4f}, cost={result['cost']['total_cost']}")
    print(f"Artifacts written to: {args.models_dir}")


if __name__ == "__main__":
    main()
