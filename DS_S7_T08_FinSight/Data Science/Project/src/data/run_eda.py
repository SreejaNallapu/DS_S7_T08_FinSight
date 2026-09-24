"""Run validated EDA and generate report-ready artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.data.eda import create_eda_figures, write_eda_report
from src.data.load_data import load_data
from src.data.run_day1 import PROJECT_ROOT
from src.data.validate import validate_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-path", type=Path, default=PROJECT_ROOT / "data/raw/german.data")
    parser.add_argument("--reports-dir", type=Path, default=PROJECT_ROOT / "reports")
    parser.add_argument("--outputs-dir", type=Path, default=PROJECT_ROOT / "outputs")
    args = parser.parse_args()
    frame = load_data(args.raw_path)
    validation = validate_dataset(frame)
    validation.raise_for_errors()
    summary = write_eda_report(frame, args.reports_dir, args.outputs_dir)
    figures = create_eda_figures(frame, args.outputs_dir / "plots")
    print(f"EDA complete: {len(frame)} rows, {summary['data_quality']['completeness_percent']:.2f}% complete")
    print(f"Reports: {args.reports_dir / 'eda_report.md'} and {args.outputs_dir / 'eda_summary.json'}")
    print(f"Figures created: {len(figures)}")


if __name__ == "__main__":
    main()
