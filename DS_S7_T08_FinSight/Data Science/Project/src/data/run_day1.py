"""Command-line entry point for the reproducible Day 1 data pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.data.load_data import load_data
from src.data.report import write_reports
from src.data.validate import validate_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-path", type=Path, default=PROJECT_ROOT / "data/raw/german.data")
    parser.add_argument("--reports-dir", type=Path, default=PROJECT_ROOT / "reports")
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--outputs-dir", type=Path, default=PROJECT_ROOT / "outputs")
    args = parser.parse_args()
    raw_path = args.raw_path
    frame = load_data(raw_path, force_download=args.force_download)
    result = validate_dataset(frame)
    result.raise_for_errors()
    write_reports(frame, result, args.reports_dir, args.outputs_dir)
    print(f"Day 1 checks passed: {len(frame)} rows, {len(frame.columns)} columns")
    print(f"Raw SHA-256 and source: {raw_path.with_suffix('.metadata.json')}")
    print(f"Reports written to: {args.reports_dir}")
    print(f"Machine-readable quality report: {args.outputs_dir / 'data_quality_report.json'}")


if __name__ == "__main__":
    main()
