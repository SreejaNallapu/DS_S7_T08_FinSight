"""Dataset contract checks that fail loudly when raw data drift or corrupt."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import numpy as np

from src.data.schema import CATEGORY_LABELS, COLUMN_NAMES, NUMERIC_COLUMNS, TARGET, TARGET_LABELS


@dataclass(frozen=True)
class ValidationResult:
    errors: tuple[str, ...]
    warnings: tuple[str, ...]

    @property
    def is_valid(self) -> bool:
        return not self.errors

    def raise_for_errors(self) -> None:
        if self.errors:
            raise ValueError("Dataset validation failed:\n- " + "\n- ".join(self.errors))


def validate_dataset(frame: pd.DataFrame) -> ValidationResult:
    errors: list[str] = []
    warnings: list[str] = []
    if list(frame.columns) != COLUMN_NAMES:
        errors.append("columns do not match the expected 21-column schema")
    if len(frame) != 1000:
        errors.append(f"expected 1000 rows, found {len(frame)}")
    missing = int(frame.isna().sum().sum())
    if missing:
        errors.append(f"found {missing} missing values")
    targets = set(frame[TARGET].dropna().unique()) if TARGET in frame else set()
    if TARGET in frame and targets != set(TARGET_LABELS):
        errors.append(f"target values must be {set(TARGET_LABELS)}, found {targets}")
    for column, labels in CATEGORY_LABELS.items():
        if column in frame:
            unknown = set(frame[column].dropna().astype(str).unique()) - set(labels)
            if unknown:
                errors.append(f"{column} has undocumented codes: {sorted(unknown)}")
    for column in NUMERIC_COLUMNS:
        if column in frame and not pd.api.types.is_numeric_dtype(frame[column]):
            errors.append(f"{column} is not numeric")
        elif column in frame:
            values = frame[column].dropna()
            if not np.isfinite(values).all():
                errors.append(f"{column} contains non-finite values")
            elif ((values <= 0) | (values % 1 != 0)).any():
                errors.append(f"{column} must contain positive integers")
    if TARGET in frame and not pd.api.types.is_numeric_dtype(frame[TARGET]):
        errors.append(f"{TARGET} is not numeric")
    duplicates = int(frame.duplicated().sum())
    if duplicates:
        warnings.append(f"found {duplicates} exact duplicate rows")
    return ValidationResult(tuple(errors), tuple(warnings))
