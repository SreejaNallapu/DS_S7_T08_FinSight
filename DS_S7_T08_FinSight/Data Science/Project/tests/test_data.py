from pathlib import Path

import pandas as pd
import pytest

from src.data.acquire import EXPECTED_SHA256, download_dataset, sha256
from src.data.load import load_german_credit
from src.data.schema import CATEGORY_LABELS, COLUMN_NAMES, NUMERIC_COLUMNS
from src.data.validate import validate_dataset


RAW_PATH = Path("data/raw/german.data")


@pytest.mark.skipif(not RAW_PATH.exists(), reason="run the Day 1 pipeline to cache raw data")
def test_cached_dataset_matches_contract():
    frame = load_german_credit(RAW_PATH)
    result = validate_dataset(frame)
    assert result.is_valid, result.errors
    assert frame.shape == (1000, 21)


def test_validation_rejects_wrong_shape_and_target():
    row = {name: 1 for name in COLUMN_NAMES}
    for column, labels in CATEGORY_LABELS.items():
        row[column] = next(iter(labels))
    frame = pd.DataFrame([row])
    frame.loc[0, "credit_risk"] = 9
    result = validate_dataset(frame)
    assert not result.is_valid
    assert any("expected 1000 rows" in error for error in result.errors)
    assert any("target values" in error for error in result.errors)


def test_validation_reports_missing_target_without_crashing():
    frame = pd.DataFrame(columns=COLUMN_NAMES).drop(columns="credit_risk")
    result = validate_dataset(frame)
    assert not result.is_valid
    assert any("21-column schema" in error for error in result.errors)


def test_validation_accepts_documented_synthetic_contract():
    row = {name: 1 for name in COLUMN_NAMES}
    for column, labels in CATEGORY_LABELS.items():
        row[column] = next(iter(labels))
    rows = []
    for index in range(1000):
        item = row.copy()
        item["duration_months"] = index + 1
        item["credit_risk"] = 1 if index < 700 else 2
        rows.append(item)
    frame = pd.DataFrame(rows)
    for column in NUMERIC_COLUMNS + ["credit_risk"]:
        frame[column] = pd.to_numeric(frame[column])
    result = validate_dataset(frame)
    assert result.is_valid, result.errors


def test_loader_rejects_wrong_column_count(tmp_path):
    bad_file = tmp_path / "bad.data"
    bad_file.write_text("A11 6 A34\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Expected 21"):
        load_german_credit(bad_file)


def test_cached_file_checksum_is_pinned():
    if RAW_PATH.exists():
        assert sha256(RAW_PATH) == EXPECTED_SHA256


def test_acquisition_rejects_corrupt_cache(tmp_path):
    cached = tmp_path / "german.data"
    cached.write_text("not the dataset\n", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum mismatch"):
        download_dataset(cached)
