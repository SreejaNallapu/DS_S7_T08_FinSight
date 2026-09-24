from unittest.mock import Mock

import numpy as np
import pytest
from botocore.exceptions import ClientError

from src.cloud.s3_utils import download_file, object_exists, upload_file
from src.data.load_data import load_data, extract_features_target
from src.data.schema import COLUMN_NAMES, NUMERIC_COLUMNS, CATEGORICAL_COLUMNS
from src.data.validate import validate_dataset
from src.data.acquire import download_dataset, EXPECTED_SHA256
from src.data.load_data import DEFAULT_RAW_PATH
import io
import json


@pytest.fixture
def frame():
    return load_data()


def test_loading_schema_and_target(frame):
    assert list(frame.columns) == COLUMN_NAMES
    assert len(NUMERIC_COLUMNS) == 7
    assert len(CATEGORICAL_COLUMNS) == 13
    X, y = extract_features_target(frame)
    assert X.shape == (1000, 20)
    assert y.value_counts().to_dict() == {1: 700, 2: 300}
    X.iloc[0, 1] = 999
    assert frame.iloc[0, 1] != 999


@pytest.mark.parametrize("value", [None, 9])
def test_target_rejects_invalid_values(frame, value):
    frame.loc[0, "credit_risk"] = value
    with pytest.raises(ValueError, match="Target"):
        extract_features_target(frame)


@pytest.mark.parametrize("value", [-1, 0, 1.5, np.inf, np.nan])
def test_validation_rejects_invalid_numeric(frame, value):
    frame["age_years"] = frame["age_years"].astype(float)
    frame.loc[0, "age_years"] = value
    assert not validate_dataset(frame).is_valid


def test_validation_unknown_category_and_duplicates(frame):
    frame["purpose"] = frame["purpose"].astype(str)
    frame.loc[0, "purpose"] = "A999"
    assert any("undocumented" in e for e in validate_dataset(frame).errors)
    frame.iloc[0] = frame.iloc[1]
    assert validate_dataset(frame).warnings


def test_s3_transfers(tmp_path):
    client = Mock()
    source = tmp_path / "input.txt"
    source.write_text("test", encoding="utf-8")
    upload_file(source, "bucket", "raw/key", client=client)
    client.upload_file.assert_called_once_with(str(source), "bucket", "raw/key")
    target = tmp_path / "nested" / "output.txt"
    assert download_file("bucket", "raw/key", target, client=client) == target
    client.download_file.assert_called_once_with("bucket", "raw/key", str(target))
    assert target.parent.is_dir()
    with pytest.raises(FileNotFoundError):
        upload_file(tmp_path / "absent", "bucket", "key", client=client)


@pytest.mark.parametrize("code, expected", [("404", False), ("NoSuchKey", False), ("403", None), ("500", None)])
def test_s3_exists_errors(code, expected):
    client = Mock()
    client.head_object.side_effect = ClientError({"Error": {"Code": code}}, "HeadObject")
    if expected is None:
        with pytest.raises(ClientError):
            object_exists("bucket", "key", client=client)
    else:
        assert object_exists("bucket", "key", client=client) is expected


def test_s3_exists_success():
    client = Mock()
    assert object_exists("bucket", "key", client=client)
    client.head_object.assert_called_once_with(Bucket="bucket", Key="key")


def test_download_writes_verified_payload_and_metadata(tmp_path, monkeypatch):
    payload = DEFAULT_RAW_PATH.read_bytes()
    monkeypatch.setattr("src.data.acquire.urlopen", lambda *args, **kwargs: io.BytesIO(payload))
    destination = tmp_path / "german.data"
    assert download_dataset(destination) == destination
    assert destination.read_bytes() == payload
    assert json.loads(destination.with_suffix(".metadata.json").read_text())["sha256"] == EXPECTED_SHA256
