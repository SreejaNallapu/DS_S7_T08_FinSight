import sqlite3

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from src.cloud.athena_queries import create_table_sql, example_queries
from src.data.eda import build_eda_summary
from src.data.load_data import load_data
from src.data.preprocess import build_analytics_dataset, prepare_data, write_analytics_dataset
from src.data.schema import COLUMN_NAMES, NUMERIC_COLUMNS
from src.features.preprocessing import split_features_target


@pytest.fixture
def frame():
    return load_data()


def test_day2_split_and_target(frame):
    prepared = prepare_data(frame)
    repeated = prepare_data(frame)
    assert isinstance(prepared.pipeline, Pipeline)
    assert prepared.split.y_train.value_counts().to_dict() == {0: 560, 1: 240}
    assert prepared.split.y_test.value_counts().to_dict() == {0: 140, 1: 60}
    assert set(prepared.split.X_train.index).isdisjoint(prepared.split.X_test.index)
    assert prepared.split.X_train.index.equals(repeated.split.X_train.index)
    assert prepared.X_train.shape[0] == 800 and prepared.X_test.shape[0] == 200
    assert prepared.X_train.shape[1] == prepared.X_test.shape[1]
    assert np.isfinite(prepared.X_train).all() and np.isfinite(prepared.X_test).all()


def test_training_only_fit_and_unseen_inference(frame, tmp_path):
    baseline = prepare_data(frame)
    # Deliberately alter only test rows: train-fitted statistics/categories must not change.
    changed = frame.copy()
    changed['purpose'] = changed['purpose'].astype(object)
    changed.loc[baseline.split.X_test.index, 'credit_amount'] = 1000000
    changed.loc[baseline.split.X_test.index, 'purpose'] = 'UNSEEN_TEST_ONLY'
    prepared = prepare_data(changed, scale_numeric=True)
    transformer = prepared.pipeline.named_steps['preprocessor']
    numeric = transformer.named_transformers_['numeric']
    assert np.allclose(numeric.named_steps['scaler'].mean_, prepared.split.X_train[NUMERIC_COLUMNS].mean())
    assert np.allclose(numeric.named_steps['imputer'].statistics_, prepared.split.X_train[NUMERIC_COLUMNS].median())
    encoder = transformer.named_transformers_['categorical'].named_steps['onehot']
    assert all('UNSEEN_TEST_ONLY' not in categories for categories in encoder.categories_)
    assert np.isfinite(prepared.X_test).all()
    artifact = tmp_path / 'preprocessing_only.joblib'
    joblib.dump(prepared.pipeline, artifact)
    assert np.allclose(joblib.load(artifact).transform(prepared.split.X_test), prepared.X_test)


@pytest.mark.parametrize('invalid', [None, 3])
def test_invalid_target_is_rejected(frame, invalid):
    frame.loc[0, 'credit_risk'] = invalid
    with pytest.raises(ValueError, match='[Tt]arget'):
        split_features_target(frame)


def test_analytics_has_no_fitted_transformations(frame):
    analytics = build_analytics_dataset(frame)
    assert analytics.shape == (1000, 24)
    pd.testing.assert_frame_equal(analytics[COLUMN_NAMES], frame)
    assert analytics.applicant_id.tolist() == list(range(1, 1001))
    assert analytics.groupby('credit_risk_label').is_bad_risk.mean().to_dict() == {'bad': 1.0, 'good': 0.0}
    with pytest.raises(ValueError, match='raw schema'):
        prepare_data(analytics)


def test_export_round_trip_and_raw_protection(frame, tmp_path, monkeypatch):
    monkeypatch.setattr('src.data.preprocess.PROJECT_ROOT', tmp_path)
    raw = tmp_path / 'data/raw/german.data'
    raw.parent.mkdir(parents=True)
    raw.write_text('original', encoding='utf-8')
    with pytest.raises(ValueError, match='raw files are protected'):
        write_analytics_dataset(frame, raw)
    assert raw.read_text() == 'original'
    output = write_analytics_dataset(frame, tmp_path / 'data/processed/german_credit_analytics.csv')
    exported = pd.read_csv(output)
    assert list(exported.columns) == list(build_analytics_dataset(frame).columns)
    assert exported.shape == (1000, 24)
    assert exported.credit_risk.value_counts().to_dict() == {1: 700, 2: 300}
    assert exported.credit_amount.sum() == frame.credit_amount.sum()
    first = output.read_bytes()
    write_analytics_dataset(frame, output)
    assert output.read_bytes() == first


def test_sql_queries_against_local_analytics(frame):
    # SQLite verifies these portable SELECT results; it is not an Athena integration test.
    with sqlite3.connect(':memory:') as connection:
        build_analytics_dataset(frame).to_sql('analytics', connection, index=False)
        queries = example_queries('main', 'analytics')
        assert connection.execute(queries['applicant_count']).fetchone()[0] == 1000
        assert connection.execute(queries['risk_counts']).fetchall() == [(1, 'good', 700), (2, 'bad', 300)]
        assert connection.execute(queries['average_credit_amount']).fetchone()[0] == pytest.approx(frame.credit_amount.mean())
        assert connection.execute(queries['average_loan_duration']).fetchone()[0] == pytest.approx(frame.duration_months.mean())
        grouped = connection.execute(queries['grouped_credit_risk']).fetchall()
        assert sum(row[2] for row in grouped) == 1000
        assert sum(row[3] for row in grouped) == 300


def test_athena_schema_and_config(frame):
    ddl = create_table_sql('s3://example-bucket/finsight/processed/german_credit_analytics/', 'demo', 'analytics')
    assert '`demo`.`analytics`' in ddl
    assert "'skip.header.line.count'='1'" in ddl
    assert 'OpenCSVSerde' in ddl
    for column in build_analytics_dataset(frame).columns:
        assert f'`{column}`' in ddl
    with pytest.raises(ValueError):
        example_queries('demo; DROP TABLE foo', 'analytics')
    with pytest.raises(ValueError):
        create_table_sql("s3://bucket/folder/'; DROP TABLE foo")


def test_eda_category_totals(frame):
    summary = build_eda_summary(frame)
    assert summary['class_distribution']['bad']['count'] == 300
    for groups in summary['categorical_bad_rates'].values():
        assert sum(item['count'] for item in groups.values()) == 1000
