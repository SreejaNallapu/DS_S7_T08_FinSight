from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

from src.data.load import load_german_credit
from src.data.schema import CATEGORICAL_COLUMNS, NUMERIC_COLUMNS, TARGET
from src.features.preprocessing import (
    build_model_pipeline,
    build_preprocessor,
    make_train_test_split,
    split_features_target,
)

RAW_PATH = Path("data/raw/german.data")


def _frame():
    return load_german_credit(RAW_PATH)


def test_feature_target_separation_and_binary_mapping():
    X, y = split_features_target(_frame())
    assert TARGET not in X.columns
    assert X.shape == (1000, 20)
    assert y.name == "is_bad_risk"
    assert y.value_counts().to_dict() == {0: 700, 1: 300}


def test_stratified_split_is_reproducible_and_disjoint():
    first = make_train_test_split(_frame(), random_state=42)
    second = make_train_test_split(_frame(), random_state=42)
    assert first.X_train.index.equals(second.X_train.index)
    assert len(first.X_train) == 800
    assert len(first.X_test) == 200
    assert set(first.X_train.index).isdisjoint(first.X_test.index)
    assert first.y_train.mean() == 0.3
    assert first.y_test.mean() == 0.3


def test_preprocessor_fits_training_data_and_handles_unseen_category():
    split = make_train_test_split(_frame())
    preprocessor = build_preprocessor(scale_numeric=True)
    transformed_train = preprocessor.fit_transform(split.X_train)
    inference_row = split.X_test.iloc[[0]].copy()
    inference_row["purpose"] = "A999"
    transformed_row = preprocessor.transform(inference_row)
    assert transformed_train.shape[0] == 800
    assert transformed_train.shape[1] == len(preprocessor.get_feature_names_out())
    assert transformed_row.shape == (1, transformed_train.shape[1])
    assert np.isfinite(transformed_train).all()


def test_numeric_scaling_can_be_disabled_for_tree_models():
    preprocessor = build_preprocessor(scale_numeric=False)
    numeric_steps = preprocessor.transformers[0][1].named_steps
    assert "scaler" not in numeric_steps
    assert set(NUMERIC_COLUMNS).isdisjoint(CATEGORICAL_COLUMNS)


def test_model_pipeline_reuses_preprocessing_at_inference():
    split = make_train_test_split(_frame())
    pipeline = build_model_pipeline(LogisticRegression(max_iter=1000))
    pipeline.fit(split.X_train, split.y_train)
    probabilities = pipeline.predict_proba(split.X_test.iloc[:3])
    assert probabilities.shape == (3, 2)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
