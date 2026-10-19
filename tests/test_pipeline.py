"""
Unit tests for data splitting and preprocessing pipeline.
"""

import pytest
import pandas as pd
from sklearn.linear_model import LogisticRegression
from src.data_generator import generate_employee_data
from src.preprocessor import get_preprocessor, create_pipeline, split_and_save_data
from src.config import TARGET_CLASSES


@pytest.fixture
def sample_dataset():
    return generate_employee_data(n_samples=120, random_state=42, save_path=None)


def test_stratified_split_proportions(sample_dataset):
    """Verify that train/test split retains the specified test_size ratio and labels."""
    X_train, X_test, y_train, y_test = split_and_save_data(
        sample_dataset, test_size=0.25, random_state=42, save_splits=False
    )

    assert len(X_train) == 90
    assert len(X_test) == 30
    assert len(y_train) == 90
    assert len(y_test) == 30

    # Ensure all target classes exist in training and testing partitions
    assert set(y_train.unique()) == set(TARGET_CLASSES)
    assert set(y_test.unique()) == set(TARGET_CLASSES)


def test_preprocessor_transformation(sample_dataset):
    """Verify that the ColumnTransformer transforms data and handles missing values without leakage."""
    X_train, X_test, y_train, y_test = split_and_save_data(
        sample_dataset, test_size=0.2, random_state=42, save_splits=False
    )

    preprocessor = get_preprocessor()
    # Fit strictly on train
    X_train_trans = preprocessor.fit_transform(X_train)
    # Transform test without fitting
    X_test_trans = preprocessor.transform(X_test)

    assert X_train_trans.shape[0] == len(X_train)
    assert X_test_trans.shape[0] == len(X_test)
    assert X_train_trans.shape[1] == X_test_trans.shape[1]


def test_pipeline_unseen_category_handling(sample_dataset):
    """Verify that one-hot encoding handles unseen categories gracefully via handle_unknown='ignore'."""
    X_train, X_test, y_train, y_test = split_and_save_data(
        sample_dataset, test_size=0.2, random_state=42, save_splits=False
    )

    pipeline = create_pipeline(LogisticRegression(max_iter=500, random_state=42))
    pipeline.fit(X_train, y_train)

    # Introduce unseen category in test sample
    X_test_modified = X_test.copy()
    X_test_modified.iloc[0, X_test_modified.columns.get_loc("department")] = "NonExistentDept"

    preds = pipeline.predict(X_test_modified)
    assert len(preds) == len(X_test)
    assert preds[0] in TARGET_CLASSES
