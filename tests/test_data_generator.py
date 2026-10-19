"""
Unit tests for data generator module.
"""

import pytest
import pandas as pd
from src.data_generator import generate_employee_data
from src.config import TARGET_CLASSES, TARGET_COLUMN, ALL_FEATURES


def test_generate_employee_data_shape():
    """Verify that the generated dataset contains the requested sample count and schema."""
    n_samples = 150
    df = generate_employee_data(n_samples=n_samples, random_state=42, save_path=None)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == n_samples
    assert TARGET_COLUMN in df.columns

    for feature in ALL_FEATURES:
        assert feature in df.columns, f"Expected feature {feature} missing from generated dataset"


def test_target_classes_validity():
    """Verify that all target classes match the approved 3-class target definitions."""
    df = generate_employee_data(n_samples=200, random_state=42, save_path=None)
    unique_classes = set(df[TARGET_COLUMN].unique())

    assert unique_classes.issubset(set(TARGET_CLASSES)), (
        f"Unexpected classes found: {unique_classes - set(TARGET_CLASSES)}"
    )
    # Ensure all 3 classes are represented
    assert len(unique_classes) == 3


def test_no_null_values_generated():
    """Verify that default generation has 0 unexpected null or NaN values."""
    df = generate_employee_data(n_samples=100, random_state=99, save_path=None)
    assert df.isnull().sum().sum() == 0


def test_reproducibility():
    """Verify that identical random seeds produce identical datasets."""
    df1 = generate_employee_data(n_samples=50, random_state=123, save_path=None)
    df2 = generate_employee_data(n_samples=50, random_state=123, save_path=None)

    pd.testing.assert_frame_equal(df1, df2)
