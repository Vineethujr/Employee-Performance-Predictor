"""
Preprocessor Module for Employee Performance Predictor.
Builds leak-free Scikit-Learn ColumnTransformer and Pipeline objects.
"""

import pandas as pd
from typing import Tuple, Dict, Any
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator

from src.config import (
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    TARGET_COLUMN,
    ID_COLUMN,
    TRAIN_DATA_PATH,
    TEST_DATA_PATH,
    RANDOM_STATE,
)


def get_preprocessor() -> ColumnTransformer:
    """
    Construct an isolated ColumnTransformer for numerical and categorical features.

    Numerical Pipeline:
      - SimpleImputer (median strategy)
      - StandardScaler (zero mean, unit variance)

    Categorical Pipeline:
      - SimpleImputer (most_frequent strategy)
      - OneHotEncoder (drops no levels, ignores unknown categories during test/inference)

    Returns:
    --------
    ColumnTransformer: Configured preprocessor object.
    """
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, NUMERICAL_FEATURES),
            ("cat", cat_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",  # Drops identifier columns like employee_id
    )

    return preprocessor


def create_pipeline(model: BaseEstimator) -> Pipeline:
    """
    Combine the preprocessor and a classification model into a unified Pipeline.

    Parameters:
    -----------
    model : BaseEstimator
        Scikit-learn classification estimator.

    Returns:
    --------
    Pipeline: End-to-end pipeline handling preprocessing and classification.
    """
    preprocessor = get_preprocessor()
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", model),
    ])
    return pipeline


def split_and_save_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = RANDOM_STATE,
    save_splits: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Perform a stratified train-test split to ensure class balance across splits.

    Crucial ML Rule:
    Splitting occurs BEFORE any transformation fitting to prevent data leakage.

    Parameters:
    -----------
    df : pd.DataFrame
        Complete raw dataset.
    test_size : float
        Proportion of dataset to allocate to the test split (default: 0.2).
    random_state : int
        Seed for deterministic splitting.
    save_splits : bool
        Whether to persist train.csv and test.csv to disk.

    Returns:
    --------
    Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        X_train, X_test, y_train, y_test
    """
    features = [col for col in df.columns if col not in [TARGET_COLUMN]]
    X = df[features]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    if save_splits:
        train_df = pd.concat([X_train, y_train], axis=1)
        test_df = pd.concat([X_test, y_test], axis=1)

        TRAIN_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        train_df.to_csv(TRAIN_DATA_PATH, index=False)
        test_df.to_csv(TEST_DATA_PATH, index=False)

        print(f"[Preprocessor] Saved Train Split: {len(train_df)} rows to {TRAIN_DATA_PATH}")
        print(f"[Preprocessor] Saved Test Split:  {len(test_df)} rows to {TEST_DATA_PATH}")

    return X_train, X_test, y_train, y_test
