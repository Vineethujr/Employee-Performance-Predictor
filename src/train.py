"""
Training Module for Employee Performance Predictor.
Instantiates, fits, and persists Scikit-learn pipelines for the approved models.
"""

import joblib
import pandas as pd
from typing import Dict, Any
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from src.config import (
    MODELS_DIR,
    RANDOM_STATE,
    MODEL_NAMES,
)
from src.preprocessor import create_pipeline


def get_model_instances() -> Dict[str, Any]:
    """
    Instantiate the 3 approved classification algorithms with explicit parameters.

    Returns:
    --------
    Dict[str, Any]: Map of model key to instantiated estimator.
    """
    models = {
        "logistic_regression": LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE,
        ),
        "decision_tree": DecisionTreeClassifier(
            max_depth=6,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=RANDOM_STATE,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=120,
            max_depth=8,
            min_samples_split=8,
            min_samples_leaf=4,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }
    return models


def train_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    models_dir: Path = MODELS_DIR,
) -> Dict[str, Pipeline]:
    """
    Train all approved models using Scikit-Learn Pipelines.

    STRICT ML ORDERING:
    The preprocessing transformers inside each pipeline are fitted
    EXCLUSIVELY on X_train. No test or validation information is exposed.

    Parameters:
    -----------
    X_train : pd.DataFrame
        Training feature matrix.
    y_train : pd.Series
        Training target labels.
    models_dir : Path
        Directory where serialized pipeline files (.joblib) are stored.

    Returns:
    --------
    Dict[str, Pipeline]: Dictionary of trained pipeline objects.
    """
    models_dir.mkdir(parents=True, exist_ok=True)
    estimators = get_model_instances()
    trained_pipelines = {}

    print("\n" + "=" * 50)
    print("Beginning Model Training Pipeline")
    print("=" * 50)

    for key, estimator in estimators.items():
        model_name = MODEL_NAMES[key]
        print(f"[*] Constructing & training pipeline for: {model_name}...")

        # Create end-to-end pipeline: Preprocessor -> Estimator
        pipeline = create_pipeline(estimator)

        # Fit strictly on training data
        pipeline.fit(X_train, y_train)

        # Persist trained artifact
        artifact_path = models_dir / f"{key}_pipeline.joblib"
        joblib.dump(pipeline, artifact_path)
        print(f"    -> Fitted and saved artifact to: {artifact_path.name}")

        trained_pipelines[key] = pipeline

    print("=" * 50)
    print(f"Successfully trained and saved {len(trained_pipelines)} model pipelines.")
    print("=" * 50 + "\n")

    return trained_pipelines


if __name__ == "__main__":
    from src.data_generator import generate_employee_data
    from src.preprocessor import split_and_save_data

    df = generate_employee_data()
    X_train, X_test, y_train, y_test = split_and_save_data(df)
    pipelines = train_models(X_train, y_train)
