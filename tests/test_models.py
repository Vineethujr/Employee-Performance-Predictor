"""
Unit tests for model training, evaluation, and hypothetical prediction inference.
"""

import pytest
import pandas as pd
from src.data_generator import generate_employee_data
from src.preprocessor import split_and_save_data
from src.train import train_models, get_model_instances
from src.evaluate import evaluate_single_pipeline, compare_and_evaluate_models
from src.predict import validate_employee_inputs, predict_employee_performance
from src.config import TARGET_CLASSES, MODEL_NAMES


@pytest.fixture(scope="module")
def trained_artifacts(tmp_path_factory):
    tmp_models_dir = tmp_path_factory.mktemp("models")
    tmp_figs_dir = tmp_path_factory.mktemp("figs")

    df = generate_employee_data(n_samples=200, random_state=42, save_path=None)
    X_train, X_test, y_train, y_test = split_and_save_data(df, test_size=0.25, random_state=42, save_splits=False)
    pipelines = train_models(X_train, y_train, models_dir=tmp_models_dir)

    return {
        "pipelines": pipelines,
        "X_test": X_test,
        "y_test": y_test,
        "tmp_figs_dir": tmp_figs_dir,
    }


def test_model_training_all_models_present(trained_artifacts):
    """Verify that all 3 approved algorithms are trained successfully."""
    pipelines = trained_artifacts["pipelines"]
    for expected_key in MODEL_NAMES.keys():
        assert expected_key in pipelines, f"Pipeline for {expected_key} was not generated."


def test_evaluation_metric_structure(trained_artifacts):
    """Verify that evaluation outputs required multi-class metrics."""
    pipelines = trained_artifacts["pipelines"]
    X_test = trained_artifacts["X_test"]
    y_test = trained_artifacts["y_test"]

    for key, pipeline in pipelines.items():
        metrics = evaluate_single_pipeline(pipeline, X_test, y_test)
        assert "accuracy" in metrics
        assert "f1_macro" in metrics
        assert "precision_macro" in metrics
        assert "recall_macro" in metrics
        assert "confusion_matrix" in metrics
        assert 0.0 <= metrics["accuracy"] <= 1.0
        assert 0.0 <= metrics["f1_macro"] <= 1.0


def test_input_validation_catches_invalid_values():
    """Verify that input validator raises ValueErrors for bad inputs."""
    invalid_sample = {
        "department": "Engineering",
        "education_level": "Bachelor's",
        "age": 105,  # Out of bounds
        "tenure_years": 4.0,
        "monthly_salary": 5000.0,
        "projects_completed": 5,
        "average_monthly_hours": 160,
        "training_hours_completed": 20,
        "last_evaluation_score": 0.75,
        "peer_review_score": 3.5,
        "satisfaction_level": 0.65,
        "absenteeism_days": 2,
        "work_life_balance_rating": 3,
        "promotion_last_3years": 0,
        "overtime_frequency": "Rare",
    }
    with pytest.raises(ValueError):
        validate_employee_inputs(invalid_sample)


def test_predict_performance_output_structure():
    """Verify that inference yields the predicted category and valid probabilities."""
    valid_sample = {
        "department": "Sales",
        "education_level": "Bachelor's",
        "age": 30,
        "tenure_years": 3.0,
        "monthly_salary": 5200.0,
        "projects_completed": 6,
        "average_monthly_hours": 165,
        "training_hours_completed": 25,
        "last_evaluation_score": 0.72,
        "peer_review_score": 3.6,
        "satisfaction_level": 0.68,
        "absenteeism_days": 3,
        "work_life_balance_rating": 3,
        "promotion_last_3years": 0,
        "overtime_frequency": "Rare",
    }
    res = predict_employee_performance(valid_sample, model_key="logistic_regression")
    assert res["predicted_category"] in TARGET_CLASSES
    assert "class_probabilities" in res
    assert len(res["class_probabilities"]) == 3
    # Probabilities should sum to approximately 1.0
    total_prob = sum(res["class_probabilities"].values())
    assert 0.98 <= total_prob <= 1.02
