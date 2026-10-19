"""
Prediction Module for Employee Performance Predictor.
Handles hypothetical employee inference, probability scoring, input validation,
and responsible AI disclosures.
"""

import json
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any, Union, Optional
from pathlib import Path

from src.config import (
    MODELS_DIR,
    MODEL_METADATA_PATH,
    TARGET_CLASSES,
    ALL_FEATURES,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    DEPARTMENTS,
    EDUCATION_LEVELS,
    OVERTIME_OPTIONS,
)


def load_pipeline(model_key: Optional[str] = None) -> Any:
    """
    Load a serialized Scikit-learn pipeline from disk.
    If model_key is omitted, loads the best-performing model recorded in metadata.

    Parameters:
    -----------
    model_key : Optional[str]
        One of 'logistic_regression', 'decision_tree', 'random_forest'.

    Returns:
    --------
    Pipeline: Fitted pipeline object ready for inference.
    """
    if model_key is None:
        if MODEL_METADATA_PATH.exists():
            with open(MODEL_METADATA_PATH, "r") as f:
                meta = json.load(f)
                model_key = meta.get("best_model_key", "random_forest")
        else:
            model_key = "random_forest"

    model_file = MODELS_DIR / f"{model_key}_pipeline.joblib"
    if not model_file.exists():
        raise FileNotFoundError(
            f"Trained model pipeline not found at: {model_file}. "
            f"Please run model training first (e.g., via main.py)."
        )

    pipeline = joblib.load(model_file)
    return pipeline


def validate_employee_inputs(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and sanitize input features for a single employee record.

    Parameters:
    -----------
    data : Dict[str, Any]
        Dictionary of employee attributes.

    Returns:
    --------
    Dict[str, Any]: Validated feature dictionary.

    Raises:
    -------
    ValueError: If a required feature is missing or violates valid bounds.
    """
    validated = {}

    # Check for missing features
    missing = [f for f in ALL_FEATURES if f not in data]
    if missing:
        raise ValueError(f"Missing required employee features: {missing}")

    # Numerical validations
    val_age = int(data["age"])
    if not (18 <= val_age <= 75):
        raise ValueError(f"Age must be between 18 and 75. Got {val_age}")
    validated["age"] = val_age

    val_tenure = float(data["tenure_years"])
    if not (0.0 <= val_tenure <= 40.0):
        raise ValueError(f"Tenure must be between 0 and 40 years. Got {val_tenure}")
    validated["tenure_years"] = val_tenure

    val_salary = float(data["monthly_salary"])
    if val_salary < 1000:
        raise ValueError(f"Monthly salary must be >= 1000. Got {val_salary}")
    validated["monthly_salary"] = val_salary

    validated["projects_completed"] = int(np.clip(data["projects_completed"], 0, 50))
    validated["average_monthly_hours"] = int(np.clip(data["average_monthly_hours"], 60, 350))
    validated["training_hours_completed"] = int(np.clip(data["training_hours_completed"], 0, 200))

    val_eval = float(data["last_evaluation_score"])
    if not (0.0 <= val_eval <= 1.0):
        raise ValueError(f"Last evaluation score must be between 0.0 and 1.0. Got {val_eval}")
    validated["last_evaluation_score"] = val_eval

    val_peer = float(data["peer_review_score"])
    if not (1.0 <= val_peer <= 5.0):
        raise ValueError(f"Peer review score must be between 1.0 and 5.0. Got {val_peer}")
    validated["peer_review_score"] = val_peer

    val_sat = float(data["satisfaction_level"])
    if not (0.0 <= val_sat <= 1.0):
        raise ValueError(f"Satisfaction level must be between 0.0 and 1.0. Got {val_sat}")
    validated["satisfaction_level"] = val_sat

    validated["absenteeism_days"] = int(np.clip(data["absenteeism_days"], 0, 60))
    validated["work_life_balance_rating"] = int(np.clip(data["work_life_balance_rating"], 1, 5))

    # Categorical validations
    if data["department"] not in DEPARTMENTS:
        raise ValueError(f"Department must be one of {DEPARTMENTS}. Got '{data['department']}'")
    validated["department"] = data["department"]

    if data["education_level"] not in EDUCATION_LEVELS:
        raise ValueError(f"Education must be one of {EDUCATION_LEVELS}. Got '{data['education_level']}'")
    validated["education_level"] = data["education_level"]

    if data["overtime_frequency"] not in OVERTIME_OPTIONS:
        raise ValueError(f"Overtime must be one of {OVERTIME_OPTIONS}. Got '{data['overtime_frequency']}'")
    validated["overtime_frequency"] = data["overtime_frequency"]

    validated["promotion_last_3years"] = int(bool(data["promotion_last_3years"]))

    return validated


def predict_employee_performance(
    employee_data: Union[Dict[str, Any], pd.DataFrame],
    model_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Generate performance prediction and class probability breakdown for an employee.

    Parameters:
    -----------
    employee_data : Union[Dict[str, Any], pd.DataFrame]
        Employee attributes.
    model_key : Optional[str]
        Pipeline model to utilize. Defaults to the optimal trained model.

    Returns:
    --------
    Dict[str, Any]: Prediction result containing category, probabilities, and disclaimer.
    """
    pipeline = load_pipeline(model_key)

    if isinstance(employee_data, dict):
        clean_input = validate_employee_inputs(employee_data)
        input_df = pd.DataFrame([clean_input])
    elif isinstance(employee_data, pd.DataFrame):
        input_df = employee_data.copy()
    else:
        raise TypeError("employee_data must be a dict or a pandas DataFrame")

    # Predict class
    prediction = pipeline.predict(input_df)[0]

    # Predict probabilities if supported
    probabilities = {}
    if hasattr(pipeline, "predict_proba"):
        probs = pipeline.predict_proba(input_df)[0]
        classes = pipeline.classes_
        for cls, p in zip(classes, probs):
            probabilities[str(cls)] = round(float(p), 4)

    result = {
        "predicted_category": str(prediction),
        "class_probabilities": probabilities,
        "model_used": model_key or "best_model",
        "ethical_disclaimer": (
            "NOTICE: This prediction is generated from a synthetic benchmark model. "
            "It must NEVER be used to make real employment, compensation, or termination decisions."
        ),
    }

    return result


if __name__ == "__main__":
    sample_employee = {
        "department": "Engineering",
        "education_level": "Master's",
        "age": 32,
        "tenure_years": 4.5,
        "monthly_salary": 7800.0,
        "projects_completed": 8,
        "average_monthly_hours": 185,
        "training_hours_completed": 45,
        "last_evaluation_score": 0.88,
        "peer_review_score": 4.2,
        "satisfaction_level": 0.82,
        "absenteeism_days": 2,
        "work_life_balance_rating": 4,
        "promotion_last_3years": 1,
        "overtime_frequency": "Rare",
    }
    pred = predict_employee_performance(sample_employee)
    print("Prediction Result:")
    print(json.dumps(pred, indent=2))
