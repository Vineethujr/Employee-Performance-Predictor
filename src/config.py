"""
Configuration module for Employee Performance Predictor.
Centralizes paths, random seeds, target classes, and feature definitions.
"""
from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "employee_data_raw.csv"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
TRAIN_DATA_PATH = PROCESSED_DATA_DIR / "train.csv"
TEST_DATA_PATH = PROCESSED_DATA_DIR / "test.csv"

MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
METRICS_SUMMARY_PATH = OUTPUTS_DIR / "metrics_summary.csv"
MODEL_METADATA_PATH = MODELS_DIR / "model_metadata.json"

# Random Seed for Reproducibility
RANDOM_STATE = 42

# Target Definition
TARGET_COLUMN = "performance_category"
TARGET_CLASSES = ["Low performance", "Meets expectations", "High performance"]

# Feature Definitions
ID_COLUMN = "employee_id"

NUMERICAL_FEATURES = [
    "age",
    "tenure_years",
    "monthly_salary",
    "projects_completed",
    "average_monthly_hours",
    "training_hours_completed",
    "last_evaluation_score",
    "peer_review_score",
    "satisfaction_level",
    "absenteeism_days",
    "work_life_balance_rating",
]

CATEGORICAL_FEATURES = [
    "department",
    "education_level",
    "overtime_frequency",
    "promotion_last_3years",
]

ALL_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES

# Department and Categorical Constants for UI and Validation
DEPARTMENTS = ["Sales", "Engineering", "Marketing", "Operations", "Finance", "HR"]
EDUCATION_LEVELS = ["High School", "Bachelor's", "Master's", "PhD"]
OVERTIME_OPTIONS = ["Never", "Rare", "Often"]
PROMOTION_OPTIONS = [0, 1]

# Model Names
MODEL_NAMES = {
    "logistic_regression": "Logistic Regression",
    "decision_tree": "Decision Tree Classifier",
    "random_forest": "Random Forest Classifier",
}
