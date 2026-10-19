"""
Main Orchestration Script for Employee Performance Predictor.
Runs the complete end-to-end machine learning and analytics workflow:
1. Synthetic Data Generation
2. Data Quality Audit & Profiling
3. Exploratory Data Analysis & Visualizations
4. Leak-Free Preprocessing & Stratified Splitting
5. Model Training (Logistic Regression, Decision Tree, Random Forest)
6. Multi-Model Evaluation & Metric Comparison
7. Sample Hypothetical Employee Inference Test
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.config import (
    RAW_DATA_PATH,
    TARGET_COLUMN,
    METRICS_SUMMARY_PATH,
    FIGURES_DIR,
)
from src.data_generator import generate_employee_data
from src.eda import perform_data_quality_audit, generate_eda_figures
from src.preprocessor import split_and_save_data
from src.train import train_models
from src.evaluate import compare_and_evaluate_models
from src.predict import predict_employee_performance


def run_pipeline():
    print("\n" + "=" * 65)
    print("   EMPLOYEE PERFORMANCE PREDICTOR - DATA ANALYTICS & ML PIPELINE   ")
    print("=" * 65)

    # Step 1: Synthetic Dataset Generation
    print("\n[Step 1/6] Generating Synthetic HR Dataset...")
    df = generate_employee_data(n_samples=1500, random_state=42)
    print(f"-> Generated {len(df)} records across {df.shape[1]} columns.")
    print("-> Class Distribution:")
    for cat, count in df[TARGET_COLUMN].value_counts().items():
        pct = (count / len(df)) * 100
        print(f"   * {cat:20s}: {count:4d} records ({pct:.1f}%)")

    # Step 2: Data Quality Audit
    print("\n[Step 2/6] Running Data Quality Audit...")
    audit = perform_data_quality_audit(df)
    print(f"-> Total records: {audit['total_records']}, Duplicates: {audit['duplicate_rows']}")
    print(f"-> Total missing values: {audit['total_missing_values']}")
    print(f"-> Total memory footprint: {audit['memory_usage_kb']} KB")

    # Step 3: Exploratory Data Analysis & Visualizations
    print("\n[Step 3/6] Generating EDA Visualizations...")
    figures = generate_eda_figures(df)
    for name, path in figures.items():
        print(f"   * Saved figure [{name}]: {path.name}")

    # Step 4: Leak-Free Preprocessing & Stratified Splitting
    print("\n[Step 4/6] Preprocessing and Splitting Dataset (80/20 Stratified)...")
    X_train, X_test, y_train, y_test = split_and_save_data(df, test_size=0.2, random_state=42)
    print(f"-> Training Set: {len(X_train)} samples")
    print(f"-> Testing Set:  {len(X_test)} samples")

    # Step 5: Model Training
    print("\n[Step 5/6] Training Approved Machine Learning Models...")
    pipelines = train_models(X_train, y_train)

    # Step 6: Multi-Model Evaluation & Comparison
    print("\n[Step 6/6] Evaluating & Benchmarking Models on Unseen Test Split...")
    summary_df = compare_and_evaluate_models(pipelines, X_test, y_test)

    # Sample Inference Demonstration
    print("\n" + "-" * 65)
    print("Sample Hypothetical Employee Inference:")
    print("-" * 65)
    sample_employee = {
        "department": "Engineering",
        "education_level": "Bachelor's",
        "age": 29,
        "tenure_years": 3.0,
        "monthly_salary": 6500.0,
        "projects_completed": 7,
        "average_monthly_hours": 175,
        "training_hours_completed": 40,
        "last_evaluation_score": 0.82,
        "peer_review_score": 4.1,
        "satisfaction_level": 0.78,
        "absenteeism_days": 3,
        "work_life_balance_rating": 4,
        "promotion_last_3years": 0,
        "overtime_frequency": "Rare",
    }
    pred_res = predict_employee_performance(sample_employee)
    print(f"-> Predicted Category: {pred_res['predicted_category']}")
    print("-> Class Probabilities:")
    for cls, prob in pred_res["class_probabilities"].items():
        print(f"   * {cls:20s}: {prob * 100:.1f}%")
    print(f"-> Notice: {pred_res['ethical_disclaimer']}")

    print("\n" + "=" * 65)
    print("   PIPELINE EXECUTION COMPLETED SUCCESSFULLY!   ")
    print("=" * 65)
    print("To launch the interactive dashboard, run:")
    print("   streamlit run app.py\n")


if __name__ == "__main__":
    run_pipeline()
