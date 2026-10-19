"""
Data Generator Module for Employee Performance Predictor.

DISCLOSURE & RESPONSIBLE AI NOTE:
---------------------------------
This script generates a SYNTHETIC human resources dataset for educational,
analytical, and prototyping purposes only. Model results derived from this
dataset do NOT demonstrate real-world employee-performance prediction accuracy.
Under no circumstances should this model or simulation be used to automate real
hiring, promotion, disciplinary, or termination decisions.
"""

import numpy as np
import pandas as pd
from typing import Optional
from pathlib import Path

from src.config import (
    RAW_DATA_PATH,
    RANDOM_STATE,
    TARGET_COLUMN,
    TARGET_CLASSES,
    DEPARTMENTS,
    EDUCATION_LEVELS,
    OVERTIME_OPTIONS,
)


def generate_employee_data(
    n_samples: int = 1500,
    random_state: int = RANDOM_STATE,
    noise_level: float = 0.35,
    save_path: Optional[Path] = RAW_DATA_PATH,
) -> pd.DataFrame:
    """
    Generate a realistic synthetic HR dataset with documented performance labels.

    Label Generation Mechanism:
    ---------------------------
    A latent performance score is computed via a weighted combination of:
      + Last evaluation score (weight: +3.0)
      + Peer review score (weight: +2.0)
      + Projects completed (weight: +1.5)
      + Satisfaction level (weight: +1.2)
      + Training hours completed (weight: +0.8)
      + Recent promotion (weight: +0.5)
      - Absenteeism days (weight: -1.5)
      - Work hours imbalance (penalty for severe underwork <130h or overwork >240h)
      + Gaussian noise N(0, noise_level^2) to reflect unobserved real-world factors.

    The continuous latent score is partitioned into three quantile-based tiers:
      - Bottom ~28%: 'Low performance'
      - Middle ~47%: 'Meets expectations'
      - Top ~25%: 'High performance'

    Parameters:
    -----------
    n_samples : int
        Number of employee records to generate (default: 1500).
    random_state : int
        Random seed for reproducibility.
    noise_level : float
        Standard deviation of Gaussian noise added to the latent performance score.
    save_path : Optional[Path]
        If provided, writes the generated dataset to this CSV path.

    Returns:
    --------
    pd.DataFrame: Synthetic employee dataset.
    """
    rng = np.random.default_rng(random_state)

    # 1. Employee Identifiers
    employee_ids = [f"EMP{i+1:04d}" for i in range(n_samples)]

    # 2. Demographic & Organizational Attributes
    dept_probs = [0.26, 0.24, 0.18, 0.14, 0.10, 0.08]  # Sales, Eng, Mkt, Ops, Fin, HR
    department = rng.choice(DEPARTMENTS, size=n_samples, p=dept_probs)

    edu_probs = [0.10, 0.55, 0.28, 0.07]  # High School, Bachelor's, Master's, PhD
    education_level = rng.choice(EDUCATION_LEVELS, size=n_samples, p=edu_probs)

    # Age: Normal distribution centered around 35 (bounded 22 to 62)
    age = np.clip(np.round(rng.normal(loc=35.0, scale=8.5, size=n_samples)), 22, 62).astype(int)

    # Tenure (years): Dependent on age (cannot exceed age - 21)
    max_tenure_possible = np.maximum(age - 21, 1)
    tenure_raw = rng.exponential(scale=4.5, size=n_samples)
    tenure_years = np.clip(np.round(np.minimum(tenure_raw, max_tenure_possible * 0.85), 1), 0.5, 25.0)

    # Base salary driven by education, department, and tenure
    edu_salary_bonus = {
        "High School": 0,
        "Bachelor's": 6000,
        "Master's": 14000,
        "PhD": 24000,
    }
    dept_salary_mult = {
        "Engineering": 1.25,
        "Finance": 1.15,
        "Marketing": 1.05,
        "Operations": 0.95,
        "Sales": 1.00,
        "HR": 0.90,
    }

    base_salary = 38000 + np.array([edu_salary_bonus[e] for e in education_level])
    mult_array = np.array([dept_salary_mult[d] for d in department])
    annual_salary = (base_salary + tenure_years * 2200 + rng.normal(0, 3000, n_samples)) * mult_array
    monthly_salary = np.round(np.clip(annual_salary / 12.0, 3200, 19500), 2)

    # 3. Work Engagement & Productivity Indicators
    # Average monthly work hours: Normal(170, 24) bounded between 110 and 260
    average_monthly_hours = np.clip(
        np.round(rng.normal(loc=172.0, scale=24.0, size=n_samples)), 110, 260
    ).astype(int)

    # Overtime frequency correlated with hours
    overtime_frequency = []
    for h in average_monthly_hours:
        if h > 195:
            overtime_frequency.append(rng.choice(["Often", "Rare", "Never"], p=[0.70, 0.25, 0.05]))
        elif h > 165:
            overtime_frequency.append(rng.choice(["Often", "Rare", "Never"], p=[0.20, 0.65, 0.15]))
        else:
            overtime_frequency.append(rng.choice(["Often", "Rare", "Never"], p=[0.05, 0.35, 0.60]))
    overtime_frequency = np.array(overtime_frequency)

    # Projects completed per year: Poisson distribution (typical 3 to 14)
    projects_completed = np.clip(rng.poisson(lam=5.5, size=n_samples), 1, 16)

    # Training hours completed: Log-normal distribution
    training_hours_completed = np.clip(
        np.round(rng.exponential(scale=28.0, size=n_samples)), 0, 110
    ).astype(int)

    # Absenteeism (days per year): Poisson distribution
    absenteeism_days = np.clip(rng.poisson(lam=4.2, size=n_samples), 0, 28)

    # Satisfaction level: 0.10 to 1.00 (Beta distribution for realistic left-skew)
    satisfaction_level = np.round(
        np.clip(rng.beta(a=5, b=2.8, size=n_samples), 0.12, 0.98), 2
    )

    # Work-life balance rating (1 to 5)
    work_life_balance_rating = np.clip(
        np.round(rng.normal(loc=3.3 - (average_monthly_hours - 160) * 0.02, scale=0.9, size=n_samples)),
        1,
        5,
    ).astype(int)

    # Peer review score: 1.0 to 5.0
    peer_review_score = np.round(
        np.clip(rng.normal(loc=3.55, scale=0.65, size=n_samples), 1.1, 4.95), 2
    )

    # Managerial last evaluation score: 0.20 to 1.00
    last_evaluation_score = np.round(
        np.clip(
            0.35
            + 0.08 * peer_review_score
            + 0.02 * projects_completed
            - 0.008 * absenteeism_days
            + rng.normal(0, 0.10, size=n_samples),
            0.20,
            1.00,
        ),
        2,
    )

    # Promotion in the last 3 years: Binary
    promotion_prob = np.clip(
        0.05 + 0.12 * (last_evaluation_score > 0.80) + 0.05 * (tenure_years > 2.0),
        0.02,
        0.40,
    )
    promotion_last_3years = (rng.uniform(size=n_samples) < promotion_prob).astype(int)

    # 4. Latent Performance Score Formulation
    # Standardize individual components to comparable scale
    z_eval = (last_evaluation_score - 0.70) / 0.15
    z_peer = (peer_review_score - 3.5) / 0.65
    z_proj = (projects_completed - 5.5) / 2.5
    z_satis = (satisfaction_level - 0.65) / 0.20
    z_train = (training_hours_completed - 28.0) / 18.0
    z_absent = (absenteeism_days - 4.2) / 3.0

    # Hours balance penalty (underworked or heavily overworked)
    hours_balance_penalty = np.where(
        average_monthly_hours < 135,
        (135 - average_monthly_hours) * 0.03,
        np.where(average_monthly_hours > 230, (average_monthly_hours - 230) * 0.02, 0.0),
    )

    latent_score = (
        3.0 * z_eval
        + 2.0 * z_peer
        + 1.5 * z_proj
        + 1.2 * z_satis
        + 0.8 * z_train
        + 0.6 * promotion_last_3years
        - 1.6 * z_absent
        - hours_balance_penalty
        + rng.normal(loc=0.0, scale=noise_level * 3.5, size=n_samples)
    )

    # Determine thresholds for 3 classes
    p28, p75 = np.percentile(latent_score, [28.0, 75.0])

    performance_categories = []
    for s in latent_score:
        if s < p28:
            performance_categories.append(TARGET_CLASSES[0])  # Low performance
        elif s < p75:
            performance_categories.append(TARGET_CLASSES[1])  # Meets expectations
        else:
            performance_categories.append(TARGET_CLASSES[2])  # High performance

    # 5. Assemble DataFrame
    df = pd.DataFrame({
        "employee_id": employee_ids,
        "department": department,
        "education_level": education_level,
        "age": age,
        "tenure_years": tenure_years,
        "monthly_salary": monthly_salary,
        "projects_completed": projects_completed,
        "average_monthly_hours": average_monthly_hours,
        "training_hours_completed": training_hours_completed,
        "last_evaluation_score": last_evaluation_score,
        "peer_review_score": peer_review_score,
        "satisfaction_level": satisfaction_level,
        "absenteeism_days": absenteeism_days,
        "work_life_balance_rating": work_life_balance_rating,
        "promotion_last_3years": promotion_last_3years,
        "overtime_frequency": overtime_frequency,
        TARGET_COLUMN: performance_categories,
    })

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_path, index=False)
        print(f"[DataGenerator] Generated {len(df)} records. Saved to: {save_path}")

    return df


if __name__ == "__main__":
    df = generate_employee_data()
    print("Class distribution:")
    print(df[TARGET_COLUMN].value_counts(normalize=True))
