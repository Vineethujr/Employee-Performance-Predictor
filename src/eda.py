"""
Exploratory Data Analysis (EDA) Module for Employee Performance Predictor.
Conducts statistical profiling, data quality audits, and produces figures.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any

from src.config import (
    FIGURES_DIR,
    TARGET_COLUMN,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    TARGET_CLASSES,
)

# Configure styling for visual clarity
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({"font.size": 11, "figure.autolayout": True})


def perform_data_quality_audit(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Perform structural data quality checks including nulls, duplicates, and ranges.

    Returns:
    --------
    Dict[str, Any]: Audit summary dictionary.
    """
    total_rows = len(df)
    missing_counts = df.isnull().sum().to_dict()
    total_missing = sum(missing_counts.values())
    duplicate_rows = int(df.duplicated().sum())

    audit_summary = {
        "total_records": total_rows,
        "total_columns": df.shape[1],
        "total_missing_values": total_missing,
        "duplicate_rows": duplicate_rows,
        "missing_per_column": missing_counts,
        "memory_usage_kb": round(df.memory_usage(deep=True).sum() / 1024, 2),
    }

    return audit_summary


def generate_eda_figures(df: pd.DataFrame, figures_dir: Path = FIGURES_DIR) -> Dict[str, Path]:
    """
    Generate and persist high-resolution publication-quality figures.

    1. Target Category Distribution (Count & Percentages)
    2. Correlation Heatmap of Numerical Attributes
    3. Department vs Performance Category Cross-tabulation
    4. Key Feature Distributions across Performance Categories

    Parameters:
    -----------
    df : pd.DataFrame
        Dataset containing features and target column.
    figures_dir : Path
        Output directory for saved PNG artifacts.

    Returns:
    --------
    Dict[str, Path]: Map of figure names to saved file paths.
    """
    figures_dir.mkdir(parents=True, exist_ok=True)
    saved_paths = {}

    # Palette mapped to classes for visual consistency
    palette = {
        "Low performance": "#E57373",      # Soft Red
        "Meets expectations": "#64B5F6",   # Soft Blue
        "High performance": "#81C784",     # Soft Green
    }

    # -------------------------------------------------------------
    # Figure 1: Target Category Distribution
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    counts = df[TARGET_COLUMN].value_counts().reindex(TARGET_CLASSES)
    pcts = (counts / len(df) * 100).round(1)

    bars = ax.bar(counts.index, counts.values, color=[palette[c] for c in counts.index], edgecolor="black", alpha=0.85)
    for bar, pct, count in zip(bars, pcts, counts):
        yval = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            yval + 15,
            f"{count} ({pct}%)",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    ax.set_title("Performance Category Distribution (Target Variable)", fontsize=13, fontweight="bold", pad=12)
    ax.set_ylabel("Number of Employees")
    ax.set_ylim(0, max(counts.values) * 1.18)
    fig_path = figures_dir / "target_distribution.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()
    saved_paths["target_distribution"] = fig_path

    # -------------------------------------------------------------
    # Figure 2: Correlation Heatmap
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 8))
    corr_matrix = df[NUMERICAL_FEATURES].corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    cmap = sns.diverging_palette(220, 20, as_cmap=True)

    sns.heatmap(
        corr_matrix,
        mask=mask,
        cmap=cmap,
        vmax=1.0,
        vmin=-1.0,
        center=0,
        annot=True,
        fmt=".2f",
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
        ax=ax,
    )
    ax.set_title("Pearson Correlation Heatmap (Numerical Features)", fontsize=13, fontweight="bold", pad=12)
    fig_path = figures_dir / "correlation_heatmap.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()
    saved_paths["correlation_heatmap"] = fig_path

    # -------------------------------------------------------------
    # Figure 3: Department vs Performance Category
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6))
    dept_ct = pd.crosstab(df["department"], df[TARGET_COLUMN], normalize="index")[TARGET_CLASSES] * 100

    dept_ct.plot(
        kind="bar",
        stacked=True,
        color=[palette[c] for c in TARGET_CLASSES],
        edgecolor="black",
        alpha=0.9,
        ax=ax,
    )
    ax.set_title("Performance Category Composition by Department (%)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Department")
    ax.set_ylabel("Percentage (%)")
    ax.legend(title="Performance Category", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.xticks(rotation=0)
    fig_path = figures_dir / "department_performance.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()
    saved_paths["department_performance"] = fig_path

    # -------------------------------------------------------------
    # Figure 4: Feature Distributions across Target Categories
    # -------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    sample_features = [
        ("last_evaluation_score", "Evaluation Score (0 - 1)"),
        ("peer_review_score", "Peer Review Score (1 - 5)"),
        ("projects_completed", "Annual Projects Completed"),
        ("satisfaction_level", "Job Satisfaction Level (0 - 1)"),
    ]

    for ax, (feat, label) in zip(axes.flatten(), sample_features):
        sns.boxplot(
            data=df,
            x=TARGET_COLUMN,
            y=feat,
            hue=TARGET_COLUMN,
            legend=False,
            order=TARGET_CLASSES,
            palette=palette,
            ax=ax,
        )
        ax.set_title(f"Distribution of {label}", fontsize=11, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel(label)

    fig_path = figures_dir / "feature_distributions.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()
    saved_paths["feature_distributions"] = fig_path

    print(f"[EDA] Generated and saved 4 figures to: {figures_dir}")
    return saved_paths


if __name__ == "__main__":
    from src.data_generator import generate_employee_data
    df = generate_employee_data()
    audit = perform_data_quality_audit(df)
    print("Data Quality Audit Summary:", audit)
    figs = generate_eda_figures(df)
