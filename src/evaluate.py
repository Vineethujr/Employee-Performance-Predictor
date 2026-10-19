"""
Evaluation Module for Employee Performance Predictor.
Calculates multi-class classification metrics, generates comparison visualizations,
and persists evaluation metadata.
"""

import json
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List
from pathlib import Path
from datetime import datetime
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from sklearn.pipeline import Pipeline

from src.config import (
    TARGET_CLASSES,
    MODEL_NAMES,
    METRICS_SUMMARY_PATH,
    MODEL_METADATA_PATH,
    FIGURES_DIR,
)


def evaluate_single_pipeline(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Dict[str, Any]:
    """
    Compute multi-class evaluation metrics for a single model pipeline on test data.

    Parameters:
    -----------
    pipeline : Pipeline
        Fitted Scikit-learn pipeline.
    X_test : pd.DataFrame
        Unseen test feature set.
    y_test : pd.Series
        Ground-truth target labels.

    Returns:
    --------
    Dict[str, Any]: Detailed evaluation metrics and confusion matrix.
    """
    y_pred = pipeline.predict(X_test)

    # Core multi-class metrics
    acc = accuracy_score(y_test, y_pred)
    prec_macro = precision_score(y_test, y_pred, labels=TARGET_CLASSES, average="macro", zero_division=0)
    rec_macro = recall_score(y_test, y_pred, labels=TARGET_CLASSES, average="macro", zero_division=0)
    f1_macro = f1_score(y_test, y_pred, labels=TARGET_CLASSES, average="macro", zero_division=0)

    prec_weighted = precision_score(y_test, y_pred, labels=TARGET_CLASSES, average="weighted", zero_division=0)
    rec_weighted = recall_score(y_test, y_pred, labels=TARGET_CLASSES, average="weighted", zero_division=0)
    f1_weighted = f1_score(y_test, y_pred, labels=TARGET_CLASSES, average="weighted", zero_division=0)

    cm = confusion_matrix(y_test, y_pred, labels=TARGET_CLASSES)
    cm_norm = confusion_matrix(y_test, y_pred, labels=TARGET_CLASSES, normalize="true")

    report_dict = classification_report(
        y_test,
        y_pred,
        labels=TARGET_CLASSES,
        output_dict=True,
        zero_division=0,
    )

    metrics = {
        "accuracy": round(float(acc), 4),
        "precision_macro": round(float(prec_macro), 4),
        "recall_macro": round(float(rec_macro), 4),
        "f1_macro": round(float(f1_macro), 4),
        "precision_weighted": round(float(prec_weighted), 4),
        "recall_weighted": round(float(rec_weighted), 4),
        "f1_weighted": round(float(f1_weighted), 4),
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_normalized": cm_norm.round(3).tolist(),
        "classification_report": report_dict,
    }

    return metrics


def compare_and_evaluate_models(
    pipelines: Dict[str, Pipeline],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    figures_dir: Path = FIGURES_DIR,
) -> pd.DataFrame:
    """
    Evaluate all models, compare metrics without bias, and save results.

    Outputs:
      - metrics_summary.csv
      - model_metadata.json
      - figures/confusion_matrices.png
      - figures/model_comparison_bar.png

    Returns:
    --------
    pd.DataFrame: Summary table comparing all models.
    """
    figures_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    summary_rows = []

    for key, pipeline in pipelines.items():
        metrics = evaluate_single_pipeline(pipeline, X_test, y_test)
        results[key] = metrics

        summary_rows.append({
            "model_key": key,
            "model_name": MODEL_NAMES[key],
            "accuracy": metrics["accuracy"],
            "f1_macro": metrics["f1_macro"],
            "precision_macro": metrics["precision_macro"],
            "recall_macro": metrics["recall_macro"],
            "f1_weighted": metrics["f1_weighted"],
            "precision_weighted": metrics["precision_weighted"],
            "recall_weighted": metrics["recall_weighted"],
        })

    summary_df = pd.DataFrame(summary_rows).sort_values(by="f1_macro", ascending=False)
    summary_df.to_csv(METRICS_SUMMARY_PATH, index=False)
    print(f"\n[Evaluate] Metrics summary table saved to: {METRICS_SUMMARY_PATH}")
    print(summary_df[["model_name", "accuracy", "f1_macro", "precision_macro", "recall_macro"]].to_string(index=False))

    # Identify top performing model based on Macro F1 (multi-class standard)
    best_row = summary_df.iloc[0]
    best_model_key = best_row["model_key"]
    best_model_name = best_row["model_name"]

    metadata = {
        "evaluation_timestamp": datetime.now().isoformat(),
        "test_sample_size": len(y_test),
        "target_classes": TARGET_CLASSES,
        "best_model_key": best_model_key,
        "best_model_name": best_model_name,
        "best_model_f1_macro": best_row["f1_macro"],
        "detailed_metrics": results,
    }

    MODEL_METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"[Evaluate] Saved model metadata to: {MODEL_METADATA_PATH}")

    # -------------------------------------------------------------
    # Visualization 1: Side-by-Side Confusion Matrices
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    for i, (key, pipeline) in enumerate(pipelines.items()):
        ax = axes[i]
        cm = np.array(results[key]["confusion_matrix"])
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Low", "Meets", "High"],
            yticklabels=["Low", "Meets", "High"],
            cbar=False,
            ax=ax,
        )
        acc = results[key]["accuracy"]
        f1 = results[key]["f1_macro"]
        ax.set_title(f"{MODEL_NAMES[key]}\nAcc: {acc:.3f} | Macro F1: {f1:.3f}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Predicted Class")
        ax.set_ylabel("True Class")

    plt.suptitle("Confusion Matrix Comparison Across Models", fontsize=14, fontweight="bold", y=1.02)
    cm_fig_path = figures_dir / "confusion_matrices.png"
    plt.savefig(cm_fig_path, dpi=300, bbox_inches="tight")
    plt.close()

    # -------------------------------------------------------------
    # Visualization 2: Metric Comparison Bar Chart
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_df = summary_df.melt(
        id_vars=["model_name"],
        value_vars=["accuracy", "f1_macro", "precision_macro", "recall_macro"],
        var_name="Metric",
        value_name="Score",
    )
    metric_labels = {
        "accuracy": "Accuracy",
        "f1_macro": "Macro F1",
        "precision_macro": "Macro Precision",
        "recall_macro": "Macro Recall",
    }
    plot_df["Metric"] = plot_df["Metric"].map(metric_labels)

    sns.barplot(
        data=plot_df,
        x="Metric",
        y="Score",
        hue="model_name",
        palette="viridis",
        edgecolor="black",
        alpha=0.9,
        ax=ax,
    )
    ax.set_title("Multi-Class Model Performance Comparison", fontsize=13, fontweight="bold", pad=12)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score (0.0 to 1.0)")
    ax.set_xlabel("")
    ax.legend(title="Algorithm", bbox_to_anchor=(1.02, 1), loc="upper left")

    for p in ax.patches:
        h = p.get_height()
        if h > 0:
            ax.annotate(
                f"{h:.2f}",
                (p.get_x() + p.get_width() / 2.0, h + 0.015),
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

    bar_fig_path = figures_dir / "model_comparison_bar.png"
    plt.savefig(bar_fig_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[Evaluate] Generated figures saved to: {figures_dir}\n")
    return summary_df


if __name__ == "__main__":
    from src.data_generator import generate_employee_data
    from src.preprocessor import split_and_save_data
    from src.train import train_models

    df = generate_employee_data()
    X_train, X_test, y_train, y_test = split_and_save_data(df)
    pipelines = train_models(X_train, y_train)
    summary = compare_and_evaluate_models(pipelines, X_test, y_test)
