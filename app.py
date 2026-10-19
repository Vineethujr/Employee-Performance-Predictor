"""
Streamlit Web Dashboard for Employee Performance Predictor.
Interactive multi-page/tab application featuring:
1. Overview & Data Health Audit
2. Employee Analytics (EDA)
3. Model Benchmark & Comparison
4. Hypothetical Employee Prediction (with Input Validation)
5. Synthetic Data Simulation
"""

import json
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from src.config import (
    RAW_DATA_PATH,
    METRICS_SUMMARY_PATH,
    MODEL_METADATA_PATH,
    FIGURES_DIR,
    TARGET_COLUMN,
    TARGET_CLASSES,
    DEPARTMENTS,
    EDUCATION_LEVELS,
    OVERTIME_OPTIONS,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    MODEL_NAMES,
)
from src.predict import predict_employee_performance, load_pipeline
from src.eda import perform_data_quality_audit
from src.data_generator import generate_employee_data

# Page Configuration
st.set_page_config(
    page_title="Employee Performance Predictor",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 1.1rem;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .metric-card h3 {
        color: #0F172A;
    }
    .metric-card p {
        color: #475569;
    }
    .disclaimer-banner {
        background-color: #FEF3C7;
        border-left: 5px solid #F59E0B;
        padding: 14px 18px;
        border-radius: 6px;
        color: #92400E;
        font-size: 0.92rem;
        margin-bottom: 1.5rem;
    }
    .badge-low {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1.1rem;
    }
    .badge-meets {
        background-color: #DBEAFE;
        color: #1E40AF;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1.1rem;
    }
    .badge-high {
        background-color: #D1FAE5;
        color: #065F46;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1.1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_data():
    if not RAW_DATA_PATH.exists():
        df = generate_employee_data(n_samples=1500, random_state=42)
    else:
        df = pd.read_csv(RAW_DATA_PATH)
    return df


@st.cache_data
def load_metrics_and_metadata():
    metrics_df = None
    metadata = {}
    if METRICS_SUMMARY_PATH.exists():
        metrics_df = pd.read_csv(METRICS_SUMMARY_PATH)
    if MODEL_METADATA_PATH.exists():
        with open(MODEL_METADATA_PATH, "r") as f:
            metadata = json.load(f)
    return metrics_df, metadata


# Load data and artifacts
df = load_data()
metrics_df, metadata = load_metrics_and_metadata()

# Sidebar Navigation
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=75)
st.sidebar.title("Navigation")
active_tab = st.sidebar.radio(
    "Go to Module:",
    [
        "📋 Overview & Data Health",
        "📊 Employee Analytics",
        "🤖 Model Comparison",
        "🔮 Hypothetical Prediction",
        "🧪 Synthetic Data Simulation",
    ],
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    **Project Specs:**
    * Models: LogReg, DecTree, RandForest
    * Target: 3-Class Multi-Tier
    * Zero Data Leakage Pipeline
    * Python 3.11 & Scikit-learn
    """
)

# Prominent Global Responsible AI Banner
st.markdown(
    """
    <div class="disclaimer-banner">
        ⚠️ <strong>SYNTHETIC DATA & ETHICAL HR NOTICE:</strong> This application analyzes a synthetic dataset generated
        via mathematical simulation. Results do <strong>NOT</strong> demonstrate real-world employee predictive accuracy.
        Predictions must <strong>never</strong> be used to automate hiring, termination, promotion, or compensation decisions.
    </div>
    """,
    unsafe_allow_html=True,
)

# ==============================================================================
# TAB 1: OVERVIEW & DATA HEALTH
# ==============================================================================
if active_tab == "📋 Overview & Data Health":
    st.markdown('<div class="main-header">📋 Dataset Overview & Quality Audit</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">System metrics, dataset schema, and data integrity verification.</div>', unsafe_allow_html=True)

    audit = perform_data_quality_audit(df)

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown(f'<div class="metric-card"><h3>{audit["total_records"]}</h3><p>Total Records</p></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-card"><h3>{audit["total_columns"]}</h3><p>Total Features</p></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="metric-card"><h3>{audit["duplicate_rows"]}</h3><p>Duplicates</p></div>', unsafe_allow_html=True)
    with col4:
        st.markdown(f'<div class="metric-card"><h3>{audit["total_missing_values"]}</h3><p>Missing Values</p></div>', unsafe_allow_html=True)
    with col5:
        st.markdown(f'<div class="metric-card"><h3>{audit["memory_usage_kb"]} KB</h3><p>Memory Usage</p></div>', unsafe_allow_html=True)

    st.markdown("### 🔍 Dataset Explorer")
    dept_filter = st.multiselect("Filter by Department:", options=DEPARTMENTS, default=DEPARTMENTS)
    perf_filter = st.multiselect("Filter by Performance Category:", options=TARGET_CLASSES, default=TARGET_CLASSES)

    filtered_df = df[df["department"].isin(dept_filter) & df[TARGET_COLUMN].isin(perf_filter)]
    st.dataframe(filtered_df.head(100), use_container_width=True)

    st.markdown("### 📊 Target Label Distribution")
    dist_cols = st.columns([1, 1])
    with dist_cols[0]:
        counts = df[TARGET_COLUMN].value_counts().reindex(TARGET_CLASSES)
        st.table(pd.DataFrame({
            "Category": counts.index,
            "Count": counts.values,
            "Percentage (%)": (counts.values / len(df) * 100).round(2),
        }))

    with dist_cols[1]:
        fig, ax = plt.subplots(figsize=(6, 3.5))
        palette = {"Low performance": "#E57373", "Meets expectations": "#64B5F6", "High performance": "#81C784"}
        ax.bar(counts.index, counts.values, color=[palette[c] for c in counts.index], edgecolor="black")
        ax.set_ylabel("Count")
        st.pyplot(fig)


# ==============================================================================
# TAB 2: EMPLOYEE ANALYTICS (EDA)
# ==============================================================================
elif active_tab == "📊 Employee Analytics":
    st.markdown('<div class="main-header">📊 Exploratory Data Analytics (EDA)</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Interactive analysis of engagement, satisfaction, evaluation, and compensation.</div>', unsafe_allow_html=True)

    eda_subtab = st.selectbox(
        "Select Analytical View:",
        [
            "Department-Level Performance",
            "Numerical Feature Correlation Heatmap",
            "Performance Feature Drivers",
            "Salary vs Evaluation Breakdown",
        ],
    )

    palette = {"Low performance": "#E57373", "Meets expectations": "#64B5F6", "High performance": "#81C784"}

    if eda_subtab == "Department-Level Performance":
        st.markdown("#### Performance Distribution Across Departments")
        dept_ct = pd.crosstab(df["department"], df[TARGET_COLUMN], normalize="index")[TARGET_CLASSES] * 100

        fig, ax = plt.subplots(figsize=(10, 5))
        dept_ct.plot(
            kind="bar",
            stacked=True,
            color=[palette[c] for c in TARGET_CLASSES],
            edgecolor="black",
            alpha=0.9,
            ax=ax,
        )
        ax.set_ylabel("Percentage (%)")
        ax.set_xlabel("Department")
        ax.legend(title="Category", bbox_to_anchor=(1.02, 1), loc="upper left")
        plt.xticks(rotation=0)
        st.pyplot(fig)

    elif eda_subtab == "Numerical Feature Correlation Heatmap":
        st.markdown("#### Pearson Correlation Matrix (Numerical Attributes)")
        corr = df[NUMERICAL_FEATURES].corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        fig, ax = plt.subplots(figsize=(9, 7))
        cmap = sns.diverging_palette(220, 20, as_cmap=True)
        sns.heatmap(corr, mask=mask, cmap=cmap, annot=True, fmt=".2f", center=0, square=True, ax=ax)
        st.pyplot(fig)

    elif eda_subtab == "Performance Feature Drivers":
        st.markdown("#### Key Drivers Comparison Across Performance Tiers")
        feature_choice = st.selectbox(
            "Select Feature to Inspect:",
            [
                "last_evaluation_score",
                "peer_review_score",
                "projects_completed",
                "satisfaction_level",
                "average_monthly_hours",
                "absenteeism_days",
                "training_hours_completed",
            ],
        )
        fig, ax = plt.subplots(figsize=(8, 4.5))
        sns.boxplot(
            data=df,
            x=TARGET_COLUMN,
            y=feature_choice,
            order=TARGET_CLASSES,
            palette=palette,
            hue=TARGET_COLUMN,
            legend=False,
            ax=ax,
        )
        ax.set_title(f"Distribution of {feature_choice} by Performance Category", fontweight="bold")
        st.pyplot(fig)

    elif eda_subtab == "Salary vs Evaluation Breakdown":
        st.markdown("#### Evaluation Score vs Monthly Salary by Department")
        fig, ax = plt.subplots(figsize=(9, 5))
        sns.scatterplot(
            data=df,
            x="last_evaluation_score",
            y="monthly_salary",
            hue=TARGET_COLUMN,
            palette=palette,
            alpha=0.75,
            s=50,
            ax=ax,
        )
        ax.set_xlabel("Last Evaluation Score")
        ax.set_ylabel("Monthly Salary ($)")
        st.pyplot(fig)


# ==============================================================================
# TAB 3: MODEL COMPARISON
# ==============================================================================
elif active_tab == "🤖 Model Comparison":
    st.markdown('<div class="main-header">🤖 Multi-Model Benchmark & Comparison</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Objective evaluation across Logistic Regression, Decision Tree, and Random Forest.</div>',
        unsafe_allow_html=True,
    )

    if metrics_df is not None and not metrics_df.empty:
        st.markdown("### 🏆 Performance Summary (Multi-Class)")
        st.dataframe(
            metrics_df.style.highlight_max(subset=["accuracy", "f1_macro", "precision_macro", "recall_macro"], color="#D1FAE5"),
            use_container_width=True,
        )

        col1, col2 = st.columns([1.2, 1])
        with col1:
            st.markdown("#### Metric Comparison Chart")
            bar_fig_path = FIGURES_DIR / "model_comparison_bar.png"
            if bar_fig_path.exists():
                st.image(str(bar_fig_path), use_column_width=True)

        with col2:
            st.markdown("#### Best Performing Model")
            best_name = metadata.get("best_model_name", "N/A")
            best_f1 = metadata.get("best_model_f1_macro", 0.0)
            st.info(
                f"**Top Model Selected**: {best_name}\n\n"
                f"* **Macro F1-Score**: `{best_f1:.4f}`\n"
                f"* Evaluated on held-out 20% test partition.\n\n"
                f"*Notice*: Model selection avoids the premature assumption that ensemble models (Random Forest) "
                f"always outperform generalized linear models (Logistic Regression)."
            )

        st.markdown("### 🎯 Confusion Matrices")
        cm_fig_path = FIGURES_DIR / "confusion_matrices.png"
        if cm_fig_path.exists():
            st.image(str(cm_fig_path), use_column_width=True)

        # Inspect detailed classification report
        selected_model_key = st.selectbox(
            "Inspect Detailed Classification Report for:",
            options=list(MODEL_NAMES.keys()),
            format_func=lambda x: MODEL_NAMES[x],
        )
        if "detailed_metrics" in metadata and selected_model_key in metadata["detailed_metrics"]:
            rep = metadata["detailed_metrics"][selected_model_key]["classification_report"]
            rep_df = pd.DataFrame(rep).transpose()
            st.dataframe(rep_df.round(3), use_container_width=True)

    else:
        st.warning("Model metrics not found. Please run `python main.py` first to train and evaluate models.")


# ==============================================================================
# TAB 4: HYPOTHETICAL EMPLOYEE PREDICTION
# ==============================================================================
elif active_tab == "🔮 Hypothetical Prediction":
    st.markdown('<div class="main-header">🔮 Hypothetical Employee Prediction</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Simulate multi-tier classification for a hypothetical employee profile.</div>', unsafe_allow_html=True)

    selected_model_key = st.selectbox(
        "Choose Classification Model:",
        options=list(MODEL_NAMES.keys()),
        format_func=lambda x: MODEL_NAMES[x],
    )

    with st.form("employee_prediction_form"):
        st.markdown("#### 1. Department & Background")
        col_a, col_b, col_c, col_d = st.columns(4)
        with col_a:
            dept = st.selectbox("Department", DEPARTMENTS)
        with col_b:
            edu = st.selectbox("Education Level", EDUCATION_LEVELS)
        with col_c:
            age = st.slider("Age", min_value=21, max_value=65, value=32)
        with col_d:
            tenure = st.slider("Tenure (Years)", min_value=0.5, max_value=25.0, value=3.5, step=0.5)

        st.markdown("#### 2. Productivity & Work Hours")
        col_e, col_f, col_g, col_h = st.columns(4)
        with col_e:
            salary = st.number_input("Monthly Salary ($)", min_value=2000.0, max_value=25000.0, value=6500.0, step=250.0)
        with col_f:
            projects = st.slider("Projects Completed", min_value=1, max_value=20, value=6)
        with col_g:
            hours = st.slider("Average Monthly Hours", min_value=100, max_value=280, value=170)
        with col_h:
            training = st.slider("Training Hours Completed", min_value=0, max_value=120, value=35)

        st.markdown("#### 3. Ratings, Engagement & Attendance")
        col_i, col_j, col_k, col_l = st.columns(4)
        with col_i:
            eval_score = st.slider("Last Evaluation Score (0.0 to 1.0)", min_value=0.20, max_value=1.00, value=0.75, step=0.01)
        with col_j:
            peer_score = st.slider("Peer Review Score (1.0 to 5.0)", min_value=1.0, max_value=5.0, value=3.8, step=0.1)
        with col_k:
            satisfaction = st.slider("Satisfaction Level (0.0 to 1.0)", min_value=0.10, max_value=1.00, value=0.70, step=0.01)
        with col_l:
            absent_days = st.slider("Absenteeism (Days)", min_value=0, max_value=30, value=3)

        st.markdown("#### 4. Lifestyle & Career Factors")
        col_m, col_n, col_o = st.columns(3)
        with col_m:
            wlb = st.slider("Work-Life Balance Rating (1 to 5)", min_value=1, max_value=5, value=3)
        with col_n:
            overtime = st.selectbox("Overtime Frequency", OVERTIME_OPTIONS, index=1)
        with col_o:
            promotion = st.selectbox("Promoted in Last 3 Years?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

        submit_btn = st.form_submit_button("Predict Performance Category 🚀", use_container_width=True)

    if submit_btn:
        employee_dict = {
            "department": dept,
            "education_level": edu,
            "age": age,
            "tenure_years": tenure,
            "monthly_salary": salary,
            "projects_completed": projects,
            "average_monthly_hours": hours,
            "training_hours_completed": training,
            "last_evaluation_score": eval_score,
            "peer_review_score": peer_score,
            "satisfaction_level": satisfaction,
            "absenteeism_days": absent_days,
            "work_life_balance_rating": wlb,
            "overtime_frequency": overtime,
            "promotion_last_3years": promotion,
        }

        try:
            pred_result = predict_employee_performance(employee_dict, model_key=selected_model_key)
            cat = pred_result["predicted_category"]
            probs = pred_result["class_probabilities"]

            badge_class = "badge-meets"
            if cat == "Low performance":
                badge_class = "badge-low"
            elif cat == "High performance":
                badge_class = "badge-high"

            st.markdown("---")
            st.markdown("### Prediction Outcome")
            st.markdown(
                f'Predicted Tier: <span class="{badge_class}">{cat}</span> '
                f'(Model: <em>{MODEL_NAMES[selected_model_key]}</em>)',
                unsafe_allow_html=True,
            )

            st.markdown("#### Classification Confidence Probabilities:")
            p_cols = st.columns(3)
            for i, target_cls in enumerate(TARGET_CLASSES):
                prob_val = probs.get(target_cls, 0.0)
                with p_cols[i]:
                    st.metric(label=target_cls, value=f"{prob_val * 100:.1f}%")
                    st.progress(prob_val)

        except Exception as e:
            st.error(f"Inference Error: {str(e)}")


# ==============================================================================
# TAB 5: SYNTHETIC DATA SIMULATION
# ==============================================================================
elif active_tab == "🧪 Synthetic Data Simulation":
    st.markdown('<div class="main-header">🧪 Synthetic Data Simulation Lab</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Adjust organization generation parameters to observe distribution dynamics.</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        This sandbox enables interactive investigation into how varying workforce attributes, sample sizes,
        and organizational noise affect performance score distributions.
        """
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        sim_samples = st.slider("Sample Population Size", min_value=200, max_value=3000, value=800, step=100)
    with col2:
        sim_noise = st.slider("Environmental Noise (Unobserved Factors)", min_value=0.10, max_value=0.80, value=0.35, step=0.05)
    with col3:
        sim_seed = st.number_input("Simulation Random Seed", min_value=1, max_value=9999, value=101)

    if st.button("Run Simulation 🔄", use_container_width=True):
        sim_df = generate_employee_data(
            n_samples=sim_samples,
            random_state=sim_seed,
            noise_level=sim_noise,
            save_path=None,
        )

        st.success(f"Generated simulated sample of {len(sim_df)} employee records!")

        sim_counts = sim_df[TARGET_COLUMN].value_counts().reindex(TARGET_CLASSES)
        sc1, sc2 = st.columns([1, 1.2])

        with sc1:
            st.markdown("#### Simulated Category Balance")
            st.table(pd.DataFrame({
                "Category": sim_counts.index,
                "Count": sim_counts.values,
                "Proportion (%)": (sim_counts.values / len(sim_df) * 100).round(2),
            }))

        with sc2:
            fig, ax = plt.subplots(figsize=(6, 3.8))
            palette = {"Low performance": "#E57373", "Meets expectations": "#64B5F6", "High performance": "#81C784"}
            ax.bar(sim_counts.index, sim_counts.values, color=[palette[c] for c in sim_counts.index], edgecolor="black")
            ax.set_ylabel("Count")
            ax.set_title("Simulated Performance Distribution", fontweight="bold")
            st.pyplot(fig)

        st.markdown("#### Sample Records Preview")
        st.dataframe(sim_df.head(20), use_container_width=True)
