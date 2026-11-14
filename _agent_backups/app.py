import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="RetainIQ – Cost-Sensitive Customer Retention Intelligence",
    page_icon="💡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- MODERN SaaS STYLING ---
st.markdown("""
    <style>
    .main { background-color: #0e1117; color: #fafafa; }
    .sidebar .sidebar-content { background-color: #161b22; }
    .metric-card {
        background: #1f242d;
        border: 1px solid #30363d;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stButton>button {
        background: linear-gradient(135deg, #2563eb, #7c3aed);
        color: white;
        font-weight: bold;
        border: none;
        border-radius: 6px;
        padding: 10px 20px;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #1d4ed8, #6d28d9);
    }
    </style>
""", unsafe_allow_html=True)

# --- LOAD ASSETS ---
DATA_PATH = os.path.join("data", "telco_churn.csv")
MODEL_DIR = "models"


@st.cache_data
def load_data():
    if not os.path.exists(DATA_PATH):
        return None
    df = pd.read_csv(DATA_PATH)
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df.loc[(df["tenure"] == 0) & df["TotalCharges"].isna(), "TotalCharges"] = 0.0
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())
    return df


@st.cache_data
def load_evaluation_results():
    results_path = os.path.join(MODEL_DIR, "evaluation_results.json")
    if os.path.exists(results_path):
        with open(results_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


@st.cache_resource
def load_models_and_preprocessor():
    models = {}
    model_files = {
        "Logistic Regression": "logistic_regression.pkl",
        "Random Forest": "random_forest.pkl",
        "Gradient Boosting": "gradient_boosting.pkl",
        "Neural Network": "neural_network.pkl"
    }
    for name, filename in model_files.items():
        path = os.path.join(MODEL_DIR, filename)
        if os.path.exists(path):
            models[name] = joblib.load(path)

    preprocessor = None
    preprocessor_path = os.path.join(MODEL_DIR, "preprocessor.pkl")
    if os.path.exists(preprocessor_path):
        preprocessor = joblib.load(preprocessor_path)
    elif models:
        # Fallback: extract preprocessor step from the first pipeline model
        for m in models.values():
            if hasattr(m, "named_steps") and "preprocessor" in m.named_steps:
                preprocessor = m.named_steps["preprocessor"]
                break

    return preprocessor, models


df = load_data()
eval_data = load_evaluation_results()
preprocessor, models = load_models_and_preprocessor()

# --- SIDEBAR NAVIGATION ---
st.sidebar.markdown("## 💡 RetainIQ")
st.sidebar.markdown("### Cost-Sensitive Churn Intelligence")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    ["Overview", "Customer Churn Prediction", "Model Comparison", "Cost Calculator", "Customer Insights"]
)

st.sidebar.markdown("---")
st.sidebar.info(
    "**Project Note:** RetainIQ evaluates machine learning models using an explicitly defined business cost matrix "
    "rather than accuracy alone."
)

if df is None:
    st.error("Dataset not found! Please ensure `telco_churn.csv` is placed in the `data/` folder.")
    st.stop()

# ==========================================
# PAGE 1: OVERVIEW
# ==========================================
if page == "Overview":
    st.title("💡 RetainIQ – Customer Retention Intelligence")
    st.markdown("### Executive Dashboard & Project Overview")
    st.markdown(
        "Welcome to **RetainIQ**, an advanced customer churn prediction and cost-sensitive evaluation platform. "
        "Traditional churn models focus solely on accuracy or F1-score. RetainIQ goes a step further by calculating "
        "the **actual financial business cost** of classification mistakes (False Negatives vs. False Positives)."
    )

    total_customers = len(df)
    churned_customers = int((df["Churn"] == "Yes").sum())
    retained_customers = total_customers - churned_customers
    churn_rate = (churned_customers / total_customers) * 100

    best_cost_model = eval_data.get("best_by", {}).get("cost", "Logistic Regression") if eval_data else "Logistic Regression"

    if eval_data and "models" in eval_data and best_cost_model in eval_data["models"]:
        model_info = eval_data["models"][best_cost_model]
        t_m = model_info.get("test", model_info)
        best_cost_val = t_m.get("total_cost", t_m.get("cost", 7080.0))
    else:
        best_cost_val = 7080.0

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Total Customers", f"{total_customers:,}")
    with col2:
        st.metric("Retained", f"{retained_customers:,}")
    with col3:
        st.metric("Churned", f"{churned_customers:,}")
    with col4:
        st.metric("Churn Rate", f"{churn_rate:.1f}%")
    with col5:
        st.metric("Best Cost Model", best_cost_model, f"${best_cost_val:,.0f}")

    st.markdown("---")

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Customer Churn Distribution")
        fig_churn = px.pie(
            df, names="Churn", hole=0.4,
            color_discrete_sequence=["#2563eb", "#7c3aed"],
            title="Proportion of Churned vs Retained Customers"
        )
        fig_churn.update_layout(paper_bgcolor="#161b22", font_color="#fafafa")
        st.plotly_chart(fig_churn, width="stretch")

    with col_b:
        st.subheader("Churn Distribution by Contract Type")
        fig_contract = px.histogram(
            df, x="Contract", color="Churn", barmode="group",
            color_discrete_sequence=["#2563eb", "#7c3aed"],
            title="Contract Type vs Churn"
        )
        fig_contract.update_layout(paper_bgcolor="#161b22", plot_bgcolor="#161b22", font_color="#fafafa")
        st.plotly_chart(fig_contract, width="stretch")

    col_c, col_d = st.columns(2)
    with col_c:
        st.subheader("Tenure vs Churn")
        fig_tenure = px.box(
            df, x="Churn", y="tenure", color="Churn",
            color_discrete_sequence=["#2563eb", "#7c3aed"],
            title="Tenure Distribution by Churn Status"
        )
        fig_tenure.update_layout(paper_bgcolor="#161b22", plot_bgcolor="#161b22", font_color="#fafafa")
        st.plotly_chart(fig_tenure, width="stretch")

    with col_d:
        st.subheader("Monthly Charges vs Churn")
        fig_charges = px.box(
            df, x="Churn", y="MonthlyCharges", color="Churn",
            color_discrete_sequence=["#2563eb", "#7c3aed"],
            title="Monthly Charges Distribution by Churn Status"
        )
        fig_charges.update_layout(paper_bgcolor="#161b22", plot_bgcolor="#161b22", font_color="#fafafa")
        st.plotly_chart(fig_charges, width="stretch")

# ==========================================
# PAGE 2: CUSTOMER CHURN PREDICTION
# ==========================================
elif page == "Customer Churn Prediction":
    st.title("🎯 Customer Churn Prediction Engine")
    st.markdown("Enter customer details below to predict their churn probability using trained models and cost-optimized decision thresholds.")

    if not models:
        st.error("Trained models not found. Please run `src/evaluation.py` first.")
        st.stop()

    selected_model_name = st.selectbox("Select Machine Learning Model", list(models.keys()))

    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            gender = st.selectbox("Gender", ["Male", "Female"])
            senior_citizen = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
            partner = st.selectbox("Partner", ["Yes", "No"])
            dependents = st.selectbox("Dependents", ["Yes", "No"])
            tenure = st.slider("Tenure (Months)", min_value=0, max_value=72, value=12)
            phone_service = st.selectbox("Phone Service", ["Yes", "No"])
            multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])

        with col2:
            internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
            online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
            online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
            device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
            tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
            streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
            streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

        with col3:
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
            payment_method = st.selectbox("Payment Method", [
                "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
            ])
            monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=200.0, value=70.0)
            total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=850.0)

        submit_btn = st.form_submit_button("Predict Churn Risk")

    if submit_btn:
        input_data = pd.DataFrame([{
            "SeniorCitizen": senior_citizen,
            "tenure": tenure,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
            "gender": gender,
            "Partner": partner,
            "Dependents": dependents,
            "PhoneService": phone_service,
            "MultipleLines": multiple_lines,
            "InternetService": internet_service,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless_billing,
            "PaymentMethod": payment_method
        }])

        model = models[selected_model_name]

        # Predict churn probability
        try:
            if hasattr(model, "predict_proba"):
                if hasattr(model, "named_steps"):
                    prob = float(model.predict_proba(input_data)[0][1])
                elif preprocessor is not None:
                    X_input = preprocessor.transform(input_data)
                    prob = float(model.predict_proba(X_input)[0][1])
                else:
                    prob = float(model.predict_proba(input_data)[0][1])
            else:
                prob = 0.5
        except Exception:
            if preprocessor is not None:
                X_input = preprocessor.transform(input_data)
                prob = float(model.predict_proba(X_input)[0][1])
            else:
                prob = 0.5

        # Retrieve model cost-optimized threshold
        model_info = eval_data.get("models", {}).get(selected_model_name, {}) if eval_data else {}
        threshold = float(model_info.get("threshold", 0.5))
        pred_class = 1 if prob >= threshold else 0

        st.markdown("---")
        st.subheader("Prediction Results")

        res_col1, res_col2, res_col3 = st.columns(3)
        with res_col1:
            st.metric("Predicted Churn Status", "Churn (Likely to Leave)" if pred_class == 1 else "Retained (Likely to Stay)")
        with res_col2:
            st.metric("Churn Probability", f"{prob * 100:.2f}%")
        with res_col3:
            st.metric("Cost-Optimized Threshold", f"{threshold:.2f}")

        if pred_class == 1:
            st.warning(
                f"⚠️ **High Churn Risk:** The selected model (`{selected_model_name}`) estimates a churn probability of "
                f"**{prob * 100:.1f}%**, which exceeds the cost-optimized threshold of **{threshold:.2f}**. "
                f"Under the $100 FN cost matrix, taking proactive retention action (e.g., offering a plan discount or tech consultation) is financially justified."
            )
        else:
            st.success(
                f"✅ **Low Churn Risk:** The selected model (`{selected_model_name}`) estimates a churn probability of "
                f"**{prob * 100:.1f}%**, which is below the threshold of **{threshold:.2f}**. The customer is predicted to remain active."
            )

# ==========================================
# PAGE 3: MODEL COMPARISON
# ==========================================
elif page == "Model Comparison":
    st.title("📊 Model Comparison & Cost-Sensitive Evaluation")
    st.markdown("Compare all four machine learning algorithms across statistical metrics and cost-sensitive business criteria.")

    if not eval_data:
        st.error("Evaluation results not found. Please run `src/evaluation.py` first.")
        st.stop()

    models_dict = eval_data.get("models", {})
    table_data = []
    for name, info in models_dict.items():
        t = info.get("test", info)
        cost_val = t.get("total_cost", t.get("cost", t.get("test_cost", 0.0)))
        cost_per_cust = t.get("cost_per_customer", 0.0)
        table_data.append({
            "Model": name,
            "Accuracy": f"{t.get('accuracy', 0.0):.4f}",
            "Precision": f"{t.get('precision', 0.0):.4f}",
            "Recall": f"{t.get('recall', 0.0):.4f}",
            "F1-score": f"{t.get('f1', 0.0):.4f}",
            "ROC-AUC": f"{t.get('roc_auc', 0.0):.4f}",
            "Threshold": f"{info.get('threshold', t.get('threshold', 0.5)):.2f}",
            "False Negatives (FN)": int(t.get('fn', 0)),
            "False Positives (FP)": int(t.get('fp', 0)),
            "Test Cost ($)": f"${cost_val:,.2f}",
            "Cost/Customer ($)": f"${cost_per_cust:,.2f}"
        })

    df_table = pd.DataFrame(table_data)
    st.dataframe(df_table, width="stretch")

    st.markdown("---")
    st.subheader("Key Findings")
    col1, col2, col3 = st.columns(3)
    best_by = eval_data.get("best_by", {})
    with col1:
        st.info(f"🏆 **Lowest Cost Model:**\n\n**{best_by.get('cost', 'Logistic Regression')}**\n\nMinimizes missed churners (FN) under the $100/$10 cost matrix.")
    with col2:
        st.info(f"⭐ **Best F1-Score Model:**\n\n**{best_by.get('f1', 'Neural Network')}**\n\nBalances precision and recall at the optimized decision boundary.")
    with col3:
        st.info(f"📈 **Best ROC-AUC Model:**\n\n**{best_by.get('roc_auc', 'Gradient Boosting')}**\n\nDemonstrates superior overall probability calibration and ranking ability.")

    st.markdown("---")
    st.subheader("Visual Model Comparison")

    metric_options = {
        "total_cost": "Test Cost ($) [Lower is Better]",
        "f1": "F1-Score [Higher is Better]",
        "roc_auc": "ROC-AUC [Higher is Better]",
        "recall": "Recall [Higher is Better]",
        "precision": "Precision [Higher is Better]",
        "accuracy": "Accuracy [Higher is Better]"
    }
    metric_to_plot = st.selectbox("Select Metric to Compare", list(metric_options.keys()), format_func=lambda x: metric_options[x])

    model_names = list(models_dict.keys())
    metric_values = []
    for m in model_names:
        t = models_dict[m].get("test", models_dict[m])
        metric_values.append(t.get(metric_to_plot, 0.0))

    fig_comp = px.bar(
        x=model_names, y=metric_values, color=model_names,
        color_discrete_sequence=["#2563eb", "#7c3aed", "#3b82f6", "#a855f7"],
        title=f"Model Comparison by {metric_options[metric_to_plot]}",
        labels={"x": "Model", "y": metric_options[metric_to_plot]}
    )
    fig_comp.update_layout(paper_bgcolor="#161b22", plot_bgcolor="#161b22", font_color="#fafafa", showlegend=False)
    st.plotly_chart(fig_comp, width="stretch")

# ==========================================
# PAGE 4: COST CALCULATOR
# ==========================================
elif page == "Cost Calculator":
    st.title("💰 Interactive Cost Calculator")
    st.markdown(
        "Adjust the illustrative costs of **False Negatives** (missed churners) and **False Positives** (wasted retention offers) "
        "to observe how business economics directly alter model total cost and profitability."
    )

    if not eval_data:
        st.error("Evaluation results not found. Please run `src/evaluation.py` first.")
        st.stop()

    col_c1, col_c2 = st.columns(2)
    with col_c1:
        fn_cost = st.number_input("False Negative Cost ($) [Missed Churn Opportunity]", min_value=0, max_value=2000, value=100, step=10)
    with col_c2:
        fp_cost = st.number_input("False Positive Cost ($) [Unnecessary Retention Offer]", min_value=0, max_value=500, value=10, step=5)

    recalculated_results = []
    for name, info in eval_data["models"].items():
        t = info.get("test", info)
        fn = int(t.get("fn", 0))
        fp = int(t.get("fp", 0))
        tp = int(t.get("tp", 0))
        tn = int(t.get("tn", 0))
        total_customers = fn + fp + tp + tn
        total_cost = (fn * fn_cost) + (fp * fp_cost)
        cost_per_cust = total_cost / total_customers if total_customers > 0 else 0.0
        recalculated_results.append({
            "Model": name,
            "False Negatives (FN)": fn,
            "False Positives (FP)": fp,
            "Total Cost ($)": total_cost,
            "Cost per Customer ($)": cost_per_cust
        })

    df_calc = pd.DataFrame(recalculated_results).sort_values("Total Cost ($)")

    st.markdown("### Recalculated Model Cost Rankings")
    st.dataframe(df_calc.style.format({"Total Cost ($)": "${:,.2f}", "Cost per Customer ($)": "${:,.2f}"}), width="stretch")

    fig_cost = px.bar(
        df_calc, x="Model", y="Total Cost ($)", color="Model",
        color_discrete_sequence=["#2563eb", "#7c3aed", "#3b82f6", "#a855f7"],
        title="Total Estimated Business Cost by Model (Under Current Cost Assumptions)"
    )
    fig_cost.update_layout(paper_bgcolor="#161b22", plot_bgcolor="#161b22", font_color="#fafafa", showlegend=False)
    st.plotly_chart(fig_cost, width="stretch")

    st.info(
        "💡 **Business Takeaway:** When the cost of a missed churner (FN) is 10x higher than an unnecessary offer (FP), "
        "models with high recall (even with lower precision) significantly outperform traditional high-precision models in total financial cost savings."
    )

# ==========================================
# PAGE 5: CUSTOMER INSIGHTS
# ==========================================
elif page == "Customer Insights":
    st.title("📈 Customer Churn Insights & Exploratory Data Analysis")
    st.markdown("Explore key demographic and service patterns influencing customer churn in the Telco dataset.")

    col1, col2 = st.columns(2)
    with col1:
        fig_payment = px.histogram(
            df, x="PaymentMethod", color="Churn", barmode="group",
            color_discrete_sequence=["#2563eb", "#7c3aed"],
            title="Churn by Payment Method"
        )
        fig_payment.update_layout(paper_bgcolor="#161b22", plot_bgcolor="#161b22", font_color="#fafafa")
        st.plotly_chart(fig_payment, width="stretch")

    with col2:
        fig_internet = px.histogram(
            df, x="InternetService", color="Churn", barmode="group",
            color_discrete_sequence=["#2563eb", "#7c3aed"],
            title="Churn by Internet Service Type"
        )
        fig_internet.update_layout(paper_bgcolor="#161b22", plot_bgcolor="#161b22", font_color="#fafafa")
        st.plotly_chart(fig_internet, width="stretch")

    col3, col4 = st.columns(2)
    with col3:
        fig_tech = px.histogram(
            df, x="TechSupport", color="Churn", barmode="group",
            color_discrete_sequence=["#2563eb", "#7c3aed"],
            title="Churn by Tech Support Availability"
        )
        fig_tech.update_layout(paper_bgcolor="#161b22", plot_bgcolor="#161b22", font_color="#fafafa")
        st.plotly_chart(fig_tech, width="stretch")

    with col4:
        fig_paper = px.histogram(
            df, x="PaperlessBilling", color="Churn", barmode="group",
            color_discrete_sequence=["#2563eb", "#7c3aed"],
            title="Churn by Paperless Billing"
        )
        fig_paper.update_layout(paper_bgcolor="#161b22", plot_bgcolor="#161b22", font_color="#fafafa")
        st.plotly_chart(fig_paper, width="stretch")

    st.markdown("---")
    st.subheader("Key Business Insights & Retention Recommendations")
    st.markdown("""
    1. **Contract Type is the Strongest Churn Predictor:** Month-to-month subscribers churn at over 40%, whereas one-year and two-year contract customers churn at less than 12% and 3% respectively. Incentivizing annual contracts directly decreases customer attrition.
    2. **Electronic Check Friction:** Customers paying via electronic check exhibit significantly higher churn rates than those using automated credit card or bank transfer payments.
    3. **Tech Support & Online Security Value:** Customers without Tech Support or Online Security churn at more than double the rate of customers with active add-on protections. Offering these services as value-added retention incentives provides high ROI.
    4. **Fiber Optic Expectations:** Fiber Optic users experience higher churn compared to DSL users, often due to higher monthly fees ($70–$100+) without bundled customer support.
    """)
