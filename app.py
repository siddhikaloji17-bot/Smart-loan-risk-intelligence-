"""
Smart Loan Default & Financial Risk Intelligence System — Streamlit Dashboard

Run with: streamlit run app.py
"""

import joblib
import numpy as np
import pandas as pd
import shap
import streamlit as st
import matplotlib.pyplot as plt
from io import BytesIO
from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

MODEL_DIR = Path(__file__).resolve().parent / "models"

st.set_page_config(page_title="Loan Risk Intelligence", layout="wide")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

    :root {
        --ink: #f4f7fb;
        --muted: #96a4bd;
        --canvas: #0b1020;
        --panel: #111a2f;
        --line: #263452;
        --cyan: #5de1e6;
        --coral: #ff6b6b;
    }

    .stApp {
        background: radial-gradient(circle at 85% 0%, #172748 0, var(--canvas) 38rem);
        color: var(--ink);
    }

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    h1 {
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 700;
        letter-spacing: 0;
        font-size: 2.65rem;
        line-height: 1.05;
    }

    h2, h3, label, .stMetric label {
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 600;
        letter-spacing: 0;
    }

    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stSidebar"] {
        background: #0e1629;
        border-right: 1px solid var(--line);
    }
    [data-testid="stSidebar"] h2 { color: var(--cyan); }
    [data-testid="stMetric"] {
        background: rgba(17, 26, 47, 0.88);
        border: 1px solid var(--line);
        border-radius: 12px;
        padding: 1rem 1.1rem;
    }
    [data-testid="stMetricValue"] {
        color: var(--ink);
        font-family: 'Space Grotesk', sans-serif;
    }
    .stButton > button, .stDownloadButton > button {
        border: 1px solid #3a527b;
        border-radius: 8px;
        background: #172748;
        color: var(--ink);
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 600;
    }
    .stButton > button:hover, .stDownloadButton > button:hover {
        border-color: var(--cyan);
        color: var(--cyan);
    }
    [data-testid="stFileUploader"] {
        background: rgba(17, 26, 47, 0.7);
        border: 1px dashed #3a527b;
        border-radius: 12px;
        padding: 0.5rem;
    }
    hr { border-color: var(--line); }
    .brand-kicker {
        color: var(--cyan);
        font-family: 'Space Grotesk', sans-serif;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        margin-bottom: 0.35rem;
    }
    .brand-subtitle { color: var(--muted); font-size: 1.05rem; margin-top: -0.5rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_DIR / "best_model.pkl")
    preprocessor = joblib.load(MODEL_DIR / "preprocessor.pkl")
    meta = joblib.load(MODEL_DIR / "metadata.pkl")
    comparison_models_path = MODEL_DIR / "comparison_models.pkl"
    comparison_metrics_path = MODEL_DIR / "comparison_metrics.pkl"
    comparison_models = joblib.load(comparison_models_path) if comparison_models_path.exists() else {}
    comparison_metrics = joblib.load(comparison_metrics_path) if comparison_metrics_path.exists() else {}
    return model, preprocessor, meta, comparison_models, comparison_metrics


model, preprocessor, meta, comparison_models, comparison_metrics = load_artifacts()


def add_derived_features(data):
    data = data.copy()
    data["debt_to_income"] = (
        data["loan_amount"] * 0.15 + data["existing_loans"] * 3000
    ) / data["annual_income"].replace(0, np.nan)
    data["loan_to_income"] = data["loan_amount"] / data["annual_income"].replace(0, np.nan)
    data["income_band"] = pd.cut(
        data["annual_income"], bins=[0, 30000, 60000, 100000, 200000, np.inf],
        labels=["very_low", "low", "medium", "high", "very_high"]
    )
    data["age_group"] = pd.cut(
        data["age"], bins=[0, 25, 35, 45, 55, 100],
        labels=["18-25", "26-35", "36-45", "46-55", "56+"]
    )
    return data


def score_applicants(data, scoring_model=model):
    prepared = add_derived_features(data)
    feature_cols = meta["numeric_features"] + meta["categorical_features"]
    processed = preprocessor.transform(prepared[feature_cols])
    return scoring_model.predict_proba(processed)[:, 1], prepared, processed


def risk_band(score):
    if score < 30:
        return "Low Risk"
    if score < 60:
        return "Medium Risk"
    return "High Risk"


def monthly_emi(principal, annual_rate, years=3):
    monthly_rate = annual_rate / 1200
    months = years * 12
    if monthly_rate == 0:
        return principal / months
    return principal * monthly_rate * (1 + monthly_rate) ** months / (
        (1 + monthly_rate) ** months - 1
    )


def create_pdf_report(applicant, score, band, model_name):
    buffer = BytesIO()
    report = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    report.setFont("Helvetica-Bold", 20)
    report.drawString(48, height - 60, "Smart Loan Risk Assessment")
    report.setFont("Helvetica", 10)
    report.drawString(48, height - 82, f"Model: {model_name}")
    report.setFont("Helvetica-Bold", 16)
    report.drawString(48, height - 125, f"Risk Score: {score:.1f} / 100")
    report.drawString(48, height - 150, f"Risk Band: {band}")
    report.setFont("Helvetica-Bold", 12)
    report.drawString(48, height - 195, "Applicant details")
    report.setFont("Helvetica", 10)
    y = height - 215
    for key, value in applicant.items():
        report.drawString(60, y, f"{key.replace('_', ' ').title()}: {value}")
        y -= 16
        if y < 55:
            report.showPage()
            y = height - 55
    report.save()
    buffer.seek(0)
    return buffer.getvalue()

st.markdown(
    f"""
    <div class="brand-kicker">Credit intelligence platform · v1.0</div>
    <h1>Smart Loan Risk Intelligence</h1>
    <div class="brand-subtitle">Explainable default prediction for faster, more confident lending decisions.</div>
    """,
    unsafe_allow_html=True,
)
st.caption(f"Production model: **{meta['best_model_name']}** · Explainability: **SHAP** · Decision score: **0–100**")

st.sidebar.header("Applicant Details")

age = st.sidebar.slider("Age", 21, 70, 32)
annual_income = st.sidebar.number_input("Annual Income (₹)", 15000, 500000, 60000, step=1000)
employment_type = st.sidebar.selectbox(
    "Employment Type", ["salaried", "self_employed", "unemployed", "retired"]
)
employment_length_yrs = st.sidebar.slider("Employment Length (yrs)", 0, 40, 4)
credit_history_length_yrs = st.sidebar.slider("Credit History Length (yrs)", 0, 40, 6)
loan_amount = st.sidebar.number_input("Loan Amount (₹)", 1000, 50000, 15000, step=500)
loan_purpose = st.sidebar.selectbox(
    "Loan Purpose",
    ["debt_consolidation", "credit_card", "home_improvement",
     "small_business", "medical", "car", "education"]
)
interest_rate = st.sidebar.slider("Interest Rate (%)", 5.0, 32.0, 14.0)
existing_loans = st.sidebar.slider("Existing Loans", 0, 8, 1)
credit_utilization = st.sidebar.slider("Credit Utilization", 0.0, 1.0, 0.35)
num_late_payments_2yr = st.sidebar.slider("Late Payments (last 2 yrs)", 0, 10, 0)
has_bankruptcy = st.sidebar.selectbox("Bankruptcy History", ["No", "Yes"]) == "Yes"

# Derived features
debt_to_income = round((loan_amount * 0.15 + existing_loans * 3000) / annual_income, 3)
loan_to_income = round(loan_amount / annual_income, 3)
income_band = pd.cut(
    [annual_income], bins=[0, 30000, 60000, 100000, 200000, np.inf],
    labels=["very_low", "low", "medium", "high", "very_high"]
)[0]
age_group = pd.cut(
    [age], bins=[0, 25, 35, 45, 55, 100],
    labels=["18-25", "26-35", "36-45", "46-55", "56+"]
)[0]

input_df = pd.DataFrame([{
    "age": age,
    "annual_income": annual_income,
    "employment_length_yrs": employment_length_yrs,
    "credit_history_length_yrs": credit_history_length_yrs,
    "loan_amount": loan_amount,
    "interest_rate": interest_rate,
    "existing_loans": existing_loans,
    "credit_utilization": credit_utilization,
    "num_late_payments_2yr": num_late_payments_2yr,
    "has_bankruptcy": int(has_bankruptcy),
    "debt_to_income": debt_to_income,
    "loan_to_income": loan_to_income,
    "employment_type": employment_type,
    "loan_purpose": loan_purpose,
    "income_band": income_band,
    "age_group": age_group,
}])

feature_cols = meta["numeric_features"] + meta["categorical_features"]
X_input = input_df[feature_cols]
X_proc = preprocessor.transform(X_input)

proba_default = model.predict_proba(X_proc)[0, 1]
risk_score = round(proba_default * 100, 1)

if risk_score < 30:
    band, color = "Low Risk", "green"
elif risk_score < 60:
    band, color = "Medium Risk", "orange"
else:
    band, color = "High Risk", "red"

emi = monthly_emi(loan_amount, interest_rate)
emi_to_income = emi / (annual_income / 12)
if band == "High Risk" or emi_to_income > 0.4:
    lending_action = "Manual review recommended"
elif band == "Medium Risk":
    lending_action = "Approve with conditions"
else:
    lending_action = "Eligible for standard review"

col1, col2, col3 = st.columns(3)
col1.metric("Risk Score (0-100)", f"{risk_score}")
col2.markdown(f"### Risk Band: :{color}[{band}]")
auto_flag = "🚩 Auto-Review Flag" if (risk_score > 80 and debt_to_income > 0.5) else "✅ No Auto-Flag"
col3.markdown(f"### {auto_flag}")

st.divider()
st.subheader("Affordability snapshot")
affordability_col1, affordability_col2, affordability_col3 = st.columns(3)
affordability_col1.metric("Estimated monthly EMI", f"₹{emi:,.0f}")
affordability_col2.metric("EMI / monthly income", f"{emi_to_income:.1%}")
affordability_col3.metric("Recommended action", lending_action)
if emi_to_income > 0.4:
    st.warning("The estimated EMI exceeds 40% of monthly income. Consider a smaller loan or longer tenure.")
else:
    st.success("The estimated EMI is within the 40% affordability guideline.")

st.divider()
st.subheader("Why this score? (SHAP Explanation)")
st.caption("Top factors influencing the predicted default risk")

cat_encoder = preprocessor.named_transformers_["cat"].named_steps["onehot"]
cat_feature_names = cat_encoder.get_feature_names_out(meta["categorical_features"])
all_feature_names = meta["numeric_features"] + list(cat_feature_names)

X_proc_dense = X_proc.toarray() if hasattr(X_proc, "toarray") else X_proc
X_proc_df = pd.DataFrame(X_proc_dense, columns=all_feature_names)

if meta["best_model_name"] == "Logistic Regression":
    background = np.zeros((1, X_proc_df.shape[1]))
    explainer = shap.LinearExplainer(model, background, feature_names=all_feature_names)
else:
    explainer = shap.TreeExplainer(model)

shap_values = explainer.shap_values(X_proc_df)
if isinstance(shap_values, list):
    shap_values = shap_values[1]

contributions = pd.DataFrame({
    "feature": all_feature_names,
    "contribution": shap_values[0],
})
contributions["absolute"] = contributions["contribution"].abs()
top_contributions = contributions.nlargest(10, "absolute").sort_values("contribution")
bar_colors = ["#168aad" if value < 0 else "#e63946"
              for value in top_contributions["contribution"]]

fig, ax = plt.subplots(figsize=(11, 5.5), facecolor="#101426")
ax.set_facecolor("#101426")
bars = ax.bar(top_contributions["feature"], top_contributions["contribution"], color=bar_colors)
ax.axhline(0, color="#71809f", linewidth=0.8)
ax.set_title("SHAP Feature Contribution", loc="left", color="#e6e9f2", fontsize=16, fontweight="bold")
ax.set_ylabel("Impact on model output", color="#9ba7c0")
ax.set_xlabel("Applicant features", color="#9ba7c0")
ax.tick_params(axis="x", labelrotation=35, colors="#9ba7c0")
ax.tick_params(axis="y", colors="#9ba7c0")
ax.grid(axis="y", linestyle="--", color="#29334d", alpha=0.8)
for spine in ax.spines.values():
    spine.set_color("#29334d")
for bar, value in zip(bars, top_contributions["contribution"]):
    ax.annotate(
        f"{value:+.2f}",
        xy=(bar.get_x() + bar.get_width() / 2, value),
        xytext=(0, 4 if value >= 0 else -14),
        textcoords="offset points",
        ha="center",
        va="bottom" if value >= 0 else "top",
        fontsize=9,
        color="#e6e9f2",
    )
fig.tight_layout()
with st.container(border=True):
    st.pyplot(fig)
plt.close(fig)

st.divider()
st.subheader("What-if analysis")
st.caption("Adjust one applicant factor to see how the predicted risk changes.")
what_if_factor = st.selectbox(
    "Factor to adjust",
    ["loan_amount", "annual_income", "interest_rate", "credit_utilization",
     "num_late_payments_2yr", "existing_loans"],
    format_func=lambda value: value.replace("_", " ").title(),
)
what_if_limits = {
    "loan_amount": (1000, 50000, loan_amount, 500),
    "annual_income": (15000, 500000, annual_income, 1000),
    "interest_rate": (5.0, 32.0, interest_rate, 0.5),
    "credit_utilization": (0.0, 1.0, credit_utilization, 0.01),
    "num_late_payments_2yr": (0, 10, num_late_payments_2yr, 1),
    "existing_loans": (0, 8, existing_loans, 1),
}
what_if_min, what_if_max, what_if_default, what_if_step = what_if_limits[what_if_factor]
what_if_value = st.slider(
    "What-if value", min_value=what_if_min, max_value=what_if_max,
    value=what_if_default, step=what_if_step,
)
what_if_input = input_df.copy()
what_if_input[what_if_factor] = what_if_value
what_if_probability, _, _ = score_applicants(what_if_input)
what_if_score = round(what_if_probability[0] * 100, 1)
delta = round(what_if_score - risk_score, 1)
what_if_col1, what_if_col2 = st.columns(2)
what_if_col1.metric("What-if risk score", f"{what_if_score:.1f}", f"{delta:+.1f} points")
what_if_col2.metric("What-if risk band", risk_band(what_if_score))

st.divider()
st.subheader("Applicant Summary")
st.dataframe(input_df.T.rename(columns={0: "Value"}))

pdf_bytes = create_pdf_report(
    input_df.iloc[0].to_dict(), risk_score, risk_band(risk_score), meta["best_model_name"]
)
st.download_button(
    "Download PDF risk report", data=pdf_bytes,
    file_name="loan_risk_report.pdf", mime="application/pdf",
)

st.divider()
batch_tab, comparison_tab = st.tabs(["Batch scoring", "Model comparison"])

with batch_tab:
    st.subheader("Score many applicants")
    st.caption("Upload a CSV containing the applicant columns used by the dashboard.")
    template_columns = [
        "age", "annual_income", "employment_type", "employment_length_yrs",
        "credit_history_length_yrs", "loan_amount", "loan_purpose", "interest_rate",
        "existing_loans", "credit_utilization", "num_late_payments_2yr", "has_bankruptcy",
    ]
    template = pd.DataFrame(columns=template_columns)
    st.download_button(
        "Download CSV template", data=template.to_csv(index=False).encode("utf-8"),
        file_name="loan_applicants_template.csv", mime="text/csv",
    )
    uploaded_file = st.file_uploader("Applicant CSV", type="csv")
    if uploaded_file is not None:
        batch_input = pd.read_csv(uploaded_file)
        required_columns = [
            "age", "annual_income", "employment_type", "employment_length_yrs",
            "credit_history_length_yrs", "loan_amount", "loan_purpose", "interest_rate",
            "existing_loans", "credit_utilization", "num_late_payments_2yr", "has_bankruptcy",
        ]
        missing_columns = sorted(set(required_columns) - set(batch_input.columns))
        if missing_columns:
            st.error(f"Missing required columns: {', '.join(missing_columns)}")
        else:
            if batch_input["has_bankruptcy"].dtype == "object":
                batch_input["has_bankruptcy"] = batch_input["has_bankruptcy"].astype(str).str.lower().isin(
                    ["yes", "true", "1"]
                ).astype(int)
            batch_probabilities, _, _ = score_applicants(batch_input)
            batch_results = batch_input.copy()
            batch_results["risk_score"] = np.round(batch_probabilities * 100, 1)
            batch_results["risk_band"] = batch_results["risk_score"].map(risk_band)
            batch_results["estimated_monthly_emi"] = np.round(
                [monthly_emi(loan, rate) for loan, rate in zip(
                    batch_results["loan_amount"], batch_results["interest_rate"]
                )], 0
            )
            batch_results["recommended_action"] = batch_results.apply(
                lambda row: "Manual review recommended" if row["risk_band"] == "High Risk"
                else "Approve with conditions" if row["risk_band"] == "Medium Risk"
                else "Eligible for standard review", axis=1
            )
            st.dataframe(batch_results, use_container_width=True)
            summary_col1, summary_col2, summary_col3 = st.columns(3)
            summary_col1.metric("Applicants scored", len(batch_results))
            summary_col2.metric("High-risk applicants", int((batch_results["risk_band"] == "High Risk").sum()))
            summary_col3.metric("Average risk score", f"{batch_results['risk_score'].mean():.1f}")
            st.download_button(
                "Download scored CSV", data=batch_results.to_csv(index=False).encode("utf-8"),
                file_name="scored_loan_applicants.csv", mime="text/csv",
            )

with comparison_tab:
    st.subheader("LR vs Random Forest vs XGBoost")
    if not comparison_models:
        st.warning("Comparison models are not available yet. Run train_model.py once to create them.")
    else:
        comparison_rows = []
        for name, comparison_model in comparison_models.items():
            probability = comparison_model.predict_proba(X_proc)[0, 1]
            comparison_rows.append({
                "Model": name,
                "Current applicant risk": round(probability * 100, 1),
                "Test ROC-AUC": round(comparison_metrics.get(name, {}).get("auc", 0), 3),
                "Test F1": round(comparison_metrics.get(name, {}).get("f1", 0), 3),
            })
        comparison_df = pd.DataFrame(comparison_rows).sort_values("Test ROC-AUC", ascending=False)
        st.dataframe(comparison_df, hide_index=True, use_container_width=True)
        comparison_chart = comparison_df.set_index("Model")["Current applicant risk"]
        st.bar_chart(comparison_chart, y_label="Predicted default risk (0-100)")
