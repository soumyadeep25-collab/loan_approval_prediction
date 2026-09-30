"""
Streamlit web app for the Loan Approval Prediction System.

A simple, shareable UI that loads the trained model + scaler and predicts
loan approval for applicant inputs, showing the decision and confidence.
Also exposes a "Run agentic pipeline" button that executes the LangGraph
agent (backend/src/agent_graph.py) to retrain/improve the model.

Run locally:
    streamlit run frontend/app.py
"""

import os
import sys
import subprocess

import streamlit as st

# Make backend/src importable (predict_core, common).
_BACKEND = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend", "src"))
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)

from predict_core import predict_applicant  # noqa: E402


st.set_page_config(page_title="Loan Approval Predictor", layout="centered")
st.title("🏦 Loan Approval Prediction System")
st.caption("Synthetic data · scikit-learn model · LangGraph agentic pipeline")

with st.sidebar:
    st.header("About")
    st.write(
        "This demo predicts whether a loan application would be approved based on "
        "applicant financials. The model and the agent behind it were built for the "
        "Infobharat ML Internship."
    )
    if st.button("🔄 Run agentic pipeline (retrain + improve)"):
        with st.spinner("Agent is working (train → evaluate → supervisor → …)…"):
            try:
                out = subprocess.run(
                    [sys.executable, os.path.join(_BACKEND, "agent_graph.py")],
                    capture_output=True, text=True, timeout=540,
                )
                st.success("Pipeline finished.")
                st.text_area("Agent trace (tail)", out.stdout[-2000:], height=200)
            except Exception as e:  # noqa: BLE001
                st.error(f"Pipeline failed: {e}")

st.subheader("Applicant details")
col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Age", 18, 100, 35)
    income = st.number_input("Monthly Income ($)", 500, 60000, 5000, step=500)
    loan = st.number_input("Loan Amount ($)", 1000, 500000, 15000, step=1000)
    credit = st.number_input("Credit Score", 300, 850, 720)
with col2:
    emp = st.selectbox("Employment Status", ["Employed", "Self-Employed", "Unemployed"])
    existing = st.number_input("Existing Loans", 0, 8, 1)
    term = st.number_input("Loan Term (months)", 6, 360, 120, step=6)
    area = st.selectbox("Property Area", ["Urban", "Semi-Urban", "Rural"])

if st.button("Predict", type="primary"):
    applicant = {
        "Age": age, "MonthlyIncome": income, "LoanAmount": loan,
        "CreditScore": credit, "EmploymentStatus": emp,
        "ExistingLoans": existing, "LoanTerm": term, "PropertyArea": area,
    }
    try:
        result = predict_applicant(applicant)
        if result["status"] == "LOAN APPROVED":
            st.success(f"✅ {result['status']}  (confidence {result['probability']*100:.1f}%)")
        else:
            st.error(f"❌ {result['status']}  (confidence {result['probability']*100:.1f}%)")
        st.progress(min(max(result["probability"], 0.0), 1.0))
    except Exception as e:  # noqa: BLE001
        st.exception(e)

st.markdown("---")
st.caption("Backend: backend/src · Model: models/loan_model.joblib · Built with opencode + freellmapi free models.")
