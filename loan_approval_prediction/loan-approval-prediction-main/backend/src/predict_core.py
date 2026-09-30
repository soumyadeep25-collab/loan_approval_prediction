"""
Shared loan-approval prediction core.

Extracted from the interactive CLI so both the console predictor (frontend/src/predict.py)
and the LangGraph agent (backend/src/agent_graph.py) reuse the exact same logic.
"""

import os
import json
import joblib
import pandas as pd

# backend/src contains preprocess/common; allow importing them.
sys_path_patch = os.path.join(os.path.dirname(__file__))
if sys_path_patch not in __import__("sys").path:
    __import__("sys").path.insert(0, sys_path_patch)

from common import models_path  # noqa: E402

NUMERIC_COLS = ["Age", "MonthlyIncome", "LoanAmount", "CreditScore", "ExistingLoans", "LoanTerm",
                "LoanToIncome", "DebtBurden", "CreditDebtIndex", "IncomePerTerm"]


def load_artifacts():
    """Load model, scaler and feature columns. Raises FileNotFoundError if missing."""
    model = joblib.load(models_path("loan_model.joblib"))
    scaler = joblib.load(models_path("scaler.joblib"))
    with open(models_path("feature_columns.json")) as f:
        feature_columns = json.load(f)
    return model, scaler, feature_columns


def predict_applicant(applicant: dict) -> dict:
    """
    Predict loan approval for a single applicant dict.

    applicant keys: Age, MonthlyIncome, LoanAmount, CreditScore,
    EmploymentStatus, ExistingLoans, LoanTerm, PropertyArea
    (Gender optional, defaults to Male).
    Returns {"status": "LOAN APPROVED"/"LOAN REJECTED", "probability": float}.
    Uses the F1-optimal decision threshold saved during training.
    """
    model, scaler, feature_columns = load_artifacts()

    # Load optimal threshold (falls back to 0.5 if absent).
    threshold = 0.5
    thr_path = models_path("decision_threshold.json")
    if os.path.exists(thr_path):
        with open(thr_path) as f:
            threshold = float(json.load(f).get("threshold", 0.5))

    gender_female = 0.0
    gender_male = 1.0
    if str(applicant.get("Gender", "Male")).strip().lower() == "female":
        gender_female, gender_male = 1.0, 0.0

    emp = applicant["EmploymentStatus"]
    area = applicant["PropertyArea"]

    age = float(applicant["Age"])
    income = float(applicant["MonthlyIncome"])
    loan = float(applicant["LoanAmount"])
    credit = float(applicant["CreditScore"])
    existing = float(applicant["ExistingLoans"])
    term = float(applicant["LoanTerm"])

    # Engineered features (must match backend/src/preprocess.py)
    annual_income = income * 12
    loan_to_income = loan / max(annual_income, 1.0)
    debt_burden = existing / max(income / 1000.0, 0.1)
    credit_debt_index = (existing + 1) * (850 - credit) / 100.0
    income_per_term = income / max(term, 1.0)

    row = {
        "Age": age,
        "MonthlyIncome": income,
        "LoanAmount": loan,
        "CreditScore": credit,
        "ExistingLoans": existing,
        "LoanTerm": term,
        "LoanToIncome": max(loan_to_income, 0.0),
        "DebtBurden": max(debt_burden, 0.0),
        "CreditDebtIndex": max(credit_debt_index, 0.0),
        "IncomePerTerm": max(income_per_term, 0.0),
        "Gender_Female": gender_female,
        "Gender_Male": gender_male,
        "EmploymentStatus_Employed": 1.0 if emp == "Employed" else 0.0,
        "EmploymentStatus_Self-Employed": 1.0 if emp == "Self-Employed" else 0.0,
        "EmploymentStatus_Unemployed": 1.0 if emp == "Unemployed" else 0.0,
        "PropertyArea_Rural": 1.0 if area == "Rural" else 0.0,
        "PropertyArea_Semi-Urban": 1.0 if area == "Semi-Urban" else 0.0,
        "PropertyArea_Urban": 1.0 if area == "Urban" else 0.0,
    }

    row_df = pd.DataFrame([row])
    row_df[NUMERIC_COLS] = scaler.transform(row_df[NUMERIC_COLS])
    row_df = row_df.reindex(columns=feature_columns, fill_value=0.0)

    prob = float(model.predict_proba(row_df)[0][1])
    pred = 1 if prob >= threshold else 0
    return {
        "status": "LOAN APPROVED" if pred == 1 else "LOAN REJECTED",
        "probability": prob,
    }
