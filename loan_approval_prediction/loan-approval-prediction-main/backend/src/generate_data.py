"""
Synthetic loan-application dataset generator (realistic, non-trivial).

Design goals (vs. a naive generator):
  * Incomes follow a right-skewed lognormal distribution (most people earn modestly,
    a few earn a lot) — realistic for a retail lending population.
  * Loan amounts are derived from annual income via a loan-to-income (LTI) ratio with
    noise, so they stay on a sensible scale (a $5k/mo earner does not request $170k).
  * Credit scores are realistic (300-850) and correlate weakly with income.
  * Approval is a *probabilistic, noisy* function of several risk factors, so the two
    classes overlap substantially — the classification problem is genuinely hard
    (~35-45% approval, not 95% separable). This exercises the models honestly.
  * A small, realistic fraction of values are missing (to exercise imputation).

Output: data/loan_dataset.csv  (columns exactly as required by the assignment)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

RANDOM_SEED = 42
N_RECORDS = 2000

# --------------------------------------------------------------------------- #
# Helper distributions
# --------------------------------------------------------------------------- #
def _lognormal(mean, sigma, size, rng):
    return rng.lognormal(mean=np.log(mean), sigma=sigma, size=size)


def generate(n: int = N_RECORDS, seed: int = RANDOM_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # --- Identity / demographics ---
    applicant_id = [f"APP{idx:05d}" for idx in range(1, n + 1)]
    gender = rng.choice(["Male", "Female"], size=n, p=[0.55, 0.45])
    age = rng.integers(21, 65, size=n)

    # --- Income: lognormal, median ~ $4,500/mo, long right tail ---
    monthly_income = np.round(_lognormal(4500, 0.55, n, rng), 2)
    monthly_income = np.clip(monthly_income, 1500, 60000)

    # --- Employment ---
    employment = rng.choice(
        ["Employed", "Self-Employed", "Unemployed"],
        size=n, p=[0.62, 0.26, 0.12],
    )
    # Unemployed people have near-zero reported income
    monthly_income = np.where(employment == "Unemployed",
                              np.round(_lognormal(900, 0.4, n, rng), 2),
                              monthly_income)
    monthly_income = np.clip(monthly_income, 500, 60000)

    # --- Credit score: realistic 300-850, weakly correlated with income ---
    income_z = (np.log(monthly_income) - np.log(monthly_income).mean()) / np.log(monthly_income).std()
    credit_base = 620 + 55 * income_z + rng.normal(0, 45, n)
    credit_score = np.clip(np.round(credit_base), 300, 850).astype(int)

    # --- Existing loans & term ---
    existing_loans = rng.poisson(lam=1.1, size=n)
    existing_loans = np.clip(existing_loans, 0, 8)
    loan_term = rng.choice([6, 12, 24, 36, 48, 60, 84, 120, 180, 240], size=n)

    # --- Property area ---
    property_area = rng.choice(
        ["Urban", "Semi-Urban", "Rural"], size=n, p=[0.45, 0.35, 0.20]
    )

    # --- Loan amount: loan-to-income ratio (annual income multiple), with noise ---
    annual_income = monthly_income * 12
    lti = np.clip(rng.normal(1.6, 0.7, n), 0.2, 5.0)  # requested loan ~ 0.2x-5x annual income
    loan_amount = np.round(annual_income * lti, 2)
    loan_amount = np.clip(loan_amount, 1000, 500000)

    # ------------------------------------------------------------------ #
    # Probabilistic, NOISY approval rule (the "true" but imperfect logic)
    # ------------------------------------------------------------------ #
    # Build a latent risk score in [0,1] from normalised factors.
    inc_score = np.clip((monthly_income - 1500) / (20000 - 1500), 0, 1)
    cred_score = np.clip((credit_score - 300) / (850 - 300), 0, 1)
    lti_score = np.clip(1 - (lti - 0.2) / (5.0 - 0.2), 0, 1)        # lower LTI = safer
    emp_score = np.where(employment == "Employed", 1.0,
                np.where(employment == "Self-Employed", 0.7, 0.15))
    debt_score = np.clip(1 - existing_loans / 8.0, 0, 1)

    risk = (0.32 * inc_score + 0.30 * cred_score + 0.16 * lti_score
            + 0.14 * emp_score + 0.08 * debt_score)
    # Add modest irreducible noise so classes overlap but remain learnable.
    risk = np.clip(risk + rng.normal(0, 0.06, n), 0, 1)
    prob_approve = 1.0 / (1.0 + np.exp(-(risk - 0.52) * 9.0))  # steeper, clearer boundary

    loan_approved = (rng.random(n) < prob_approve).astype(int)

    df = pd.DataFrame({
        "ApplicantID": applicant_id,
        "Gender": gender,
        "Age": age,
        "MonthlyIncome": monthly_income,
        "LoanAmount": loan_amount,
        "CreditScore": credit_score,
        "EmploymentStatus": employment,
        "ExistingLoans": existing_loans,
        "LoanTerm": loan_term,
        "PropertyArea": property_area,
        "LoanApproved": loan_approved,
    })

    # --- Inject realistic missingness (~3-4%) ---
    for col in ["MonthlyIncome", "CreditScore", "LoanAmount", "ExistingLoans"]:
        mask = rng.random(n) < 0.035
        df.loc[mask, col] = np.nan
    for col in ["EmploymentStatus", "PropertyArea", "Gender"]:
        mask = rng.random(n) < 0.02
        df.loc[mask, col] = np.nan

    return df


if __name__ == "__main__":
    from common import data_path
    import os
    os.makedirs("data", exist_ok=True)
    df = generate()
    df.to_csv(data_path("loan_dataset.csv"), index=False)
    print(f"Dataset saved to {data_path('loan_dataset.csv')}. Shape: {df.shape}")
    print(f"Approval rate: {df['LoanApproved'].mean():.3f}")
    print(df.head())
