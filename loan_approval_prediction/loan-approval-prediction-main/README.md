# Loan Approval Prediction System

A Machine Learning internship deliverable (Infobharat Interns — IBI). Predicts
whether a loan application should be **Approved** or **Rejected** from applicant
financials, with a full, honest ML workflow: data generation → preprocessing →
EDA → visualization → model training/evaluation → interactive prediction.

Built with a **modular backend/frontend layout** (mirroring the `open_assets`
reference project):

```
backend/src/      generate_data, preprocess, eda, visualize,
                  train_model, evaluate, agent_graph, predict_core, common
frontend/src/     predict.py  (interactive console predictor)
frontend/app.py   Streamlit web demo
data/             loan_dataset.csv (2000 synthetic applicants)
models/           loan_model.joblib, scaler.joblib, feature_columns.json,
                  model_metrics.json, decision_threshold.json
images/           the four required visualizations (PNG)
notebooks/        Loan_Approval_Prediction.ipynb
```

> **Advanced layer:** the project also ships a **LangChain + LangGraph agentic
> pipeline** (`backend/src/agent_graph.py`) — see the *Advanced* section.

---

## 1. Dataset

Synthetic but realistic: **2000 applicant records** generated with
`backend/src/generate_data.py`. Columns match the assignment exactly:

`ApplicantID, Gender, Age, MonthlyIncome, LoanAmount, CreditScore,
EmploymentStatus, ExistingLoans, LoanTerm, PropertyArea, LoanApproved (target)`.

Design choices that make it a *genuine* ML problem (not trivially separable):

- Incomes are right-skewed (lognormal) — most people earn modestly, a few earn a lot.
- Loan amounts are derived from **loan-to-income** ratios (realistic ~0.2×–5× annual income).
- Approval is a **probabilistic, noisy** function of income, credit score, loan-to-income,
  employment, and existing debt — so the two classes **overlap** (≈50% approval rate).
- ~3–4% of values are missing to exercise imputation.

## 2. Libraries & tools

`Python 3.11`, `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn`,
`langchain`, `langgraph`, `openai` (LLM explainer), `streamlit` (web demo).
Coding was driven with **opencode** + the **freellmapi** free OpenAI-compatible
proxy (Gemini / Kimi models) and the Hermes agent directly.

## 3. Run it

```bash
# 1. set up environment (Python 3.11 venv)
python -m venv .venv && source .venv/bin/activate
pip install numpy pandas scikit-learn matplotlib seaborn langchain langgraph openai streamlit

# 2. full pipeline (data -> preprocess -> eda -> viz -> train -> evaluate)
python backend/src/generate_data.py
python backend/src/preprocess.py
python backend/src/eda.py
python backend/src/visualize.py
python backend/src/train_model.py
python backend/src/evaluate.py

# 3. interactive console predictor
python frontend/src/predict.py

# 4. (optional) Streamlit web demo
streamlit run frontend/app.py
```

## 4. Model & evaluation

`train_model.py` runs a **proper** ML workflow:

- **Stratified 5-fold cross-validation** to compare four candidates:
  Logistic Regression, Random Forest, HistGradientBoosting, Tuned Random Forest
  (all with `class_weight="balanced"` for the imbalanced-to-moderate target).
- Model selection by **CV F1**.
- **Threshold tuning** on out-of-fold CV predictions for a balanced operating point.
- **Feature importance**: permutation importance (model-agnostic) + MDI where available.
- Final metrics reported on a held-out 20% test split.

Representative results (seeded, reproducible):

| Candidate (CV F1) | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| LogisticRegression (selected) | 0.61 | 0.59 | 0.90 | 0.71 |
| Random Forest | 0.65 | 0.61 | 0.70 | 0.65 |

Top predictive features (permutation importance): **MonthlyIncome, CreditScore,
EmploymentStatus, LoanAmount, CreditDebtIndex**.

> The classes are intentionally overlapping, so ~0.65–0.71 F1 is the honest
> ceiling for this synthetic generator — a realistic result for lending risk.

Metrics and the chosen threshold are saved to `models/model_metrics.json` and
`models/decision_threshold.json`.

## 5. Visualizations (`images/`)

1. `approval_by_employment.png` — approval rate by employment status (bar).
2. `correlation_heatmap.png` — correlation heatmap of numeric features.
3. `income_vs_loan.png` — MonthlyIncome vs LoanAmount, colored by approval (scatter).
4. `approval_pie.png` — approved vs rejected distribution (pie).

## 6. Interactive predictor

`frontend/src/predict.py` prompts for Age, MonthlyIncome, LoanAmount, CreditScore,
EmploymentStatus, ExistingLoans, LoanTerm, PropertyArea and prints **Approved /
Rejected** with the model's confidence. Input is fully validated (type, range,
allowed values, abbreviations E/S/U and U/SU/R accepted, `q` to quit) so bad
input never crashes the program.

---

## 7. Advanced: LangGraph Agentic Pipeline

`backend/src/agent_graph.py` models the workflow as a **LangGraph `StateGraph`**
with a real *supervisor* decision node:

```
eda → train → evaluate → supervisor ──(action)──▶
        ├─ retrain            ──▶ evaluate (loop, capped)
        ├─ retune_threshold   ──▶ evaluate (loop, capped)
        ├─ try_different_model─▶ train ─▶ evaluate (loop, capped)
        └─ finish             ──▶ predict → explain → END
```

The supervisor receives the current evaluation metrics + feature importances and
**chooses the next action via LLM tool-calling** (freellmapi, OpenAI-compatible).
If the LLM is unavailable (rate-limited), it falls back to a transparent,
rule-based supervisor so the graph always runs and terminates (step cap).

The `explain` node uses a free LLM to narrate *why* a loan was approved/rejected,
again with a rule-based fallback.

Run it:

```bash
export FREELLMAPI_KEY="<freellmapi-client-key>"   # optional (enables LLM supervisor/explainer)
python backend/src/agent_graph.py --applicant '{"Age":29,"MonthlyIncome":8200,"LoanAmount":12000,"CreditScore":760,"EmploymentStatus":"Employed","ExistingLoans":0,"LoanTerm":120,"PropertyArea":"Urban"}'
```

---

## 8. LinkedIn / demo

Share: the four visualizations in `images/`, the model-performance table above,
the key insight that **income + credit score + employment status dominate approval**,
and the GitHub repo link. Mention the agentic LangGraph layer as a bonus.

**Repo:** https://github.com/anandxdj/loan-approval-prediction

## 9. Bonus / extensions

- ✅ Multiple models compared (CV).
- ✅ Feature-importance analysis (permutation + MDI).
- ✅ Streamlit web app.
- ☐ Deploy online, prediction reports (easy add-ons).
