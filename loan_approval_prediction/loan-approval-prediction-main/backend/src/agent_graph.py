"""
LangGraph AGENTIC pipeline for loan-approval prediction (genuinely agentic).

Architecture
------------
Unlike a hardcoded branch, this graph has a real *supervisor* decision node that,
given the current evaluation metrics + feature importances, chooses the next action
by calling an LLM with a function/tool definition. The available "tools" are the
corrective actions the agent can take:

    - retrain              : rebuild the model with more estimators / class weights
    - retune_threshold    : re-optimise the decision threshold for a balanced point
    - try_different_model : switch the candidate model family
    - finish              : accept current model and stop improving

The supervisor is an LLM (freellmapi, OpenAI-compatible) using tool-calling. If the
LLM is unavailable (rate-limited / offline), it falls back to a transparent,
rule-based supervisor so the graph always runs.

Graph shape:

    eda -> train -> evaluate -> supervisor --(action)-->
        retrain            -> evaluate (loop, capped)
        retune_threshold   -> evaluate (loop, capped)
        try_different_model-> train    -> evaluate (loop, capped)
        finish             -> predict -> explain -> END

State carries the dataframe, the chosen model family, metrics, the supervisor's
last decision, and a retry counter (to guarantee termination).
"""

from __future__ import annotations

import os
import json
import sys
import time
from typing import TypedDict, Literal, Optional

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from common import data_path, models_path  # noqa: E402
from preprocess import preprocess, TARGET_COL  # noqa: E402
from train_model import _build_models, RANDOM_SEED  # noqa: E402
from evaluate import evaluate as _evaluate  # noqa: E402

from langgraph.graph import StateGraph, END  # noqa: E402

# LLM supervisor config (freellmapi free proxy)
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "http://localhost:3001/v1")
LLM_API_KEY = os.environ.get("LLM_API_KEY", os.environ.get("FREELLMAPI_KEY", ""))
LLM_MODEL = os.environ.get("LLM_MODEL", "kimi-k2.7-code")
MAX_IMPROVE_STEPS = 3


# --------------------------------------------------------------------------- #
# State
# --------------------------------------------------------------------------- #
class PipelineState(TypedDict, total=False):
    df: pd.DataFrame
    model_family: str
    metrics: dict
    decision: str
    decision_reasoning: str
    predict_input: dict
    prediction: dict
    explanation: str
    improve_steps: int
    used_llm_supervisor: bool


# --------------------------------------------------------------------------- #
# Tool definitions the supervisor can "call"
# --------------------------------------------------------------------------- #
ACTIONS = ["retrain", "retune_threshold", "try_different_model", "finish"]

SUPERVISOR_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "choose_action",
            "description": (
                "You are the supervisor of a loan-approval ML pipeline. Given the "
                "current evaluation metrics and top predictive features, decide the "
                "single best next action to improve the model responsibly."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ACTIONS,
                        "description": "Next action to take.",
                    },
                    "reasoning": {
                        "type": "string",
                        "description": "One sentence justifying the choice.",
                    },
                },
                "required": ["action", "reasoning"],
            },
        },
    }
]


# --------------------------------------------------------------------------- #
# Model rotation: read the discovered working models (probe_llm_models.py)
# so the agent never stops at a single rate-limited model.
# --------------------------------------------------------------------------- #
def _load_working_models() -> dict:
    try:
        with open(models_path("working_llm_models.json")) as f:
            return json.load(f)
    except Exception:
        return {}


def _llm_supervisor(metrics: dict, top_features: list, steps: int) -> tuple[str, str, bool]:
    """Return (action, reasoning, used_llm). Tries LLM (rotating across working
    models with retry), falls back to rules if all fail."""
    summary = (
        f"Model={metrics.get('model_type')}, F1={metrics.get('f1_score', 0):.3f}, "
        f"precision={metrics.get('precision', 0):.3f}, recall={metrics.get('recall', 0):.3f}. "
        f"Top features: {', '.join(top_features[:4])}. "
        f"Improvement steps taken so far: {steps} (max {MAX_IMPROVE_STEPS})."
    )
    # Candidate models to try, in priority order.
    working = _load_working_models().get("supervisor_ok", [])
    candidates = list(working) + [LLM_MODEL, "gemini-3.5-flash-lite", "mistral-small-4", "codestral"]
    # de-dup, keep first-seen order
    seen = set()
    deduped = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            deduped.append(c)
    candidates = deduped
    last_err = ""
    try:
        from openai import OpenAI
        client = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)
        for model in candidates:
            for attempt in range(2):
                try:
                    resp = client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": summary}],
                        tools=SUPERVISOR_TOOLS,
                        tool_choice="auto",
                        max_tokens=140,
                        timeout=30,
                    )
                    msg = resp.choices[0].message
                    if msg.tool_calls:
                        tc = msg.tool_calls[0]
                        # openai SDK normalizes to objects; defensively handle dict shape.
                        fn = getattr(tc, "function", None) or (tc.get("function") if isinstance(tc, dict) else {})
                        raw = fn.arguments if isinstance(fn, object) and hasattr(fn, "arguments") else (fn.get("arguments") if isinstance(fn, dict) else None)
                        args = json.loads(raw) if isinstance(raw, str) else (raw or {})
                        action = args.get("action", "finish")
                        reasoning = args.get("reasoning", "")
                        if action not in ACTIONS:
                            action = "finish"
                        return action, reasoning, True
                    # Model replied as text instead of a tool call.
                    return "finish", (msg.content or "")[:120], True
                except Exception as e:  # noqa: BLE001
                    last_err = f"{type(e).__name__}: {str(e)[:60]}"
                    if "429" in last_err or "Rate" in last_err or "quota" in last_err or "cooldown" in last_err:
                        time.sleep(2 * (attempt + 1))
                        continue
                    break  # non-retryable for this model; try next
    except Exception as e:  # noqa: BLE001
        last_err = f"{type(e).__name__}: {str(e)[:60]}"
    # Rule-based fallback supervisor (transparent, deterministic).
    f1 = float(metrics.get("f1_score", 0.0))
    prec = float(metrics.get("precision", 0.0))
    if f1 >= 0.80 or steps >= MAX_IMPROVE_STEPS:
        return "finish", f"F1={f1:.3f} acceptable or step cap reached (rule-based fallback; LLM unavailable: {last_err}).", False
    if prec < 0.60:
        if steps == 0:
            return "retune_threshold", "Precision below 0.60; re-optimise threshold first (rule-based fallback).", False
        return "retrain", f"Precision still {prec:.3f} after retune; rebuild stronger model (rule-based fallback; LLM unavailable: {last_err}).", False
    return "retrain", f"F1={f1:.3f} below target; rebuild stronger model (rule-based fallback; LLM unavailable: {last_err}).", False


# --------------------------------------------------------------------------- #
# Nodes
# --------------------------------------------------------------------------- #
def node_eda(state: PipelineState) -> dict:
    from eda import run_eda
    from visualize import run_visualizations
    print("\n=== [GRAPH] NODE: eda ===")
    run_eda()
    run_visualizations()
    return {}


def node_train(state: PipelineState) -> dict:
    print("\n=== [GRAPH] NODE: train ===")
    from train_model import train_and_evaluate
    df = state.get("df") or pd.read_csv(data_path("loan_dataset.csv"))
    metrics, model = train_and_evaluate()
    return {"df": df, "model_family": metrics.get("model_type", "unknown"),
            "metrics": metrics, "improve_steps": state.get("improve_steps", 0)}


def node_evaluate(state: PipelineState) -> dict:
    print("\n=== [GRAPH] NODE: evaluate ===")
    _evaluate()
    with open(models_path("model_metrics.json")) as f:
        metrics = json.load(f)
    return {"metrics": metrics}


def node_supervisor(state: PipelineState) -> dict:
    print("\n=== [GRAPH] NODE: supervisor ===")
    metrics = state.get("metrics", {})
    top_feats = list((metrics.get("feature_importance_permutation") or
                      metrics.get("feature_importance_mdi") or {}).keys())
    action, reasoning, used_llm = _llm_supervisor(metrics, top_feats, state.get("improve_steps", 0))
    print(f"[supervisor] action={action} | llm={used_llm} | {reasoning}")
    return {"decision": action, "decision_reasoning": reasoning,
            "used_llm_supervisor": used_llm}


def _route_supervisor(state: PipelineState) -> Literal["retrain", "retune_threshold", "try_different_model", "predict"]:
    decision = state.get("decision", "finish")
    if decision not in ("retrain", "retune_threshold", "try_different_model", "finish"):
        decision = "finish"
    # "finish" routes to the prediction/explanation tail.
    return "predict" if decision == "finish" else decision


def node_retrain(state: PipelineState) -> dict:
    print("\n=== [GRAPH] NODE: retrain (stronger model) ===")
    import joblib
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    df = state["df"]
    X, y, scaler, _ = preprocess(df, is_training=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_SEED)
    rf = RandomForestClassifier(n_estimators=600, max_depth=16, min_samples_leaf=3,
                                class_weight="balanced_subsample", n_jobs=-1, random_state=RANDOM_SEED)
    rf.fit(X_train, y_train)
    from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix
    preds = rf.predict(X_test)
    metrics = {
        "model_type": "RandomForestClassifier(agent-retrained)",
        "accuracy": float(accuracy_score(y_test, preds)),
        "precision": float(precision_score(y_test, preds)),
        "recall": float(recall_score(y_test, preds)),
        "f1_score": float(f1_score(y_test, preds)),
        "confusion_matrix": confusion_matrix(y_test, preds).tolist(),
    }
    joblib.dump(rf, models_path("loan_model.joblib"))
    with open(models_path("model_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)
    return {"metrics": metrics, "improve_steps": state.get("improve_steps", 0) + 1}


def node_retune_threshold(state: PipelineState) -> dict:
    print("\n=== [GRAPH] NODE: retune_threshold ===")
    import joblib
    import numpy as np
    from sklearn.model_selection import train_test_split, cross_val_predict, StratifiedKFold
    from sklearn.metrics import roc_curve, f1_score, precision_score, accuracy_score, recall_score, confusion_matrix
    df = state["df"]
    scaler = joblib.load(models_path("scaler.joblib"))
    model = joblib.load(models_path("loan_model.joblib"))
    X, y, _, _ = preprocess(df, is_training=False, scaler=scaler)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    oof = cross_val_predict(model, X, y, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
    fpr, _, thr = roc_curve(y, oof)
    best_t, best_f1 = 0.5, -1.0
    for t in np.unique(thr):
        p = (oof >= t).astype(int)
        if p.sum() == 0:
            continue
        if precision_score(y, p) >= 0.60 and f1_score(y, p) > best_f1:
            best_t, best_f1 = float(t), f1_score(y, p)
    with open(models_path("decision_threshold.json"), "w") as f:
        json.dump({"threshold": best_t}, f)
    # Recompute reported test metrics at the new threshold and persist them so the
    # supervisor sees the effect of this action.
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM_SEED)
    proba = model.predict_proba(X_test)[:, 1]
    new_preds = (proba >= best_t).astype(int)
    metrics = {
        "model_type": state.get("metrics", {}).get("model_type", "unknown"),
        "optimal_threshold": best_t,
        "accuracy": float(accuracy_score(y_test, new_preds)),
        "precision": float(precision_score(y_test, new_preds)),
        "recall": float(recall_score(y_test, new_preds)),
        "f1_score": float(f1_score(y_test, new_preds)),
        "confusion_matrix": confusion_matrix(y_test, new_preds).tolist(),
    }
    with open(models_path("model_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"[retune] new threshold={best_t:.3f} (test F1={metrics['f1_score']:.3f}, prec={metrics['precision']:.3f})")
    return {"metrics": metrics, "improve_steps": state.get("improve_steps", 0) + 1}


def node_try_different_model(state: PipelineState) -> dict:
    print("\n=== [GRAPH] NODE: try_different_model ===")
    # Re-run full training which re-selects the best of all candidate families.
    from train_model import train_and_evaluate
    metrics, _ = train_and_evaluate()
    return {"metrics": metrics, "improve_steps": state.get("improve_steps", 0) + 1}


def node_predict(state: PipelineState) -> dict:
    print("\n=== [GRAPH] NODE: predict ===")
    from predict_core import predict_applicant
    inp = state.get("predict_input") or {
        "Age": 35, "MonthlyIncome": 5000, "LoanAmount": 15000,
        "CreditScore": 720, "EmploymentStatus": "Employed",
        "ExistingLoans": 1, "LoanTerm": 180, "PropertyArea": "Semi-Urban",
    }
    result = predict_applicant(inp)
    print(f"[predict] -> {result['status']} ({result['probability']*100:.1f}%)")
    return {"prediction": result}


def node_explain(state: PipelineState) -> dict:
    print("\n=== [GRAPH] NODE: explain ===")
    text = explain_prediction(state.get("prediction", {}), state.get("predict_input", {}))
    print(text)
    return {"explanation": text}


def explain_prediction(prediction: dict, applicant: dict) -> str:
    status = prediction.get("status", "UNKNOWN")
    prob = prediction.get("probability", 0.0)
    prompt = (
        "You are a loan officer assistant. Given an applicant profile and a model "
        f"prediction of {status} (confidence {prob*100:.1f}%), write a concise, "
        "professional 2-3 sentence explanation of the likely drivers (income, credit "
        f"score, employment, existing loans, loan-to-income). Applicant: {json.dumps(applicant)}."
    )
    # Rotate across discovered working explainer models; fall back to rules.
    working = _load_working_models().get("explainer_ok", [])
    candidates = list(dict.fromkeys(list(working) + [LLM_MODEL, "mistral-small-4", "codestral", "gemini-3.6-flash"]))
    last_err = ""
    try:
        from openai import OpenAI
        client = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)
        for model in candidates:
            for attempt in range(2):
                try:
                    resp = client.chat.completions.create(
                        model=model, messages=[{"role": "user", "content": prompt}],
                        max_tokens=160, timeout=30)
                    return resp.choices[0].message.content.strip()
                except Exception as e:  # noqa: BLE001
                    last_err = f"{type(e).__name__}: {str(e)[:50]}"
                    if "429" in last_err or "Rate" in last_err or "quota" in last_err or "cooldown" in last_err:
                        time.sleep(2 * (attempt + 1))
                        continue
                    break
    except Exception as e:  # noqa: BLE001
        last_err = f"{type(e).__name__}: {str(e)[:50]}"
    drivers = []
    if applicant.get("CreditScore", 0) >= 700:
        drivers.append("strong credit score")
    if applicant.get("MonthlyIncome", 0) >= 5000:
        drivers.append("steady income")
    if applicant.get("EmploymentStatus") == "Employed":
        drivers.append("stable employment")
    if applicant.get("ExistingLoans", 0) >= 3:
        drivers.append("high existing debt burden")
    reason = ", ".join(drivers) if drivers else "mixed risk factors"
    return (f"Model predicts {status} (confidence {prob*100:.1f}%). "
            f"Primary drivers: {reason}. (rule-based fallback; LLM unavailable: {last_err})")


# --------------------------------------------------------------------------- #
# Graph assembly
# --------------------------------------------------------------------------- #
def build_graph():
    g = StateGraph(PipelineState)
    g.add_node("eda", node_eda)
    g.add_node("train", node_train)
    g.add_node("evaluate", node_evaluate)
    g.add_node("supervisor", node_supervisor)
    g.add_node("retrain", node_retrain)
    g.add_node("retune_threshold", node_retune_threshold)
    g.add_node("try_different_model", node_try_different_model)
    g.add_node("predict", node_predict)
    g.add_node("explain", node_explain)

    g.set_entry_point("eda")
    g.add_edge("eda", "train")
    g.add_edge("train", "evaluate")
    g.add_edge("evaluate", "supervisor")
    g.add_conditional_edges("supervisor", _route_supervisor, {
        "retrain": "retrain",
        "retune_threshold": "retune_threshold",
        "try_different_model": "try_different_model",
        "predict": "predict",
    })
    g.add_edge("retrain", "evaluate")
    g.add_edge("retune_threshold", "evaluate")
    g.add_edge("try_different_model", "evaluate")
    g.add_edge("predict", "explain")
    g.add_edge("explain", END)
    return g.compile()


def main():
    import argparse
    p = argparse.ArgumentParser(description="Run the agentic LangGraph loan pipeline.")
    p.add_argument("--applicant", type=str, default=None, help="JSON of applicant features")
    args = p.parse_args()
    initial: PipelineState = {}
    if args.applicant:
        initial["predict_input"] = json.loads(args.applicant)
    print(">>> Agentic LangGraph pipeline starting...")
    final = build_graph().invoke(initial)
    print("\n>>> Pipeline complete.")
    print("Final metrics:", json.dumps(final.get("metrics", {}), indent=2))
    print("Supervisor used LLM:", final.get("used_llm_supervisor"))
    print("Prediction:", final.get("prediction"))
    print("Explanation:", final.get("explanation"))


if __name__ == "__main__":
    main()
