"""Evaluate the saved model on a held-out split using the trained decision threshold."""

import os
import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

from common import data_path, models_path
from preprocess import preprocess

RANDOM_SEED = 42


def evaluate():
    df = pd.read_csv(data_path("loan_dataset.csv"))
    scaler = joblib.load(models_path("scaler.joblib"))
    model = joblib.load(models_path("loan_model.joblib"))
    with open(models_path("feature_columns.json")) as f:
        feature_cols = json.load(f)

    threshold = 0.5
    thr_path = models_path("decision_threshold.json")
    if os.path.exists(thr_path):
        with open(thr_path) as f:
            threshold = float(json.load(f).get("threshold", 0.5))

    X, y, _, _ = preprocess(df, is_training=False, scaler=scaler)
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_SEED)

    proba = model.predict_proba(X_test)[:, 1]
    preds = (proba >= threshold).astype(int)

    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds)
    rec = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    cm = confusion_matrix(y_test, preds)

    print("\n" + "=" * 50)
    print("EVALUATION (held-out test, threshold=%.3f)" % threshold)
    print("=" * 50)
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print("\nConfusion Matrix [Rejected, Approved]:")
    print(f"TN: {cm[0,0]:<5} FP: {cm[0,1]:<5}")
    print(f"FN: {cm[1,0]:<5} TP: {cm[1,1]:<5}")
    print(classification_report(y_test, preds, target_names=["Rejected", "Approved"], zero_division=0))


if __name__ == "__main__":
    evaluate()
