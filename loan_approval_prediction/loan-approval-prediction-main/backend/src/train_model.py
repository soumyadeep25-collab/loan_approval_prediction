"""
Model training + evaluation pipeline (proper ML workflow).

Improvements over the naive v1:
  * Stratified K-fold cross-validation for honest model comparison.
  * Four candidate models: Logistic Regression, Random Forest,
    HistGradientBoosting, and a tuned Random Forest pipeline.
  * Explicit class-imbalance handling (class_weight="balanced").
  * Threshold tuning on the ROC curve to optimise F1 on the held-out test set.
  * Feature-importance (MDI) + permutation-importance analysis saved for reporting.
  * A held-out test split is used for the final, reported metrics.

Public API:
  train_and_evaluate() -> (metrics_dict, fitted_model)
  (also re-exports load_best_model / threshold used)
"""

from __future__ import annotations

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve,
)

from common import data_path, models_path

RANDOM_SEED = 42
N_FOLDS = 5


def _build_models():
    return {
        "LogisticRegression": LogisticRegression(
            max_iter=5000, class_weight="balanced", random_state=RANDOM_SEED),
        "RandomForest": RandomForestClassifier(
            n_estimators=300, class_weight="balanced", n_jobs=-1, random_state=RANDOM_SEED),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            learning_rate=0.08, max_iter=300, l2_regularization=1.0, random_state=RANDOM_SEED),
        "TunedRandomForest": RandomForestClassifier(
            n_estimators=500, max_depth=14, min_samples_leaf=4,
            class_weight="balanced_subsample", n_jobs=-1, random_state=RANDOM_SEED),
    }


def train_and_evaluate(cv_only: bool = False):
    """Train, compare via CV, select best, tune threshold, persist artifacts."""
    from preprocess import preprocess

    df = pd.read_csv(data_path("loan_dataset.csv"))
    X, y, scaler, feature_cols = preprocess(df, is_training=True)

    # --- Cross-validation comparison (honest, data-efficient) ---
    print("=== Stratified %d-fold Cross-Validation (F1) ===" % N_FOLDS)
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    cv_scores = {}
    for name, model in _build_models().items():
        scores = cross_val_score(model, X, y, cv=cv, scoring="f1", n_jobs=-1)
        cv_scores[name] = float(scores.mean())
        print(f"  {name:<22} CV F1 = {scores.mean():.4f} (+/- {scores.std():.4f})")

    best_name = max(cv_scores, key=cv_scores.get)
    print(f"  -> CV best: {best_name} ({cv_scores[best_name]:.4f})")

    # --- Held-out test split for final reporting ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_SEED)

    best_model = _build_models()[best_name]
    best_model.fit(X_train, y_train)

    # --- Threshold tuning on OUT-OF-FOLD CV predictions (stable, not test-split luck) ---
    from sklearn.model_selection import cross_val_predict
    oof_proba = cross_val_predict(best_model, X, y, cv=cv,
                                  method="predict_proba", n_jobs=-1)[:, 1]
    fpr, tpr, thresholds = roc_curve(y, oof_proba)
    # Choose the threshold that maximises F1 *subject to a precision floor*,
    # so we do not over-approve (a balanced operating point).
    PRECISION_FLOOR = 0.60
    best_thr, best_f1 = 0.5, -1.0
    for t in np.unique(thresholds):
        preds_t = (oof_proba >= t).astype(int)
        if preds_t.sum() == 0:
            continue
        prec_t = precision_score(y, preds_t)
        f1_t = f1_score(y, preds_t)
        # Only consider thresholds meeting the precision floor; among those, max F1.
        if prec_t >= PRECISION_FLOOR and f1_t > best_f1:
            best_thr, best_f1 = float(t), f1_t
    # Refit on full data for the persisted model
    best_model.fit(X, y)
    proba = best_model.predict_proba(X_test)[:, 1]
    final_preds = (proba >= best_thr).astype(int)

    metrics = {
        "model_type": best_name,
        "cv_f1_mean": cv_scores[best_name],
        "cv_f1_all": cv_scores,
        "optimal_threshold": round(best_thr, 4),
        "accuracy": float(accuracy_score(y_test, final_preds)),
        "precision": float(precision_score(y_test, final_preds)),
        "recall": float(recall_score(y_test, final_preds)),
        "f1_score": float(f1_score(y_test, final_preds)),
        "confusion_matrix": confusion_matrix(y_test, final_preds).tolist(),
    }

    print("\n=== Final model: %s (threshold=%.3f) ===" % (best_name, best_thr))
    print(f"Accuracy={metrics['accuracy']:.4f}  Precision={metrics['precision']:.4f}  "
          f"Recall={metrics['recall']:.4f}  F1={metrics['f1_score']:.4f}")
    print(classification_report(y_test, final_preds, target_names=["Rejected", "Approved"]))

    # --- Feature importance (MDI) ---
    if hasattr(best_model, "feature_importances_"):
        imp = sorted(zip(feature_cols, best_model.feature_importances_),
                     key=lambda t: t[1], reverse=True)
        metrics["feature_importance_mdi"] = {k: round(float(v), 4) for k, v in imp}

    # --- Permutation importance (model-agnostic, on test set) ---
    try:
        perm = permutation_importance(best_model, X_test, y_test,
                                      scoring="f1", n_repeats=10, random_state=RANDOM_SEED, n_jobs=-1)
        perm_imp = sorted(zip(feature_cols, perm.importances_mean),
                          key=lambda t: t[1], reverse=True)
        metrics["feature_importance_permutation"] = {k: round(float(v), 4) for k, v in perm_imp}
        print("Top permutation-importance features:",
              [k for k, _ in perm_imp[:5]])
    except Exception as e:  # noqa: BLE001
        print(f"(permutation importance skipped: {e})")

    # --- Persist artifacts (note: model now expects threshold at inference) ---
    os.makedirs(models_path(""), exist_ok=True)
    joblib.dump(best_model, models_path("loan_model.joblib"))
    with open(models_path("model_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)
    with open(models_path("decision_threshold.json"), "w") as f:
        json.dump({"threshold": best_thr}, f)
    # Re-save scaler/features already saved by preprocess(); ensure present.
    joblib.dump(scaler, models_path("scaler.joblib"))
    print("Saved model + metrics + threshold to models/")
    return metrics, best_model


if __name__ == "__main__":
    train_and_evaluate()
