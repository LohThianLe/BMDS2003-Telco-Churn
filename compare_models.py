"""
compare_models.py - merges results from all trained models into one table.

Run this AFTER model/logistic_regression.py, model/decision_tree.py, and
task4_xgboost/xgboost_model.py have all been run at least once (their .pkl
files must exist in outputs/). It reloads each saved model, re-scores it on
the correct test set (scaled for LR, unscaled for tree-based models), and
writes a single outputs/model_comparison.csv with all three side by side.

This replaces the old behaviour where decision_tree.py wrote a 2-model
comparison and xgboost_model.py wrote a separate, never-merged file.
"""

import os
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")

y_test = pd.read_csv(os.path.join(OUTPUTS_DIR, "clean_y_test.csv")).values.ravel()
X_test_unscaled = pd.read_csv(os.path.join(OUTPUTS_DIR, "clean_X_test.csv"))
X_test_scaled = pd.read_csv(os.path.join(OUTPUTS_DIR, "clean_X_test_scaled.csv"))

MODELS = {
    "Logistic Regression": ("logistic_regression_model.pkl", X_test_scaled),
    "Decision Tree": ("decision_tree_model.pkl", X_test_unscaled),
    "XGBoost": ("xgboost_model.pkl", X_test_unscaled),
}

rows = {"Metric": ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]}

for name, (filename, X_test) in MODELS.items():
    path = os.path.join(OUTPUTS_DIR, filename)
    if not os.path.exists(path):
        print(f"Skipping {name}: {path} not found (run its training script first).")
        continue

    model = joblib.load(path)
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    rows[name] = [
        round(accuracy_score(y_test, y_pred) * 100, 2),
        round(precision_score(y_test, y_pred) * 100, 2),
        round(recall_score(y_test, y_pred) * 100, 2),
        round(f1_score(y_test, y_pred) * 100, 2),
        round(roc_auc_score(y_test, y_pred_proba) * 100, 2),
    ]

comparison = pd.DataFrame(rows)
print(comparison.to_string(index=False))

out_path = os.path.join(OUTPUTS_DIR, "model_comparison.csv")
comparison.to_csv(out_path, index=False)
print(f"\nSaved combined comparison to {out_path}")
