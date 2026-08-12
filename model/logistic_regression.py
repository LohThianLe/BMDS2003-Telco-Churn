"""
model/logistic_regression.py - Task 1: Logistic Regression Baseline Model

Why Logistic Regression as the baseline:
Logistic Regression is used as the baseline model because it is a simple,
interpretable linear classifier that provides a clear reference point against
which the more complex models (Decision Tree/KNN, Random Forest, XGBoost) can
be evaluated. Its coefficients are directly interpretable as log-odds, which is
valuable for explaining churn drivers to business stakeholders, and it trains
quickly even with the one-hot-encoded categorical features produced by
preprocessing.py.

Reference:
James, G., Witten, D., Hastie, T., & Tibshirani, R. (2013). An Introduction
to Statistical Learning: with Applications in R (Chapter 4: Classification).
Springer.

"""

import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

# Resolve project directories dynamically
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
CHARTS_DIR = os.path.join(OUTPUTS_DIR, "charts")

# 1. Load the preprocessed SCALED dataset
X_train = pd.read_csv(os.path.join(OUTPUTS_DIR, "clean_X_train_scaled.csv"))
X_test = pd.read_csv(os.path.join(OUTPUTS_DIR, "clean_X_test_scaled.csv"))
y_train = pd.read_csv(os.path.join(OUTPUTS_DIR, "clean_y_train.csv")).values.ravel()
y_test = pd.read_csv(os.path.join(OUTPUTS_DIR, "clean_y_test.csv")).values.ravel()

# 2. Initialize and train Logistic Regression model
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# 3. Make predictions
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

# 4. Calculate evaluation metrics
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

print("=" * 40)
print("   LOGISTIC REGRESSION MODEL RESULTS   ")
print("=" * 40)
print(f"Accuracy  : {acc * 100:.2f}%")
print(f"Precision : {prec * 100:.2f}%")
print(f"Recall    : {rec * 100:.2f}%")
print(f"F1-Score  : {f1 * 100:.2f}%")
print(f"ROC-AUC   : {roc_auc * 100:.2f}%")
print("=" * 40)

# 5. Plot and save Confusion Matrix
os.makedirs(CHARTS_DIR, exist_ok=True)
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Churn", "Churn"])

fig, ax = plt.subplots(figsize=(6, 5))
disp.plot(cmap="Blues", ax=ax)
plt.title("Logistic Regression Confusion Matrix")
plt.tight_layout()
plt.savefig(os.path.join(CHARTS_DIR, "logistic_regression_cm.png"), dpi=300)
plt.close()

# 6. Save trained model file for Task 3 / Streamlit integration
model_path = os.path.join(OUTPUTS_DIR, "logistic_regression_model.pkl")
joblib.dump(model, model_path)
print(f"Model saved to {model_path}")