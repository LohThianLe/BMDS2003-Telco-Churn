"""
model/decision_tree.py - Task 2: Decision Tree Model

Why Decision Tree was chosen:
Decision Tree was selected as the second model because it captures non-linear
relationships and feature interactions that a linear model like Logistic
Regression cannot (e.g. tenure only mattering within certain contract types).
It requires no feature scaling, handles the mixed binary/one-hot encoded
features produced by preprocessing.py natively, and produces an explicit,
visualisable set of decision rules that can be directly inspected to explain
churn drivers to non-technical stakeholders.


Note: class_weight='balanced' is used to account for the class imbalance in
Churn (~73% No / ~27% Yes). Task 1's Logistic Regression was trained without
this setting, so this is a noted difference between the two models' handling
of the imbalance (documented in the report rather than changed here).
"""

import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import GridSearchCV
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

# 1. Load the preprocessed UNSCALED dataset (tree models don't need scaling)
X_train = pd.read_csv(os.path.join(OUTPUTS_DIR, "clean_X_train.csv"))
X_test = pd.read_csv(os.path.join(OUTPUTS_DIR, "clean_X_test.csv"))
y_train = pd.read_csv(os.path.join(OUTPUTS_DIR, "clean_y_train.csv")).values.ravel()
y_test = pd.read_csv(os.path.join(OUTPUTS_DIR, "clean_y_test.csv")).values.ravel()

# 2. Hyperparameter tuning - max_depth, via GridSearchCV
param_grid = {"max_depth": [3, 4, 5, 6, 7, 8, 10, None]}
grid_search = GridSearchCV(
    DecisionTreeClassifier(class_weight="balanced", random_state=42),
    param_grid,
    cv=5,
    scoring="f1",
)
grid_search.fit(X_train, y_train)
model = grid_search.best_estimator_
print(f"Best max_depth found: {grid_search.best_params_['max_depth']}")

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
print("      DECISION TREE MODEL RESULTS      ")
print("=" * 40)
print(f"Accuracy  : {acc * 100:.2f}%")
print(f"Precision : {prec * 100:.2f}%")
print(f"Recall    : {rec * 100:.2f}%")
print(f"F1-Score  : {f1 * 100:.2f}%")
print(f"ROC-AUC   : {roc_auc * 100:.2f}%")
print("=" * 40)

os.makedirs(CHARTS_DIR, exist_ok=True)

# 5. Plot and save Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Churn", "Churn"])

fig, ax = plt.subplots(figsize=(6, 5))
disp.plot(cmap="Blues", ax=ax)
plt.title("Decision Tree Confusion Matrix")
plt.tight_layout()
plt.savefig(os.path.join(CHARTS_DIR, "decision_tree_cm.png"), dpi=300)
plt.close()

# 6. Plot and save the tree diagram (visual justification for model choice)
# Limited to depth=3 for readability even if the tuned tree is deeper
fig, ax = plt.subplots(figsize=(20, 10))
plot_tree(
    model,
    max_depth=3,
    feature_names=X_train.columns,
    class_names=["No Churn", "Churn"],
    filled=True,
    fontsize=8,
    ax=ax,
)
plt.title("Decision Tree (first 3 levels)")
plt.tight_layout()
plt.savefig(os.path.join(CHARTS_DIR, "decision_tree_plot.png"), dpi=300)
plt.close()

# 7. Save trained model file for Task 3 / Streamlit integration
model_path = os.path.join(OUTPUTS_DIR, "decision_tree_model.pkl")
joblib.dump(model, model_path)
print(f"Model saved to {model_path}")

# 8. Compare with Task 1's Logistic Regression
print()
print("=" * 55)
print("   COMPARISON: DECISION TREE vs LOGISTIC REGRESSION   ")
print("=" * 55)

lr_model_path = os.path.join(OUTPUTS_DIR, "logistic_regression_model.pkl")
if os.path.exists(lr_model_path):
    lr_model = joblib.load(lr_model_path)
    # Logistic Regression was trained on the SCALED features
    X_test_scaled = pd.read_csv(os.path.join(OUTPUTS_DIR, "clean_X_test_scaled.csv"))
    lr_pred = lr_model.predict(X_test_scaled)
    lr_pred_proba = lr_model.predict_proba(X_test_scaled)[:, 1]

    lr_acc = accuracy_score(y_test, lr_pred)
    lr_prec = precision_score(y_test, lr_pred)
    lr_rec = recall_score(y_test, lr_pred)
    lr_f1 = f1_score(y_test, lr_pred)
    lr_roc_auc = roc_auc_score(y_test, lr_pred_proba)

    comparison = pd.DataFrame({
        "Metric": ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"],
        "Logistic Regression": [lr_acc, lr_prec, lr_rec, lr_f1, lr_roc_auc],
        "Decision Tree": [acc, prec, rec, f1, roc_auc],
    })
    comparison["Logistic Regression"] = (comparison["Logistic Regression"] * 100).round(2)
    comparison["Decision Tree"] = (comparison["Decision Tree"] * 100).round(2)
    print(comparison.to_string(index=False))

    comparison.to_csv(os.path.join(OUTPUTS_DIR, "model_comparison.csv"), index=False)
    print(f"\nComparison table saved to {os.path.join(OUTPUTS_DIR, 'model_comparison.csv')}")
else:
    print(f"Logistic Regression model not found at {lr_model_path} - skipping comparison.")
