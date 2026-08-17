"""
task4_xgboost/xgboost_model.py - Task 4: XGBoost Model

Why XGBoost was chosen:
XGBoost was selected as the fourth model because gradient boosting builds
trees sequentially, with each new tree correcting the errors of the previous
ones, which typically gives stronger predictive performance than a single
Decision Tree while still handling the mixed binary/one-hot encoded features
from preprocessing.py natively (no scaling required). Built-in regularisation
and scale_pos_weight support also make it well suited to this dataset's
class imbalance (~73% No / ~27% Yes), which is explicitly handled below.

Reference:
Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System.
In Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge
Discovery and Data Mining (pp. 785-794). ACM.
"""

import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV
import joblib
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve
import os
import json


# Resolve project directories dynamically, so this works no matter where it's run from
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
CHARTS_DIR = os.path.join(OUTPUTS_DIR, "charts")
os.makedirs(CHARTS_DIR, exist_ok=True)

# load the data
X_train = pd.read_csv(os.path.join(OUTPUTS_DIR, "clean_X_train.csv"))
X_test = pd.read_csv(os.path.join(OUTPUTS_DIR, "clean_X_test.csv"))
y_train = pd.read_csv(os.path.join(OUTPUTS_DIR, "clean_y_train.csv")).values.ravel()
y_test = pd.read_csv(os.path.join(OUTPUTS_DIR, "clean_y_test.csv")).values.ravel()

neg = (y_train == 0).sum()
pos = (y_train == 1).sum()
scale_pos_weight = neg / pos
print("scale_pos_weight:", scale_pos_weight)

# Tune Hyperparameters
param_grid = {
    'learning_rate': [0.01, 0.1, 0.2],
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7]
}

grid_search = GridSearchCV(
    xgb.XGBClassifier(random_state=42, scale_pos_weight=scale_pos_weight),
    param_grid, cv=3, scoring='f1'
)
grid_search.fit(X_train, y_train)

print("Best parameters:", grid_search.best_params_)

with open(os.path.join(OUTPUTS_DIR, "xgboost_best_params.json"), "w") as f:
    json.dump(grid_search.best_params_, f, indent=2)

model = grid_search.best_estimator_

y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:,1]

print(y_pred[:10])
print(y_pred_proba[:10])

#  Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("ROC-AUC:", roc_auc)

print(classification_report(y_test, y_pred))

# Plot Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Churn', 'Churn'], yticklabels=['No Churn', 'Churn'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('XGBoost Confusion Matrix')
plt.savefig(os.path.join(CHARTS_DIR, "xgboost_confusion_matrix.png"))
plt.close()

fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"XGBoost (AUC = {roc_auc:.3f})")
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('XGBoost ROC Curve')
plt.legend()
plt.savefig(os.path.join(CHARTS_DIR, "xgboost_roc_curve.png"))
plt.close()

plt.figure(figsize=(8, 6))
xgb.plot_importance(model, max_num_features=10,importance_type='gain')
plt.title('XGBoost Feature Importance')
plt.tight_layout()
plt.savefig(os.path.join(CHARTS_DIR, "xgboost_feature_importance.png"))
plt.close()

xgb_model_path = os.path.join(OUTPUTS_DIR, "xgboost_model.pkl")
joblib.dump(model, xgb_model_path)
print(f"Model saved to {xgb_model_path}")

results_row = pd.DataFrame([{
    "model": "XGBoost",
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1": f1,
    "roc_auc": roc_auc
}])

results_path = os.path.join(OUTPUTS_DIR, "model_results.csv")
if os.path.exists(results_path):
    existing = pd.read_csv(results_path)
    existing = existing[existing["model"] != "XGBoost"]
    combined = pd.concat([existing, results_row], ignore_index=True)
else:
    combined = results_row
combined.to_csv(results_path, index=False)
print("Results saved to", results_path)