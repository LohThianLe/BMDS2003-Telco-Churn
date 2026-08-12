import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

rf_model = joblib.load('outputs/random_forest_model.pkl')
lr_model = joblib.load('outputs/logistic_regression_model.pkl')
dt_model = joblib.load('outputs/decision_tree_model.pkl')
xgb_model = joblib.load('outputs/xgboost_model.pkl')
print("All models loaded!")

X_test = pd.read_csv('outputs/clean_X_test.csv')
X_test_scaled = pd.read_csv('outputs/clean_X_test_scaled.csv')
y_test = pd.read_csv('outputs/clean_y_test.csv').values.ravel()

lr_pred = lr_model.predict(X_test_scaled)
lr_prob = lr_model.predict_proba(X_test_scaled)[:, 1]

dt_pred = dt_model.predict(X_test)
dt_prob = dt_model.predict_proba(X_test)[:, 1]

rf_pred = rf_model.predict(X_test)
rf_prob = rf_model.predict_proba(X_test)[:, 1]

xgb_pred = xgb_model.predict(X_test)
xgb_prob = xgb_model.predict_proba(X_test)[:, 1]

def get_metrics(y_true, y_pred, y_prob):
    return [
        accuracy_score(y_true, y_pred) * 100,
        precision_score(y_true, y_pred) * 100,
        recall_score(y_true, y_pred) * 100,
        f1_score(y_true, y_pred) * 100,
        roc_auc_score(y_true, y_prob) * 100
    ]

lr_metrics = get_metrics(y_test, lr_pred, lr_prob)
dt_metrics = get_metrics(y_test, dt_pred, dt_prob)
rf_metrics = get_metrics(y_test, rf_pred, rf_prob)
xgb_metrics = get_metrics(y_test, xgb_pred, xgb_prob)

comparison = pd.DataFrame({
    "Metric": ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"],
    "Logistic Regression": lr_metrics,
    "Decision Tree": dt_metrics,
    "Random Forest": rf_metrics,
    "XGBoost": xgb_metrics
})
comparison = comparison.round(2)

print(comparison.to_string(index=False))
comparison.to_csv('outputs/model_comparison.csv', index=False)
print("\nComparison table saved to outputs/model_comparison.csv")