import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import joblib

X_train = pd.read_csv('outputs/clean_X_train.csv')
X_test = pd.read_csv('outputs/clean_X_test.csv')
y_train = pd.read_csv('outputs/clean_y_train.csv').values.ravel()
y_test = pd.read_csv('outputs/clean_y_test.csv').values.ravel()

print("Training Random Forest Model...")
model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42) #100 trees
model.fit(X_train, y_train)
joblib.dump(model, 'outputs/random_forest_model.pkl')
print("\nModel saved to outputs/random_forest_model.pkl")

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1] #this gives 2 columns = column 0 (no churn) and column 1 (churn) so so we take column 1 

#True Positive (TP) — actually churned, model correctly said "churn"
#True Negative (TN) — actually stayed, model correctly said "no churn"
#False Positive (FP) — actually stayed, model wrongly said "churn" (false alarm)
#False Negative (FN) — actually churned, model wrongly said "no churn" (missed them)

print("\n=== RANDOM FOREST PERFORMANCE METRICS ===")
print(f"Accuracy   : {accuracy_score(y_test, y_pred):.4f}") #compare test result vs answer
print(f"Precision  : {precision_score(y_test, y_pred):.4f}") #precision = TP / (TP + FP)
print(f"Recall     : {recall_score(y_test, y_pred):.4f}") #Recall = TP / (TP + FN)
print(f"F1-Score   : {f1_score(y_test, y_pred):.4f}") #F1-Score = a blended average of Precision and Recall
print(f"ROC-AUC    : {roc_auc_score(y_test, y_prob):.4f}") #how well the model separates churners from non-churners overall

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Random Forest Confusion Matrix')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.tight_layout()

os.makedirs('outputs/charts', exist_ok=True) #if edy there then dont complain
plt.savefig('outputs/charts/random_forest_cm.png')
plt.close() #closing the chart itself out of the memory so it doesn't consume resources(??)
print("\nConfusion matrix chart saved successfully in 'outputs/charts/'!")