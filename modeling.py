import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

# 1. Load the clean preprocessed data / 加载清洗后的预处理数据
X_train = pd.read_csv('outputs/clean_X_train.csv')
X_test = pd.read_csv('outputs/clean_X_test.csv')
y_train = pd.read_csv('outputs/clean_y_train.csv').values.ravel()
y_test = pd.read_csv('outputs/clean_y_test.csv').values.ravel()

# 2. Train the Logistic Regression Model / 训练逻辑回归模型
print("Training Logistic Regression Baseline Model...")
# class_weight='balanced' helps the model deal with our imbalanced churn data
model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
model.fit(X_train, y_train)

# 3. Make predictions on the test set / 在测试集上进行预测
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1] # Probability scores for ROC-AUC

# 4. Evaluate the model / 计算评估指标
print("\n=== MODEL PERFORMANCE METRICS  ===")
print(f"Accuracy   : {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision  : {precision_score(y_test, y_pred):.4f}")
print(f"Recall     : {recall_score(y_test, y_pred):.4f}")
print(f"F1-Score   : {f1_score(y_test, y_pred):.4f}")
print(f"ROC-AUC    : {roc_auc_score(y_test, y_prob):.4f}")

# 5. Plot and save Confusion Matrix / 绘制并保存混淆矩阵
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Logistic Regression Confusion Matrix')
plt.ylabel('Actual Label ')
plt.xlabel('Predicted Label ')
plt.tight_layout()

# Save the matrix plot / 保存矩阵图
os.makedirs('outputs/charts', exist_ok=True)
plt.savefig('outputs/charts/logistic_regression_cm.png')
plt.close()
print("\nConfusion matrix chart saved successfully in 'outputs/charts/'!")