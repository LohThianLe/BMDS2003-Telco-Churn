"""
preprocessing.py
Data cleaning and preparation for the Telco Customer Churn dataset.

Steps:
1. Drop customerID (identifier, no predictive value)
2. Fix TotalCharges (blank values -> 0, for tenure=0 customers)
3. Collapse "No internet/phone service" into "No" (redundant categories)
4. Encode binary Yes/No columns as 0/1
5. One-hot encode multi-category columns (drop_first=True)
6. Train/test split (70/30, random_state=42)
7. Produce a scaled version of numeric columns alongside the unscaled version
8. Save all outputs to outputs/
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

RAW_PATH = "data/Telco_Cusomer_Churn.csv"
OUTPUT_DIR = "outputs"

# ---------- 1. Load data ----------
df = pd.read_csv(RAW_PATH)

# ---------- 2. Drop identifier column ----------
df = df.drop(columns=["customerID"])

# ---------- 3. Fix TotalCharges ----------
# Blank strings -> NaN -> fill with 0 (tenure=0 customers, no charges yet)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"] = df["TotalCharges"].fillna(0)

# ---------- 4. Collapse redundant categories ----------
service_cols = [
    "MultipleLines", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
]
for col in service_cols:
    df[col] = df[col].replace({"No internet service": "No", "No phone service": "No"})

# ---------- 5. Encode binary columns as 0/1 ----------
binary_yes_no_cols = ["Partner", "Dependents", "PhoneService", "PaperlessBilling", "Churn"] + service_cols
for col in binary_yes_no_cols:
    df[col] = df[col].map({"Yes": 1, "No": 0})

df["gender"] = df["gender"].map({"Male": 1, "Female": 0})
# SeniorCitizen is already 0/1, no change needed

# ---------- 6. One-hot encode multi-category columns ----------
df = pd.get_dummies(
    df,
    columns=["Contract", "InternetService", "PaymentMethod"],
    drop_first=True,
    dtype=int,
)

# ---------- 7. Split features and target ----------
X = df.drop(columns=["Churn"])
y = df["Churn"]

# ---------- 8. Train/test split ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42
)

# ---------- 9. Scaled version (fit on train only, applied to both) ----------
numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges"]

X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()

scaler = StandardScaler()
X_train_scaled[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
X_test_scaled[numeric_cols] = scaler.transform(X_test[numeric_cols])

# ---------- 10. Save outputs ----------
X_train.to_csv(f"{OUTPUT_DIR}/clean_X_train.csv", index=False)
X_test.to_csv(f"{OUTPUT_DIR}/clean_X_test.csv", index=False)
y_train.to_csv(f"{OUTPUT_DIR}/clean_y_train.csv", index=False)
y_test.to_csv(f"{OUTPUT_DIR}/clean_y_test.csv", index=False)

X_train_scaled.to_csv(f"{OUTPUT_DIR}/clean_X_train_scaled.csv", index=False)
X_test_scaled.to_csv(f"{OUTPUT_DIR}/clean_X_test_scaled.csv", index=False)

print("Preprocessing complete.")
print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")
print(f"y_train: {y_train.shape}, y_test: {y_test.shape}")
