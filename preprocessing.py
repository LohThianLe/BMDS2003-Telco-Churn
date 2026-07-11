import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load data 
df = pd.read_csv('data/Telco_Cusomer_Churn.csv')
df = df.drop(columns=['customerID']) # Drop useless ID 

# Fix missing total charges 
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'].replace(' ', np.nan), errors='coerce')
df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())

# Convert Yes/No to 1/0 
binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn', 'gender']
le = LabelEncoder()
for col in binary_cols:
    df[col] = le.fit_transform(df[col])

# One-hot encode multi-category columns 
multi_cols = ['MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 
              'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaymentMethod']
df = pd.get_dummies(df, columns=multi_cols, drop_first=True).astype(int)

# Split data into Features (X) and Target (y)
X = df.drop(columns=['Churn'])
y = df['Churn']

# 80% Train, 20% Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale numerical columns 
num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
scaler = StandardScaler()
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])

# Save files 
os.makedirs('outputs', exist_ok=True)
X_train.to_csv('outputs/clean_X_train.csv', index=False)
X_test.to_csv('outputs/clean_X_test.csv', index=False)
y_train.to_csv('outputs/clean_y_train.csv', index=False)
y_test.to_csv('outputs/clean_y_test.csv', index=False)
print("Clean data generated successfully in 'outputs/' folder!")