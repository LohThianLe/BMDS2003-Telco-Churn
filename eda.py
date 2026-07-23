import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

print("Loading data...")
#use the pandas function to open and read the csv save it to variable data frame call df(its a virual excel table)
df = pd.read_csv('data/Telco_Cusomer_Churn.csv')


print("\n--- DATASET INFO ---")
#show the summary of data such as names of columns ,type of data is inside(text , integer ,decimals..)
print(df.info())


print("\n--- MISSING VALUES ---")
#isnull().sum() will scan every column and count empty cells.
print(df.isnull().sum())

print("\n--- DUPLICATES  ---")
#then duplicated.sum() use for check any exact duplicate rows
print(f"Total duplicate rows: {df.duplicated().sum()}")

#use to make some space in totalCharges change to NaN ,avoid the machine mistake
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'].replace(' ', np.nan), errors='coerce')#force the whole to become to decimal,for calculate

print("\n--- DESCRIPTIVE STATISTICS  ---")
print(df[['tenure', 'MonthlyCharges', 'TotalCharges']].describe())

os.makedirs('outputs/charts', exist_ok=True)
print("\nGenerating charts... ")

# Set visual style
sns.set_theme(style="whitegrid")

# Chart 1: Churn vs Non-churn count 
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='Churn', palette='Set2')
plt.title('Churn vs Non-Churn Count')
plt.savefig('outputs/charts/01_churn_count.png')
plt.close()

# Chart 2: Distribution of tenure and monthly charges 
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.histplot(df['tenure'], kde=True, ax=axes[0], color='skyblue')
axes[0].set_title('Distribution of Tenure (Months)')
sns.histplot(df['MonthlyCharges'], kde=True, ax=axes[1], color='salmon')
axes[1].set_title('Distribution of Monthly Charges')
plt.tight_layout()
plt.savefig('outputs/charts/02_distributions.png')
plt.close()

# Chart 3: Monthly charges by Churn (Boxplot) 
plt.figure(figsize=(6, 5))
sns.boxplot(data=df, x='Churn', y='MonthlyCharges', palette='Set2')
plt.title('Monthly Charges by Churn')
plt.savefig('outputs/charts/03_monthly_charges_boxplot.png')
plt.close()

# Chart 4: Churn rate by contract, payment, internet 
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
sns.countplot(data=df, x='Contract', hue='Churn', ax=axes[0], palette='viridis')
axes[0].set_title('Churn by Contract Type')
sns.countplot(data=df, x='PaymentMethod', hue='Churn', ax=axes[1], palette='viridis')
axes[1].tick_params(axis='x', rotation=45) # Rotate labels so they don't overlap
axes[1].set_title('Churn by Payment Method')
sns.countplot(data=df, x='InternetService', hue='Churn', ax=axes[2], palette='viridis')
axes[2].set_title('Churn by Internet Service')
plt.tight_layout()
plt.savefig('outputs/charts/04_categorical_churn.png')
plt.close()

# Chart 5: Correlation heatmap 
plt.figure(figsize=(6, 5))
# Only correlate numeric columns 
numeric_df = df[['tenure', 'MonthlyCharges', 'TotalCharges']].dropna()
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.savefig('outputs/charts/05_correlation_heatmap.png')
plt.close()

print("All charts saved successfully in 'outputs/charts/' folder! ")