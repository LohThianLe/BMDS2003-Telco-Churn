"""
eda_visualization/eda.py - Task 1: Exploratory Data Analysis & Visualisation
Generates 6 saved chart visualisations into outputs/charts/
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Dynamic path resolution to project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "Telco_Cusomer_Churn.csv")
CHARTS_DIR = os.path.join(BASE_DIR, "outputs", "charts")
os.makedirs(CHARTS_DIR, exist_ok=True)

# Set global seaborn style
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({"font.size": 11, "figure.titlesize": 14})

# 1. Load Data
df = pd.read_csv(DATA_PATH)

print("--- DATASET INFO ---")
df.info()

# Convert TotalCharges to numeric BEFORE the missing-value check, so blank-string
# entries (11 customers with tenure=0, not yet billed) are counted as NaN and show
# up correctly below instead of being silently missed.
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

print("\n--- MISSING VALUES ---")
print(df.isnull().sum())

print("\n--- DUPLICATE ROWS ---")
print(f"Total duplicate rows: {df.duplicated().sum()}")

print("\n--- DESCRIPTIVE STATISTICS ---")
print(df[["tenure", "MonthlyCharges", "TotalCharges"]].describe())

# Now fill the (documented) TotalCharges blanks with 0 for charting purposes —
# these are tenure=0 customers who haven't been billed yet, not "unknown" values.
df["TotalCharges"] = df["TotalCharges"].fillna(0)

# Translate target variable for clear labels
df["Churn_Label"] = df["Churn"].map({"Yes": "Churned", "No": "Retained"})



# CHART 1: Churn vs Non-Churn Count & Percentage

fig, ax = plt.subplots(figsize=(7, 5))
churn_counts = df["Churn_Label"].value_counts()
colors = ["#2b5c8f", "#d95f02"]

bars = ax.bar(churn_counts.index, churn_counts.values, color=colors, width=0.5)
ax.set_title("Customer Churn Distribution (Class Imbalance)", fontweight="bold")
ax.set_ylabel("Number of Customers")
ax.set_ylim(0, 6000)

total = len(df)
for bar in bars:
    height = bar.get_height()
    pct = (height / total) * 100
    ax.annotate(
        f"{height:,}\n({pct:.1f}%)",
        xy=(bar.get_x() + bar.get_width() / 2, height),
        xytext=(0, 5),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontweight="bold",
    )

plt.tight_layout()
plt.savefig(
    os.path.join(CHARTS_DIR, "01_churn_count.png"), dpi=300, bbox_inches="tight"
)
plt.close()

print("\n[Chart 1 Interpretation] Churned customers make up a minority of the "
      f"base ({churn_counts.get('Churned', 0) / total * 100:.1f}%), confirming a "
      "class imbalance that Task 2's preprocessing needs to account for (e.g. "
      "class weighting) so models don't just learn to predict 'no churn' every time.")



# CHART 2: Distribution of Tenure and Monthly Charges

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

sns.histplot(
    df["tenure"],
    kde=True,
    color="#2b5c8f",
    ax=axes[0],
    bins=30,
)
axes[0].set_title("Distribution of Customer Tenure (Months)", fontweight="bold")
axes[0].set_xlabel("Tenure (Months)")
axes[0].set_ylabel("Customer Count")

sns.histplot(
    df["MonthlyCharges"],
    kde=True,
    color="#d95f02",
    ax=axes[1],
    bins=30,
)
axes[1].set_title("Distribution of Monthly Charges ($)", fontweight="bold")
axes[1].set_xlabel("Monthly Charges ($)")
axes[1].set_ylabel("Customer Count")

plt.tight_layout()
plt.savefig(
    os.path.join(CHARTS_DIR, "02_distributions.png"),
    dpi=300,
    bbox_inches="tight",
)
plt.close()

print("\n[Chart 2 Interpretation] Tenure is not normally distributed — there's a "
      "large spike of very new customers (0-5 months) alongside a second cluster "
      "near 70+ months, suggesting two distinct customer types: recent sign-ups "
      "and long-term loyal customers. Monthly charges are spread fairly evenly "
      "across price tiers, with a visible cluster around $20 (likely customers "
      "with no add-on internet/streaming services).")



# CHART 3: Monthly Charges Boxplot (Churn vs Retained)

# Explicit category order, used for BOTH the boxplot and the median labels below,
# so the two can never drift out of sync with each other (this is what caused the
# bug: sns.boxplot() previously defaulted to first-appearance order, while
# groupby() defaults to alphabetical order -- two different orderings that
# happened to look plausible but put each label on the wrong box).
order = ["Retained", "Churned"]

fig, ax = plt.subplots(figsize=(8, 5))
sns.boxplot(
    data=df,
    x="Churn_Label",
    y="MonthlyCharges",
    hue="Churn_Label",
    palette=["#2b5c8f", "#d95f02"],
    legend=False,
    ax=ax,
    width=0.4,
    order=order,
)
ax.set_title(
    "Monthly Charges Comparison: Retained vs Churned Customers",
    fontweight="bold",
)
ax.set_xlabel("Customer Status")
ax.set_ylabel("Monthly Charges ($)")

# Overlay median annotations -- looked up by label (.loc), not by position,
# so each annotation is guaranteed to land on the box it actually describes.
medians = df.groupby("Churn_Label")["MonthlyCharges"].median()
for i, label in enumerate(order):
    median = medians.loc[label]
    ax.text(
        i,
        median + 1.5,
        f"Median: ${median:.2f}",
        ha="center",
        va="bottom",
        fontweight="bold",
        color="black",
    )

plt.tight_layout()
plt.savefig(
    os.path.join(CHARTS_DIR, "03_monthly_charges_boxplot.png"),
    dpi=300,
    bbox_inches="tight",
)
plt.close()

print(f"\n[Chart 3 Interpretation] Churned customers have a higher median monthly "
      f"charge (${medians.get('Churned', 0):.2f}) than retained customers "
      f"(${medians.get('Retained', 0):.2f}), suggesting price sensitivity — "
      "customers paying more per month may feel they're not getting enough value "
      "and are more likely to leave.")



# CHART 4: Categorical Churn Rates (Contract, Payment, Internet)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

cat_cols = ["Contract", "InternetService", "PaymentMethod"]
titles = ["Contract Type", "Internet Service", "Payment Method"]

for idx, col in enumerate(cat_cols):
    churn_pct = (
        df.groupby(col)["Churn"]
        .apply(lambda x: (x == "Yes").mean() * 100)
        .reset_index()
    )
    sns.barplot(
        data=churn_pct,
        x=col,
        y="Churn",
        hue=col,
        palette="Oranges_r",
        legend=False,
        ax=axes[idx],
    )
    axes[idx].set_title(f"Churn Rate by {titles[idx]}", fontweight="bold")
    axes[idx].set_ylabel("Churn Rate (%)")
    axes[idx].set_xlabel("")
    axes[idx].set_ylim(0, 60)
    axes[idx].tick_params(axis="x", rotation=15)

    for p in axes[idx].patches:
        axes[idx].annotate(
            f"{p.get_height():.1f}%",
            (p.get_x() + p.get_width() / 2.0, p.get_height()),
            ha="center",
            va="center",
            xytext=(0, 5),
            textcoords="offset points",
            fontweight="bold",
        )

plt.tight_layout()
plt.savefig(
    os.path.join(CHARTS_DIR, "04_categorical_churn.png"),
    dpi=300,
    bbox_inches="tight",
)
plt.close()

print("\n[Chart 4 Interpretation] Month-to-month contracts, electronic check "
      "payments, and fiber optic internet all show noticeably higher churn rates "
      "than their alternatives (one/two-year contracts, automatic payment methods, "
      "DSL/no internet). This points to a lack of long-term commitment and possibly "
      "service/reliability dissatisfaction with fiber optic as key churn drivers — "
      "worth targeting with retention offers (e.g. contract upgrade incentives).")



# CHART 5: Numerical Correlation Heatmap (Inc. SeniorCitizen)

fig, ax = plt.subplots(figsize=(7, 6))
df_num = df[["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]].copy()
df_num["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

corr = df_num.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))

sns.heatmap(
    corr,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    vmin=-1,
    vmax=1,
    ax=ax,
    mask=mask,
    cbar_kws={"shrink": 0.8},
)
ax.set_title(
    "Correlation Matrix of Numerical Features & Churn", fontweight="bold"
)

plt.tight_layout()
plt.savefig(
    os.path.join(CHARTS_DIR, "05_correlation_heatmap.png"),
    dpi=300,
    bbox_inches="tight",
)
plt.close()

print(f"\n[Chart 5 Interpretation] Tenure and TotalCharges are strongly positively "
      f"correlated (r={corr.loc['TotalCharges', 'tenure']:.2f}), which makes sense "
      "since longer-tenured customers accumulate more total billing — this is a "
      "multicollinearity risk worth flagging for Task 2/3's modelling. Churn shows "
      f"a moderate negative correlation with tenure (r={corr.loc['Churn', 'tenure']:.2f}), "
      "reinforcing that newer customers are more likely to leave.")



# CHART 6: ADDITIONAL PATTERN — Churn Rate by Tenure Group

bins = [0, 12, 24, 48, 72]
labels = ["0-1 Year", "1-2 Years", "2-4 Years", "4-6 Years"]
df["TenureGroup"] = pd.cut(df["tenure"], bins=bins, labels=labels, include_lowest=True)

fig, ax = plt.subplots(figsize=(8, 5))
tenure_churn = (
    df.groupby("TenureGroup", observed=False)["Churn"]
    .apply(lambda x: (x == "Yes").mean() * 100)
    .reset_index()
)

sns.barplot(
    data=tenure_churn,
    x="TenureGroup",
    y="Churn",
    hue="TenureGroup",
    palette="Reds_r",
    legend=False,
    ax=ax,
    width=0.5,
)
ax.set_title(
    "Additional Insight: Churn Rate by Customer Tenure Cohort",
    fontweight="bold",
)
ax.set_xlabel("Customer Tenure Group")
ax.set_ylabel("Churn Rate (%)")
ax.set_ylim(0, 60)

for p in ax.patches:
    ax.annotate(
        f"{p.get_height():.1f}%",
        (p.get_x() + p.get_width() / 2.0, p.get_height()),
        ha="center",
        va="center",
        xytext=(0, 5),
        textcoords="offset points",
        fontweight="bold",
    )

plt.tight_layout()
plt.savefig(
    os.path.join(CHARTS_DIR, "06_additional_pattern_tenure_groups.png"),
    dpi=300,
    bbox_inches="tight",
)
plt.close()

print("\n[Chart 6 Interpretation] Churn rate is highest among customers in their "
      "first year (0-1 Year group) and drops sharply as tenure increases, "
      "flattening out for customers who have stayed 4+ years. This suggests "
      "retention efforts should focus on the first 12 months, since customers "
      "who make it past that window are far less likely to leave.")

print("\nEDA visualisations successfully generated and saved to outputs/charts/")