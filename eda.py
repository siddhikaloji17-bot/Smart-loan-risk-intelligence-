"""
Exploratory Data Analysis for the loan default dataset.
Saves plots to outputs/ so they can be reused in your interview slides/report.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
OUT = "/home/claude/loan_risk_project/outputs"

df = pd.read_csv("/home/claude/loan_risk_project/data/loan_data.csv")

print("Shape:", df.shape)
print("\nMissing values:\n", df.isnull().sum()[df.isnull().sum() > 0])
print("\nDefault rate:", df["default"].mean().round(4))

# 1. Target distribution
plt.figure(figsize=(5, 4))
sns.countplot(x="default", data=df, palette=["#4C72B0", "#C44E52"])
plt.title("Default vs Non-Default Distribution")
plt.xlabel("Default (1 = Yes)")
plt.tight_layout()
plt.savefig(f"{OUT}/01_target_distribution.png", dpi=120)
plt.close()

# 2. Default rate by loan purpose
plt.figure(figsize=(8, 4))
rate_by_purpose = df.groupby("loan_purpose")["default"].mean().sort_values(ascending=False)
sns.barplot(x=rate_by_purpose.values, y=rate_by_purpose.index, palette="Reds_r")
plt.title("Default Rate by Loan Purpose")
plt.xlabel("Default Rate")
plt.tight_layout()
plt.savefig(f"{OUT}/02_default_by_purpose.png", dpi=120)
plt.close()

# 3. Default rate by employment type
plt.figure(figsize=(6, 4))
rate_by_emp = df.groupby("employment_type")["default"].mean().sort_values(ascending=False)
sns.barplot(x=rate_by_emp.index, y=rate_by_emp.values, palette="Blues_r")
plt.title("Default Rate by Employment Type")
plt.ylabel("Default Rate")
plt.tight_layout()
plt.savefig(f"{OUT}/03_default_by_employment.png", dpi=120)
plt.close()

# 4. DTI and credit utilization vs default (boxplots)
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
sns.boxplot(x="default", y="debt_to_income", data=df, ax=axes[0], palette=["#4C72B0", "#C44E52"])
axes[0].set_title("Debt-to-Income by Default")
sns.boxplot(x="default", y="credit_utilization", data=df, ax=axes[1], palette=["#4C72B0", "#C44E52"])
axes[1].set_title("Credit Utilization by Default")
plt.tight_layout()
plt.savefig(f"{OUT}/04_dti_utilization_boxplots.png", dpi=120)
plt.close()

# 5. Correlation heatmap (numeric features only)
plt.figure(figsize=(9, 7))
numeric_df = df.select_dtypes(include="number")
corr = numeric_df.corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig(f"{OUT}/05_correlation_heatmap.png", dpi=120)
plt.close()

print(f"\nSaved 5 EDA plots to {OUT}/")
