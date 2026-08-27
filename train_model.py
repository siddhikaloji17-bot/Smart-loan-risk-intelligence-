"""
Feature engineering + imbalance handling + model training/comparison
for the Smart Loan Default & Financial Risk Intelligence System.

Trains Logistic Regression, Random Forest, and XGBoost; compares them
with ROC-AUC, Precision-Recall, F1, and confusion matrix; saves the
best model + preprocessing pipeline for later use (SHAP + Streamlit app).
"""

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve,
    f1_score, classification_report, confusion_matrix
)
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR / "data" / "loan_data.csv"
OUT = PROJECT_DIR / "outputs"
MODEL_DIR = PROJECT_DIR / "models"

df = pd.read_csv(DATA_PATH)

# ---------------------------------------------------------------
# 1. FEATURE ENGINEERING
# ---------------------------------------------------------------
df["loan_to_income"] = df["loan_amount"] / df["annual_income"].replace(0, np.nan)
df["income_band"] = pd.cut(
    df["annual_income"],
    bins=[0, 30000, 60000, 100000, 200000, np.inf],
    labels=["very_low", "low", "medium", "high", "very_high"]
)
df["age_group"] = pd.cut(
    df["age"], bins=[0, 25, 35, 45, 55, 100],
    labels=["18-25", "26-35", "36-45", "46-55", "56+"]
)

target = "default"
numeric_features = [
    "age", "annual_income", "employment_length_yrs", "credit_history_length_yrs",
    "loan_amount", "interest_rate", "existing_loans", "credit_utilization",
    "num_late_payments_2yr", "has_bankruptcy", "debt_to_income", "loan_to_income"
]
categorical_features = [
    "employment_type", "loan_purpose", "income_band", "age_group"
]

X = df[numeric_features + categorical_features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ---------------------------------------------------------------
# 2. PREPROCESSING PIPELINE
# ---------------------------------------------------------------
numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])
categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])
preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

X_train_proc = preprocessor.fit_transform(X_train)
X_test_proc = preprocessor.transform(X_test)

# ---------------------------------------------------------------
# 3. HANDLE CLASS IMBALANCE (SMOTE on training data only)
# ---------------------------------------------------------------
print(f"Before SMOTE: {np.bincount(y_train)}")
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train_proc, y_train)
print(f"After SMOTE:  {np.bincount(y_train_bal)}")

# ---------------------------------------------------------------
# 4. TRAIN + COMPARE MODELS
# ---------------------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=10, random_state=42, n_jobs=-1),
    "XGBoost": XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        eval_metric="logloss", random_state=42, n_jobs=-1
    ),
}

results = {}
plt.figure(figsize=(7, 6))

for name, model in models.items():
    model.fit(X_train_bal, y_train_bal)
    proba = model.predict_proba(X_test_proc)[:, 1]
    preds = model.predict(X_test_proc)

    auc = roc_auc_score(y_test, proba)
    f1 = f1_score(y_test, preds)
    results[name] = {"model": model, "auc": auc, "f1": f1, "proba": proba, "preds": preds}

    fpr, tpr, _ = roc_curve(y_test, proba)
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")

    print(f"\n===== {name} =====")
    print(f"ROC-AUC: {auc:.4f} | F1: {f1:.4f}")
    print(classification_report(y_test, preds, digits=3))

plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUT}/06_roc_comparison.png", dpi=120)
plt.close()

# ---------------------------------------------------------------
# 5. PICK BEST MODEL (by ROC-AUC) AND SAVE ARTIFACTS
# ---------------------------------------------------------------
best_name = max(results, key=lambda k: results[k]["auc"])
best_model = results[best_name]["model"]
print(f"\nBest model: {best_name} (AUC={results[best_name]['auc']:.4f})")

# Confusion matrix for best model
cm = confusion_matrix(y_test, results[best_name]["preds"])
plt.figure(figsize=(5, 4))
plt.imshow(cm, cmap="Blues")
plt.title(f"Confusion Matrix - {best_name}")
plt.colorbar()
plt.xticks([0, 1], ["No Default", "Default"])
plt.yticks([0, 1], ["No Default", "Default"])
for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha="center", va="center",
                  color="white" if cm[i, j] > cm.max() / 2 else "black")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig(f"{OUT}/07_confusion_matrix_best_model.png", dpi=120)
plt.close()

# Precision-Recall curve for best model
prec, rec, _ = precision_recall_curve(y_test, results[best_name]["proba"])
plt.figure(figsize=(6, 5))
plt.plot(rec, prec)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title(f"Precision-Recall Curve - {best_name}")
plt.tight_layout()
plt.savefig(f"{OUT}/08_precision_recall_curve.png", dpi=120)
plt.close()

joblib.dump(best_model, f"{MODEL_DIR}/best_model.pkl")
joblib.dump(preprocessor, f"{MODEL_DIR}/preprocessor.pkl")
joblib.dump({name: item["model"] for name, item in results.items()}, MODEL_DIR / "comparison_models.pkl")
joblib.dump({
    name: {"auc": item["auc"], "f1": item["f1"]}
    for name, item in results.items()
}, MODEL_DIR / "comparison_metrics.pkl")
joblib.dump({
    "numeric_features": numeric_features,
    "categorical_features": categorical_features,
    "best_model_name": best_name,
}, f"{MODEL_DIR}/metadata.pkl")

X_test.assign(default=y_test).to_csv(f"{MODEL_DIR}/X_test_raw.csv", index=False)

print(f"\nSaved best model ({best_name}), preprocessor, and metadata to {MODEL_DIR}/")
