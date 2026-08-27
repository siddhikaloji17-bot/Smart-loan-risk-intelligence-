"""
SHAP explainability layer — turns the model into a "risk intelligence"
system by showing WHICH features drive each prediction.
"""

import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

MODEL_DIR = "/home/claude/loan_risk_project/models"
OUT = "/home/claude/loan_risk_project/outputs"

model = joblib.load(f"{MODEL_DIR}/best_model.pkl")
preprocessor = joblib.load(f"{MODEL_DIR}/preprocessor.pkl")
meta = joblib.load(f"{MODEL_DIR}/metadata.pkl")
X_test_raw = pd.read_csv(f"{MODEL_DIR}/X_test_raw.csv")

feature_cols = meta["numeric_features"] + meta["categorical_features"]
X_test = X_test_raw[feature_cols]
X_test_proc = preprocessor.transform(X_test)

# Get feature names after one-hot encoding
cat_encoder = preprocessor.named_transformers_["cat"].named_steps["onehot"]
cat_feature_names = cat_encoder.get_feature_names_out(meta["categorical_features"])
all_feature_names = meta["numeric_features"] + list(cat_feature_names)

X_test_df = pd.DataFrame(
    X_test_proc.toarray() if hasattr(X_test_proc, "toarray") else X_test_proc,
    columns=all_feature_names
)

# Use a sample for speed
sample = X_test_df.sample(n=min(500, len(X_test_df)), random_state=42)

model_name = meta["best_model_name"]
if model_name == "Logistic Regression":
    explainer = shap.LinearExplainer(model, sample)
elif model_name == "Random Forest":
    explainer = shap.TreeExplainer(model)
else:  # XGBoost
    explainer = shap.TreeExplainer(model)

shap_values = explainer.shap_values(sample)
if isinstance(shap_values, list):  # some explainers return a list per class
    shap_values = shap_values[1]

# 1. Global feature importance (summary plot)
plt.figure()
shap.summary_plot(shap_values, sample, show=False, plot_size=(9, 6))
plt.tight_layout()
plt.savefig(f"{OUT}/09_shap_summary.png", dpi=120, bbox_inches="tight")
plt.close()

# 2. Bar plot of mean |SHAP value| per feature
plt.figure()
shap.summary_plot(shap_values, sample, plot_type="bar", show=False, plot_size=(9, 6))
plt.tight_layout()
plt.savefig(f"{OUT}/10_shap_feature_importance_bar.png", dpi=120, bbox_inches="tight")
plt.close()

# 3. Individual applicant explanation (waterfall for one high-risk case)
proba_all = model.predict_proba(X_test_df)[:, 1]
sample_idx = sample.index
proba_sample = proba_all[X_test_df.index.get_indexer(sample_idx)]
highest_risk_pos = np.argmax(proba_sample)

exp = shap.Explanation(
    values=shap_values[highest_risk_pos],
    base_values=explainer.expected_value if not isinstance(explainer.expected_value, np.ndarray)
                else explainer.expected_value[1],
    data=sample.iloc[highest_risk_pos].values,
    feature_names=all_feature_names
)
plt.figure()
shap.plots.waterfall(exp, show=False, max_display=12)
plt.tight_layout()
plt.savefig(f"{OUT}/11_shap_individual_explanation.png", dpi=120, bbox_inches="tight")
plt.close()

print(f"Saved 3 SHAP plots to {OUT}/")
print(f"Model explained: {model_name}")
print(f"Example applicant risk probability shown in waterfall: {proba_sample[highest_risk_pos]:.2%}")
