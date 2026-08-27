# Smart Loan Default & Financial Risk Intelligence System

A data science project that predicts loan default risk, converts predictions
into an interpretable **0–100 risk score**, explains each score with **SHAP**,
and demos live through a **Streamlit dashboard**.

## Project Structure

```
loan_risk_project/
├── data/
│   ├── generate_data.py      # Creates a realistic synthetic loan dataset
│   └── loan_data.csv         # Generated dataset (15,000 applicants)
├── eda.py                    # Exploratory data analysis → outputs/01-05
├── train_model.py            # Feature engineering, SMOTE, model training/comparison
├── shap_explain.py           # Global + individual SHAP explainability → outputs/09-11
├── app/
├── app.py                    # Streamlit risk-scoring dashboard
├── models/                   # Saved best model + preprocessing pipeline
├── outputs/                  # All generated plots (EDA, ROC, confusion matrix, SHAP)
└── README.md
```

## How to Run (in order)

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# 1. Generate the dataset (or replace with a real Kaggle CSV — see note below)
python data/generate_data.py

# 2. Run EDA — saves plots to outputs/
python eda.py

# 3. Feature engineering + imbalance handling + train & compare models
python train_model.py

# 4. SHAP explainability
python shap_explain.py

# 5. Launch the interactive dashboard
streamlit run app.py
```

The dashboard is designed as an interview demo: start with the single-applicant
score, open SHAP to explain the decision, use What-if analysis to test a change,
then demonstrate batch scoring, PDF reporting, and model comparison.

## Using a Real Dataset Instead of Synthetic Data

Replace `data/loan_data.csv` with a real dataset such as Kaggle's
**"Lending Club Loan Data"** or **"Home Credit Default Risk"**, making sure
your CSV has (or is renamed to have) these columns:

`age, annual_income, employment_type, employment_length_yrs,
credit_history_length_yrs, loan_amount, loan_purpose, interest_rate,
existing_loans, credit_utilization, num_late_payments_2yr,
has_bankruptcy, default`

Then just skip step 1 and run `eda.py` onward.

## What Each Stage Does

| Stage                  | What happens                                                   | Why it matters for the interview                                       |
| ---------------------- | -------------------------------------------------------------- | ---------------------------------------------------------------------- |
| EDA                    | Default rate by purpose/employment, correlation heatmap        | Shows you understand the data before modeling                          |
| Feature Engineering    | DTI, loan-to-income, income/age binning                        | Demonstrates domain-aware feature creation                             |
| Imbalance Handling     | SMOTE applied only to training data                            | Prevents data leakage; classic interview question                      |
| Modeling               | Logistic Regression, Random Forest, XGBoost compared           | Shows you can justify model choice, not just run one                   |
| Evaluation             | ROC-AUC, F1, Precision-Recall, Confusion Matrix (not accuracy) | Accuracy is misleading on imbalanced data — be ready to explain why    |
| SHAP                   | Global feature importance + individual applicant waterfall     | Turns a black-box model into an explainable "risk intelligence" system |
| Risk Score + Dashboard | 0–100 score, Low/Medium/High bands, business rule overlay      | Bridges ML output to a business decision a loan officer could use      |

## Dashboard Features

- **What-if analysis:** Adjust loan and borrower factors with live risk-score changes.
- **Affordability snapshot:** Estimated EMI, EMI-to-income ratio, and lending recommendation.
- **Batch scoring:** Upload many applicants, download scored results, and review high-risk counts.
- **PDF reports:** Download a formatted risk report for an individual applicant.
- **Model comparison:** Compare Logistic Regression, Random Forest, and XGBoost predictions and test metrics.

## Results Summary (on synthetic data)

- Best model: **Logistic Regression** (ROC-AUC ≈ 0.745)
- Random Forest and XGBoost were close but slightly behind — a good sign
  the underlying signal is fairly linear, and a talking point for
  "why did you pick the simpler model?"

## Key Interview Talking Points

1. **Why not accuracy?** With ~25% default rate, a model predicting "no
   default" for everyone would still be 75% accurate but useless. AUC/F1/Recall
   matter more.
2. **Why SMOTE only on training data?** Applying it before the train/test split
   leaks synthetic information into the test set, inflating performance.
3. **Business framing:** Missing an actual defaulter (false negative) is usually
   costlier for a lender than rejecting a good applicant (false positive) —
   so recall on the "default" class is prioritized.
4. **Explainability:** SHAP lets you tell a rejected applicant _why_ — e.g.
   "high debt-to-income ratio and 2 late payments in the last year drove
   this decision," which regulators increasingly require.
