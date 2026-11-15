# RetainIQ – Cost-Sensitive Customer Retention Intelligence

RetainIQ is an advanced machine learning-based customer churn prediction and cost-sensitive evaluation platform designed for subscription-based and telecommunication companies. 

---

## 📋 Problem Statement
Telecommunication and subscription-based companies lose significant revenue when customers discontinue their services (customer churn). While companies frequently use machine learning to predict churn, traditional evaluation metrics like accuracy, precision, recall, or F1-score fail to capture the **financial business consequences** of prediction errors:
* **False Negatives (FN):** Failing to identify a customer who actually churns results in a missed retention opportunity and lost lifetime value.
* **False Positives (FP):** Incorrectly identifying a loyal customer as likely to churn leads to wasted expenditure on unnecessary retention discounts or offers.

A model with higher classification accuracy does not necessarily produce the lowest business cost.

---

## 💡 Proposed Solution
RetainIQ solves this by introducing a **cost-sensitive evaluation framework**. Instead of selecting models purely based on statistical metrics, RetainIQ:
1. Analyzes historical customer records.
2. Trains and compares four distinct machine learning algorithms.
3. Performs validation-based **decision threshold optimization** to minimize financial cost rather than blindly using a 0.5 threshold.
4. Ranks models using an explicitly defined business cost matrix.
5. Provides an interactive SaaS analytics dashboard to explore churn risks, evaluate models, simulate cost adjustments, and visualize customer insights.

---

## 📊 Dataset Description & Source
* **Dataset:** IBM Telco Customer Churn Dataset (Kaggle)
* **Records:** 7,043 customer records
* **Features:** 21 columns (demographics, account info, services subscribed, charges)
* **Target:** `Churn` (Yes / No)

---

## 🛠️ Technologies Used
* **Programming Language:** Python
* **Data Processing & ML:** Pandas, NumPy, Scikit-Learn
* **Explainability:** SHAP PermutationExplainer
* **Model Storage:** Joblib
* **Frontend & Visualization:** Streamlit, Plotly

---

## 🤖 Machine Learning Algorithms Trained
1. **Logistic Regression** (with feature scaling and pipeline preprocessing)
2. **Random Forest Classifier** (ensemble decision trees)
3. **Gradient Boosting Classifier** (boosting-based iterative error correction)
4. **Neural Network (MLPClassifier)** (multi-layer perceptron with tuned hidden layers)

---

## 💰 Cost-Sensitive Evaluation Framework
RetainIQ defines a customizable business cost matrix:
* **False Negative (FN):** $100 (Missed churn opportunity)
* **False Positive (FP):** $10 (Unnecessary retention offer)
* **True Positive (TP):** $0
* **True Negative (TN):** $0

Models are evaluated by calculating:
$$\text{Estimated Total Cost} = (\text{FN} \times \text{FN Cost}) + (\text{FP} \times \text{FP Cost})$$

---

## 📂 Project Folder Structure
RetainIQ/
│
├── data/
│   └── telco_churn.csv
│
├── models/
│   ├── preprocessor.pkl
│   ├── logistic_regression.pkl
│   ├── random_forest.pkl
│   ├── gradient_boosting.pkl
│   ├── neural_network.pkl
│   ├── evaluation_results.json
│   └── predictions.pkl
│
├── src/
│   ├── preprocessing.py
│   ├── prediction_service.py
│   ├── bulk_prediction.py
│   ├── train_models.py
│   ├── evaluation.py
│   ├── explainability.py
│   └── __init__.py
│
├── app.py
├── requirements.txt
└── README.md

---

## 🚀 Installation & Execution Instructions

### 1. Install Dependencies
Open your terminal in the project root folder and install the required packages:
```powershell
python -m pip install -r requirements.txt
```

### Saved-model prediction service

The shared inference service in `src/prediction_service.py` uses the four saved
preprocessing-and-classifier pipelines in `models/` and the matching
validation-selected thresholds in `models/evaluation_results.json`. It does not
train models, select new thresholds, or use test data. The separate
`models/preprocessor.pkl` is preserved; inference uses the preprocessor bundled
inside each saved model pipeline.

Pass a pandas DataFrame containing all 19 feature columns listed in
`src/preprocessing.py`. `customerID` is optional and is preserved in the
results when supplied. Do not include the `Churn` target column. Numeric and
categorical input types are checked; missing feature values are reported and
handled only by the imputers already fitted in the saved pipelines. The
returned churn scores are model outputs and are not asserted to be calibrated
probabilities.

```python
import pandas as pd

from src.prediction_service import load_prediction_assets, predict_customers

assets = load_prediction_assets()
customer_rows = pd.read_csv("your_customers.csv")
batch = predict_customers(customer_rows, "Logistic Regression", assets)

print(batch.results)
print("Missing values handled by the saved pipeline:", batch.missing_value_counts)
```

### Bulk CSV prediction

The **Bulk CSV Prediction** dashboard page scores uploaded UTF-8 CSV files
(UTF-8 BOM is also supported). Include all 19 feature columns defined in
`src/preprocessing.py` (in any order):

`SeniorCitizen`, `tenure`, `MonthlyCharges`, `TotalCharges`, `gender`,
`Partner`, `Dependents`, `PhoneService`, `MultipleLines`, `InternetService`,
`OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`,
`StreamingTV`, `StreamingMovies`, `Contract`, `PaperlessBilling`,
`PaymentMethod`.

`customerID` is optional, and the `Churn` target column must not be included.
Blank or missing `customerID` values are rejected. Unexpected or missing
feature columns are rejected.
Missing feature values are reported and handled by the imputers already fitted
in the selected model pipeline. Malformed files and missing or unexpected
feature columns are rejected with an error.

Select one of the four saved models. Each row is scored with that model's
existing validation-selected threshold. Results remain in source row order;
the risk-rank column sorts scores conceptually, with earlier rows winning ties.
When `customerID` is absent, the customer identifier column uses the
one-based CSV data-row number (excluding the header). Model churn scores are
not claimed to be calibrated probabilities, and no retention savings are
estimated.

After scoring, **Download CSV** exports the existing results without running
predictions again. The report includes the customer identifier, original
one-based row number, selected model, saved threshold, model churn score,
predicted class, and risk rank, in original input order. The filename is
model-specific, for example
`retainiq_bulk_predictions_logistic_regression.csv`.

The report and its pre-generated CSV are held in Streamlit session memory only;
they are not written to disk or a database and are cleared when a new scoring
attempt begins or the user logs out. Exported text fields that could be
interpreted as spreadsheet formulas are prefixed with an apostrophe in the CSV
only. This does not change model inputs or the displayed prediction results.

### Individual score explanations

The **Customer Churn Prediction** page can generate an explanation for the
most recently scored customer when **Generate explanation** is clicked. It uses
the selected saved pipeline and the shared prediction service; explanations
are not generated for every row in a bulk upload. The model churn score, its
saved decision threshold, predicted class, SHAP base value, and per-feature
contributions are displayed separately.

SHAP contributions use the 19 raw input features and a deterministic
20-customer background sampled only from the reconstructed training split.
They describe how the model attributes movement in its score relative to that
background; they are not causes, causal effects, or evidence that an
intervention will work. Model churn scores are not claimed to be calibrated
probabilities. The saved pipelines, preprocessing, and validation-selected
thresholds are not retrained or changed for explanations.
