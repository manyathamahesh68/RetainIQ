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
│   ├── train_models.py
│   ├── evaluation.py
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
python -m pip install pandas numpy scikit-learn plotly streamlit joblib
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