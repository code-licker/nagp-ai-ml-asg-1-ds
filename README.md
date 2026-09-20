# Customer Churn Prediction
**Data Science Assignment — Customer Churn Prediction**

## Overview
In this project, we predict whether a telecom customer is likely to churn (cancel their service) using the IBM Telco Customer Churn dataset. By identifying at-risk customers early, the company's retention team can reach out to them with special offers, discounts, or support to prevent them from leaving.

We follow the full data science process:
1. Cleaning the data and handling missing/blank values
2. Exploratory Data Analysis (EDA) with 7 charts to understand customer behavior
3. Creating 3 new features (Feature Engineering)
4. Training and comparing Decision Tree models with different settings
5. Evaluating the models and explaining why Recall matters more than Precision for churn
6. Checking feature importance to see what makes customers leave
7. Saving the model pipeline and building a Flask REST API to serve predictions

---

## Project Structure

```
customer_churn_project/
│
├── data/
│   ├── TelcoCustomerChurn.csv                 # Telco customer dataset
│   └── TelcoCustomerChurn - Data Dictionary.csv
├── notebook/
│   └── churn_analysis.ipynb                   # Complete analysis and modeling notebook
├── model/
│   └── churn_model.pkl                        # Saved model pipeline (preprocessing + decision tree)
├── app.py                                     # Flask REST API with POST /predict endpoint
├── features.py                                # Custom feature engineering transformer
├── sample_request.json                        # Example customer input to test the API
├── requirements.txt                           # Required python packages
└── README.md                                  # Setup and project documentation
```

---

## What We Learned from the Data (Key Insights)

1. **Overall Churn Rate**: About 26.5% of customers churned, while 73.5% stayed.
2. **Contract Type Matters Most**: Customers on month-to-month contracts churn at ~42.7%, while customers on 1-year (~11.3%) and 2-year (<3%) contracts almost never leave.
3. **New Customers Leave First**: Most customer churn happens in the first 1 to 5 months. Once customers stay past 20 months, they usually remain loyal.
4. **Higher Monthly Bills = Higher Churn**: Churned customers paid higher monthly bills on average (~$80 vs ~$65).
5. **Add-on Services Keep Customers**: Subscribing to Tech Support or Online Security cuts churn by more than half.
6. **Gender has No Impact**: Both male (26.2%) and female (26.9%) customers churn at almost the exact same rate.
7. **Senior Citizens are at High Risk**: Senior citizens churn at 41.7% compared to 23.6% for younger customers. They are often on fixed incomes and may need simpler tech support.

---

## Why We Prioritize Recall Over Precision

In this assignment, we had to answer:
> *For a telecom company trying to identify customers who may churn, would you prioritize Precision or Recall? Why?*

**Our Answer**: We prioritize **Recall**.
* **Missing a churner (False Negative)** is very expensive. The customer leaves, and the company loses their entire annual subscription revenue ($1,000+), plus the cost to replace them is high.
* **A false alarm (False Positive)** is harmless. The model flags someone who wasn't planning to leave, and the company sends them a friendly email or small discount offer (costing $10–$20).
* Therefore, catching as many actual churners as possible (high Recall) is the best business strategy. Our depth-limited balanced model caught **78.3% of all churners** in the test set.

---

## Bonus Activities & Grand Model Comparison

As suggested in the assignment guidelines, we implemented additional activities:
1. **Hyperparameter Tuning (`GridSearchCV`)**: Optimized Decision Tree depth, leaf sizes, and split criteria with 5-fold cross-validation.
2. **Handling Class Imbalance**: Evaluated `class_weight='balanced'` to prevent the model from ignoring the minority churn class.
3. **Comparing Additional Ensemble Models**: Trained **Random Forest (100 trees, Bagging)** and **Gradient Boosting (Boosting)**.
4. **Financial ROI / Dollar Value Simulation**: Calculated actual net revenue saved based on an annual Customer Lifetime Value ($1,200) and retention offer cost ($25 per customer).

### Grand Comparison Table (Test Set)

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Net Revenue Saved ($) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Decision Tree (Baseline)** | 72.98% | 49.14% | 51.16% | 0.5013 | 0.6605 | $329,800 |
| **Decision Tree (Depth=5, Gini)** | 78.85% | 62.23% | 51.69% | 0.5648 | 0.8187 | $336,350 |
| **Decision Tree (Balanced Weights)** | 72.83% | 49.25% | **76.47%** | **0.5992** | 0.8261 | **$493,025** |
| **Random Forest (100 Trees)** | 79.27% | **66.23%** | 44.74% | 0.5340 | 0.8412 | $291,725 |
| **Gradient Boosting** | **79.79%** | 65.51% | 50.45% | 0.5700 | **0.8428** | $328,800 |

> **Key Financial Finding**: While Gradient Boosting achieves the highest raw accuracy (79.79%), the **Balanced Decision Tree** generates the highest **Net Revenue Saved ($493,025)** — saving the company **over $163,000 more** than the baseline model by catching 76.5% of all actual churners!

---

## Setup and How to Run

### 1. Install Dependencies
Make sure Python 3.10+ is installed, then run:
```bash
pip install -r requirements.txt
```

### 2. Run the Notebook
Open the notebook in Jupyter Notebook, JupyterLab, or VS Code:
```bash
jupyter notebook notebook/churn_analysis.ipynb
```
All cells run top to bottom and show each step with clear explanations and charts.

### 3. Run the Flask API
Start the API server by running:
```bash
python app.py
```
By default, the server starts on `http://127.0.0.1:5000`.

---

## Testing the API

### Endpoint
`POST http://127.0.0.1:5000/predict`

### Sample Request (`sample_request.json`)
```json
{
  "gender": "Female",
  "SeniorCitizen": 0,
  "Partner": "Yes",
  "Dependents": "No",
  "tenure": 1,
  "PhoneService": "No",
  "MultipleLines": "No phone service",
  "InternetService": "DSL",
  "OnlineSecurity": "No",
  "OnlineBackup": "Yes",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "No",
  "StreamingMovies": "No",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 29.85,
  "TotalCharges": 29.85
}
```

### Testing with curl (in another terminal):
```bash
curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d @sample_request.json
```

### Available Test Payloads

Four sample payloads covering distinct customer personas are provided:

| File | Profile Description | Expected Prediction | Churn Probability |
|---|---|:---:|:---:|
| [`sample_request_high_risk.json`](file:///c:/Users/rohanbaranwal/Code/nagp/ml-ai/assignment/DS/customer_churn_project/sample_request_high_risk.json) | Senior, Month-to-month, Fiber optic, No security/support, $95.50/mo | **Yes** | **~81%** |
| [`sample_request.json`](file:///c:/Users/rohanbaranwal/Code/nagp/ml-ai/assignment/DS/customer_churn_project/sample_request.json) | New customer (1 month tenure), DSL, Month-to-month, Electronic check | **Yes** | **~59%** |
| [`sample_request_moderate.json`](file:///c:/Users/rohanbaranwal/Code/nagp/ml-ai/assignment/DS/customer_churn_project/sample_request_moderate.json) | 28 months tenure, 1-Year Contract, Fiber with Tech Support | **No** | **~10%** |
| [`sample_request_loyal.json`](file:///c:/Users/rohanbaranwal/Code/nagp/ml-ai/assignment/DS/customer_churn_project/sample_request_loyal.json) | 65 months tenure, 2-Year Contract, DSL, Full Security suite | **No** | **~1%** |

You can test any profile with curl:
```bash
# High Risk Customer
curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d @sample_request_high_risk.json

# Loyal Customer
curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d @sample_request_loyal.json
```

All 4 profiles are also compiled into [`sample_requests_all.json`](file:///c:/Users/rohanbaranwal/Code/nagp/ml-ai/assignment/DS/customer_churn_project/sample_requests_all.json).

### Error Handling
If you send invalid data (for example, negative tenure `"tenure": -5`), the API will return HTTP 400:
```json
{
  "error": "Tenure cannot be negative"
}
```

