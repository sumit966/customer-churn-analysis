# Customer Churn Analysis and Prediction

End-to-end churn analytics project on 5,000 telecom customers. Cleans and models raw data, trains logistic regression + random forest classifiers, ranks the top churn drivers with feature importance, and serves predictions via a FastAPI endpoint.

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)

## Overview

A complete churn analytics pipeline that:

1. **Generates** 5,000 synthetic telecom customer records with realistic churn logic
2. **Preprocesses** categorical + numeric features with OneHotEncoder + StandardScaler
3. **Trains** Logistic Regression and Random Forest on stratified train/test split
4. **Evaluates** with Accuracy, Precision, Recall, F1, and AUC-ROC
5. **Ranks churn drivers** by feature importance
6. **Generates charts** for churn by contract, tenure, payment method, and driver importance
7. **Writes a Markdown report** with KPIs, model performance, and retention recommendations
8. **Serves predictions** via a FastAPI endpoint with risk levels

## Problem Statement

Customer churn is the silent killer of subscription businesses:

- **5% churn increase = 25%+ profit loss** (classic Bain & Company finding)
- Most companies discover churn **after it happens** — too late to intervene
- Generic retention campaigns waste budget on low-risk customers
- Without ranked drivers, teams don't know **which factor to fix first**

Raw transaction data alone doesn't answer these questions. Structured analysis and prediction do.

## Solution

A churn analytics pipeline that:

- Predicts per-customer churn probability in milliseconds
- Classifies each customer as Low / Medium / High risk
- Ranks the top churn drivers so retention teams know what to prioritize
- Produces charts for stakeholder communication
- Serves predictions over HTTP for CRM integration

## Architecture

Raw Customer Data (5000 records)
         |
         v
Preprocessing (OneHotEncoder + StandardScaler)
         |
         v
Stratified Train/Test Split
         |
         v
+--------------------+
| Logistic Regression |
| Random Forest (Best)|
+--------------------+
         |
         v
Evaluate (Acc, Prec, Recall, F1, AUC-ROC)
         |
         v
Rank Churn Drivers (feature importance)
         |
         v
Charts + Markdown Report
         |
         v
FastAPI Endpoint: /predict  /drivers  /metrics
         |
         v
Docker + GitHub Actions CI

## Tech Stack

| Category | Technologies |
|----------|-------------|
| ML | Scikit-learn (LogisticRegression, RandomForest) |
| Data | Pandas, NumPy |
| Preprocessing | ColumnTransformer, OneHotEncoder, StandardScaler |
| Visualization | Matplotlib, Seaborn |
| API | FastAPI, Uvicorn, Pydantic |
| Testing | pytest, httpx |
| Containerization | Docker |
| CI/CD | GitHub Actions |
| Language | Python 3.11+ |

## Project Structure

customer-churn-analysis/
├── src/
│   ├── __init__.py
│   ├── data_generator.py      # Generate synthetic churn dataset
│   ├── preprocess.py          # Encode + scale features
│   ├── train.py               # Train + rank drivers
│   ├── evaluate.py            # Evaluate saved model
│   └── report.py              # Charts + Markdown report
├── api/
│   ├── __init__.py
│   └── main.py                # FastAPI prediction service
├── tests/
│   ├── __init__.py
│   └── test_api.py            # pytest tests
├── data/                       # Generated CSVs (git-ignored)
├── models/                     # Saved models (git-ignored)
├── reports/                    # Charts + report (git-ignored)
├── .github/workflows/ci.yml
├── Dockerfile
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md

## Quick Start

### 1. Clone

git clone https://github.com/sumit966/customer-churn-analysis.git
cd customer-churn-analysis

### 2. Virtual environment

Windows:
python -m venv venv
venv\Scripts\activate

macOS / Linux:
python3 -m venv venv
source venv/bin/activate

### 3. Install dependencies

pip install -r requirements.txt

### 4. Generate dataset

python src/data_generator.py

Output:
[OK] Generated 5000 customers -> data/churn.csv
[OK] Churn rate: XX.XX%

### 5. Train models

python src/train.py

Output:
[Training] logistic_regression...
   Accuracy:  0.XX
   AUC-ROC:   0.XX
[Training] random_forest...
   Accuracy:  0.XX
   AUC-ROC:   0.XX

[Best] random_forest (AUC-ROC: 0.XX)

[Top churn drivers]
   cat__contract_Month-to-month: 0.XX
   num__tenure: 0.XX
   ...

### 6. Generate charts + report

python src/report.py

Creates:
reports/
├── churn_by_contract.png
├── churn_by_tenure.png
├── churn_by_payment.png
├── top_churn_drivers.png
└── report.md

### 7. Start the API

uvicorn api.main:app --reload

API runs at http://localhost:8000

### 8. Open Swagger UI

http://localhost:8000/docs

## API Usage

### POST /predict

Request:
{
  "gender": "Male",
  "senior_citizen": 0,
  "partner": "Yes",
  "dependents": "No",
  "tenure": 12,
  "contract": "Month-to-month",
  "internet_service": "Fiber optic",
  "tech_support": "No",
  "payment_method": "Electronic check",
  "paperless_billing": "Yes",
  "monthly_charges": 95.5,
  "total_charges": 1146.0
}

Response:
{
  "churn_probability": 0.72,
  "risk_level": "High",
  "top_drivers": [
    {"feature": "cat__contract_Month-to-month", "importance": 0.24},
    {"feature": "num__tenure", "importance": 0.18},
    {"feature": "cat__internet_service_Fiber optic", "importance": 0.14}
  ]
}

### GET /drivers

Returns the top churn drivers ranked by feature importance.

### GET /metrics

Returns all model metrics from training (accuracy, precision, recall, F1, AUC-ROC).

### Other Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| / | GET | API info |
| /health | GET | Health check |
| /docs | GET | Swagger UI |
| /redoc | GET | Alternative docs |

## Model Performance

| Model | Accuracy | Precision | Recall | F1 | AUC-ROC |
|-------|----------|-----------|--------|----|---------|
| Logistic Regression | 0.XX | 0.XX | 0.XX | 0.XX | 0.XX |
| Random Forest | 0.XX | 0.XX | 0.XX | 0.XX | 0.XX |

(Run `python src/train.py` to see exact numbers on your machine.)

## Top Churn Drivers

Based on feature importance from the trained random forest:

1. **Contract = Month-to-month** — highest churn driver by far
2. **Tenure** — first 12 months are the riskiest
3. **Internet Service = Fiber optic** — higher churn than DSL
4. **Payment Method = Electronic check** — churn-heavy payment
5. **Tech Support = No** — support users stick around
6. **Monthly Charges** — higher charges correlate with churn

## Business Insights

| Finding | Recommendation |
|---------|----------------|
| Month-to-month customers churn 3x more | Offer annual plan discounts |
| First-year customers are highest risk | Build onboarding + early-engagement program |
| Fiber optic users churn more | Investigate pricing or service quality issues |
| Electronic-check payers churn more | Incentivize auto-pay migration |
| Tech-support subscribers churn less | Bundle support into base plans |

## Testing

pytest tests/ -v

Tests cover:
- Root endpoint responds
- Health check reports model status
- /predict returns valid probability (0 to 1) and risk level
- Invalid payload (missing fields) returns 422

## Docker

Build:
docker build -t customer-churn-analysis .

Run:
docker run -p 8000:8000 customer-churn-analysis

The Dockerfile generates data and trains the model automatically during build.

## CI/CD Pipeline

Every push to main triggers GitHub Actions:

1. Install Python 3.11 + dependencies
2. Generate synthetic dataset
3. Train Logistic Regression + Random Forest
4. Run pytest test suite

See .github/workflows/ci.yml.

## Key Learnings

- Contract type is the single strongest churn predictor — behavioral, not demographic
- Tenure buckets outperform raw tenure — churn risk is nonlinear
- Feature importance from random forest beats correlation for driver ranking
- OneHotEncoder + StandardScaler in ColumnTransformer keeps preprocessing clean and reusable
- Stratified split preserves class balance between train and test
- AUC-ROC is the right metric for imbalanced classification (churn is minority class)
- Per-customer probability + risk level lets CRM teams prioritize outreach

## Future Improvements

- SHAP values for per-customer explanations
- XGBoost / LightGBM comparison
- Survival analysis (Cox proportional hazards)
- Uplift modeling — estimate impact of retention offers
- Real dataset (Kaggle Telco Customer Churn)
- Deploy to GCP Cloud Run
- Real-time scoring integration with CRM
- A/B test retention campaigns

## License

MIT License - see LICENSE file.

## Author

Sumit Raj
- M.Tech Applied AI & ML @ VNIT Nagpur
- Ex-Software Engineer Intern @ Salesforce
- GitHub: https://github.com/sumit966
- LinkedIn: https://www.linkedin.com/in/er-sumit-raj-/
- Portfolio: https://sumit966-github-io.vercel.app
- Email: info.sr0909@gmail.com

If you found this project useful, please consider giving it a star!
