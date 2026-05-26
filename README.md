# 📊 Customer Churn Predictor

A machine learning web application that predicts whether a telecom
customer is likely to churn (leave the company) based on their
account details and service usage.

## 🔗 Live Demo
[Click here to try the app](https://your-username-customer-churn-predictor.streamlit.app)

## 📌 Problem Statement
Customer churn costs telecom companies millions in lost revenue.
This tool helps businesses identify at-risk customers so they can
take proactive retention measures.

## 📊 Dataset
[Telco Customer Churn - Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- 7,043 customer records
- 21 features including demographics, services, and billing info

## 🛠️ Tech Stack
- **Python** — core language
- **Pandas & NumPy** — data cleaning and manipulation
- **Matplotlib, Seaborn & Plotly** — data visualization
- **Scikit-learn** — preprocessing, model evaluation
- **XGBoost** — final prediction model (tuned with GridSearchCV)
- **Streamlit** — interactive web interface
- **Streamlit Community Cloud** — deployment

## 📈 Model Performance
| Metric    | Score |
|-----------|-------|
| Accuracy  | XX%   |
| Precision | XX%   |
| Recall    | XX%   |
| F1 Score  | XX%   |
| AUC-ROC   | XX%   |

## 🔍 Key Findings
- Month-to-month contracts have ~43% churn rate vs ~3% for two-year
- New customers (low tenure) are most at risk
- Higher monthly charges correlate with higher churn
- Lack of tech support and online security increases churn risk

## 📂 Project Structure
- `app.py` — Streamlit web application
- `best_model.pkl` — trained XGBoost model
- `preprocessed_data.pkl` — scaler and feature names for input processing
- `requirements.txt` — Python dependencies
