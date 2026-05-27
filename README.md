# 📊 Customer Churn Predictor

A machine learning web application that predicts whether a telecom
customer is likely to churn (leave the company) based on their
account details and service usage.

## 🔗 Live Demo
[Click here to try the app](https://customer-churn-predictor-akzrpxgblzm2f5wgyvsrd9.streamlit.app/)

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
- **Scikit-learn** — preprocessing and model evaluation
- **XGBoost** — final prediction model (tuned with GridSearchCV)
- **Streamlit** — interactive web interface
- **Streamlit Community Cloud** — deployment

## 📈 Model Performance (Tuned XGBoost)
| Metric    | Score  |
|-----------|--------|
| Accuracy  | 74.8% |
| Precision | 51.64% |
| Recall    | 80.21% |
| F1 Score  | 62.83% |
| AUC-ROC   | 84.50% |

## 🔍 Key Findings
- Month-to-month contracts have ~43% churn rate vs ~3% for two-year contracts
- New customers (low tenure) are most at risk of churning
- Higher monthly charges correlate with higher churn
- Lack of tech support and online security increases churn risk
- Fiber optic internet customers churn more than DSL customers
- Electronic check payment method has the highest churn rate

## 📂 Project Structure
customer-churn-predictor/
├── app.py                # Streamlit web application
├── best_model.json       # Trained XGBoost model
├── scaler_params.json    # Feature scaling parameters
├── feature_names.json    # Feature names for input processing
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation
