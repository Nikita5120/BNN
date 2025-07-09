# Bayesian Churn Predictor

A churn prediction system using Bayesian Neural Networks (BNNs) that provides uncertainty estimates for each prediction.

## Features
- BNN with variational inference
- Monte Carlo dropout-based confidence
- SHAP explainability
- Streamlit dashboard

## How to Run
1. Clone repo and install dependencies
2. Train model using `train_model.ipynb`
3. Run `streamlit run app/main.py`

## Dataset
- [Telco Churn Dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)