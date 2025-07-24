Objective:
To build a machine learning model that predicts the LTV of a customer using transactional data, and segments them as High, Medium, or Low value customers.

Files Included:
model.py — Python script used to preprocess data
train the model
generate outputs
ltv_model.pkl — Saved trained model using joblib
customer_ltv_features.csv — Final dataset with predicted LTV and segment labels
ltv_predictions.csv — Test set predictions
LTV_Report.pdf — Final report containing project summary

Tools & Libraries Used
Python
Pandas
NumPy
Matplotlib
Seaborn
Scikit-learn (Random Forest)
XGBoost
Joblib

Features Used
Recency: Days since last purchase
Frequency: Number of unique purchases
AOV: Average Order Value (total spent / invoices)

Steps Followed
Data Preprocessing & Cleaning
Feature Engineering (Recency, Frequency, AOV)
LTV Calculation & Labeling
Model Training (Random Forest, XGBoost)
Evaluation using MAE, RMSE, R²
Visualization of predictions and feature importance
Model and results export
LTV Segmentation Logic
High: LTV > 1000
Medium: 500 < LTV ≤ 1000
Low: LTV ≤ 500

How to Use
Run model.py to train the model and save predictions
Use joblib.load("ltv_model.pkl") in any other script to reuse the trained model
Input new customer data to predict their LTV

