# Credit Card Fraud Detection

## Overview

This project is an AI/ML-based Credit Card Fraud Detection System that identifies potentially fraudulent transactions using machine learning.

The project focuses on handling highly imbalanced transaction data and improving fraud detection by comparing multiple machine learning models, applying imbalance-handling techniques, and optimizing the classification threshold.

## Features

- Exploratory Data Analysis (EDA)
- Data preprocessing and feature scaling
- Logistic Regression baseline
- Random Forest classification
- XGBoost classification
- Class-weighted machine learning
- SMOTE for handling class imbalance
- Classification threshold optimization
- Model evaluation using fraud-focused metrics
- SHAP-based model explainability
- Saved trained model for prediction
- Streamlit web application

## Dataset

The project uses the Credit Card Fraud Detection dataset.

The dataset contains:

- 284,807 transactions
- 30 input features
- 1 target variable (`Class`)
- 284,315 legitimate transactions
- 492 fraudulent transactions

The `Class` column represents:

- `0` → Legitimate transaction
- `1` → Fraudulent transaction

The dataset is highly imbalanced, making accuracy alone unsuitable for evaluating the model.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Imbalanced-learn
- Matplotlib
- Seaborn
- SHAP
- Joblib
- Streamlit

## Machine Learning Workflow

```text
Raw Data
    ↓
Data Exploration
    ↓
Train/Test Split
    ↓
Feature Scaling
    ↓
Baseline Models
    ↓
Class Imbalance Handling
    ↓
SMOTE
    ↓
XGBoost
    ↓
Threshold Optimization
    ↓
Model Evaluation
    ↓
SHAP Explainability
    ↓
Model Saving
    ↓
Streamlit Application
