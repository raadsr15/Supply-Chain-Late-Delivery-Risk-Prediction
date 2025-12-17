# Supply Chain Late Delivery Risk Prediction (ML)

## Overview
This project builds a machine learning system to predict whether an order will be delivered late using real-world supply chain and logistics data. The goal is to help organizations identify high-risk shipments in advance and understand the operational factors that contribute to delivery delays.

The project focuses on clean data preprocessing, leakage-safe target creation, and systematic benchmarking of multiple classification models rather than complex deep learning, making it practical and industry-oriented.

---

## Dataset
**DataCo Smart Supply Chain Dataset**  

This dataset contains historical sales data from **Favorita**, a major grocery retailer in Ecuador. The objective is to predict daily sales for multiple **product families** across different **stores** using time-series data, promotional information, and external economic factors.

The primary training data includes **date**, **store number**, **product family**, **promotion status**, and the target variable **sales**, which represents total daily sales (fractional values are possible). The test dataset contains the **15 days following the final date in the training set**.

Several supplementary datasets provide important contextual information:
- **Store metadata** (city, state, store type, and cluster)
- **Daily oil prices**, reflecting Ecuador’s dependence on oil and its impact on economic activity
- **Holiday and event information**, including transferred holidays, bridge days, and special workdays

Additional real-world factors such as **biweekly public-sector salary payments** and the **April 2016 earthquake in Ecuador** significantly influenced consumer purchasing behavior. These elements make the dataset particularly suitable for realistic demand forecasting and applied machine learning projects.

**Source:**  
Kaggle – Store Sales: Time Series Forecasting  
https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data


---

## Problem Definition
**Task:** Binary classification  
**Target:**  
- `late_delivery = 1` → Order delivered late  
- `late_delivery = 0` → Order delivered on time  

The target is derived from the `delivery_status` column, and all leakage-prone fields are removed before training.

---

## Project Workflow
1. **Data Cleaning**
   - Removal of sensitive and irrelevant fields (PII)
   - Handling missing values
   - Standardized column naming

2. **Exploratory Data Analysis (EDA)**
   - Sales distribution analysis
   - Customer segment and market distribution
   - Delivery status breakdown
   - Shipping mode vs. delivery status analysis

3. **Feature Preparation**
   - Categorical encoding using ordinal encoding
   - Numerical feature scaling
   - Leakage-safe feature selection

4. **Model Benchmarking**
   - Multiple classifiers trained on the same data
   - Stratified train/test split
   - 10-fold cross-validation

5. **Evaluation & Interpretation**
   - Accuracy and cross-validation scores
   - Confusion matrices
   - Precision, recall, and F1-score
   - Feature importance analysis for tree-based models

---

## Models Evaluated
- Logistic Regression (baseline)
- Random Forest
- Extra Trees
- Gradient Boosting
- K-Nearest Neighbors (KNN)
- Decision Tree
- AdaBoost  
- XGBoost 
- LightGBM 

---
