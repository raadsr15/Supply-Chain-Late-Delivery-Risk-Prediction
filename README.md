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

## Exploratory Data Analysis (EDA)

This section summarizes key insights obtained from exploratory analysis of the supply chain dataset. The goal of EDA was to understand sales behavior, customer composition, market distribution, and delivery performance before model development.

---

### 1. Sales Distribution

<img width="984" height="584" alt="image" src="https://github.com/user-attachments/assets/fa14e5b6-afbc-453a-b838-05b32b5f8643" />

- Sales values are **right-skewed**, with the majority of transactions occurring at **lower sales volumes**.
- A small number of orders exhibit **very high sales values**, indicating the presence of **outliers** or bulk purchases.
- This skewness suggests that scaling and robust models are necessary to handle extreme values effectively.

**Insight:**  
Most orders are small to medium in size, while a limited number of high-volume transactions can disproportionately influence model learning.

---

### 2. Customer Segment Distribution

<img width="984" height="584" alt="image" src="https://github.com/user-attachments/assets/6ac112a8-25f3-4721-8fc4-2fd44e7bec26" />

- The **Consumer** segment dominates the dataset, accounting for **over half of all orders**.
- **Corporate** customers form the second-largest segment.
- **Home Office** customers represent the smallest share.

**Insight:**  
Supply chain operations are primarily driven by consumer demand, but corporate orders still form a significant portion and may exhibit different delivery patterns.

---

### 3. Market Distribution

<img width="795" height="784" alt="image" src="https://github.com/user-attachments/assets/88eec0c2-7370-49a6-8916-64f7deb9d314" />

- **LATAM** and **Europe** are the largest markets by order volume.
- **Pacific Asia** shows moderate activity.
- **USCA** and **Africa** contribute relatively fewer orders.

**Insight:**  
Regional differences in demand and logistics volume suggest that geographic features are important predictors of delivery performance.

---

### 4. Delivery Status Distribution

<img width="984" height="584" alt="image" src="https://github.com/user-attachments/assets/9a781647-3a37-4b53-9dea-c98b1fbe002a" />

- A large proportion of orders are labeled as **Late Delivery**, significantly outnumbering on-time deliveries.
- **Advance shipping** and **shipping on time** occur less frequently.
- **Shipping canceled** represents a small fraction of total orders.

**Insight:**  
Late deliveries are a major operational issue in the dataset, justifying the need for predictive models focused on delay risk identification.

---

### 5. Delivery Status by Shipping Mode

<img width="623" height="464" alt="image" src="https://github.com/user-attachments/assets/1e78c20e-1784-4ff6-b5ee-4f7352cee7da" />

- **First Class** shipping shows a **very high percentage of late deliveries**, indicating possible over-promising on fast delivery.
- **Same Day** shipping has a comparatively higher proportion of **on-time deliveries**, but still suffers from delays.
- **Second Class** shipping is heavily affected by late deliveries.
- **Standard Class** shows the **best balance**, with a higher share of advance and on-time shipments.

**Insight:**  
Shipping mode plays a critical role in delivery outcomes. Faster shipping options do not necessarily guarantee timely delivery and may introduce higher delay risks.

---

### Overall EDA Takeaways
- The dataset exhibits **class imbalance** in delivery outcomes, particularly with late deliveries dominating.
- Customer segment, market, and shipping mode all show strong relationships with delivery performance.
- These patterns justify the use of **machine learning classification models** and emphasize the importance of feature selection and model interpretability.

---
