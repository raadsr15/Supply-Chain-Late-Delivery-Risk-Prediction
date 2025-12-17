# Supply Chain Late Delivery Risk Prediction (ML)

## Overview
Timely delivery is a key performance driver in supply chain operations, directly affecting customer satisfaction and operational efficiency. This project develops a machine learning–based solution to identify late delivery risks using real-world logistics and order data from the DataCo Smart Supply Chain dataset.

The workflow begins with structured data cleaning and preprocessing, including the removal of sensitive fields, handling missing values, and standardizing features. Exploratory analysis was conducted to understand customer behavior, regional demand patterns, and delivery performance across different shipping modes. A binary classification target was then created to distinguish late deliveries from on-time shipments, with careful handling of leakage-prone variables.

Multiple machine learning models were benchmarked using a stratified train–test split and cross-validation, including Random Forest, Gradient Boosting, Extra Trees, AdaBoost, KNN, XGBoost, and LightGBM. Tree-based ensemble models delivered consistently strong performance, highlighting clear operational patterns governing delivery delays.

Feature importance analysis showed that shipping timelines, delivery schedules, and shipping modes are the primary drivers of late deliveries, while customer demographics play a secondary role. Overall, the project demonstrates how interpretable machine learning models can support proactive logistics monitoring and data-driven decision-making in supply chain environments.

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


## Results and Model Performance

### Target Distribution
The target variable (`late_delivery`) shows a **moderate class imbalance**:

- **Late deliveries:** ~54.8%  
- **On-time deliveries:** ~45.2%

This indicates that late delivery is a frequent operational issue, reinforcing the relevance of predictive modeling for delay risk mitigation.

---

### Model Benchmarking Results
Multiple machine learning classifiers were evaluated using a **stratified train–test split** and **10-fold cross-validation**. Performance was assessed using accuracy, precision, recall, F1-score, confusion matrices, and cross-validation stability.

#### Overall Performance Summary
| Model | Test Accuracy | CV Accuracy |
|------|--------------|------------|
| Random Forest | 100.00% | 100.00% |
| Gradient Boosting | 100.00% | 100.00% |
| Decision Tree | 100.00% | 100.00% |
| XGBoost | 100.00% | 100.00% |
| LightGBM | 100.00% | 100.00% |
| Extra Trees | 99.99% | 99.98% |
| AdaBoost | 99.71% | 99.67% |
| KNN | 61.93% | 61.44% |

---

### Key Observations

#### 1. Tree-Based Models Dominate
Ensemble tree-based models (Random Forest, Gradient Boosting, Extra Trees, XGBoost, LightGBM) achieved **near-perfect classification performance**, significantly outperforming distance-based methods such as KNN.

This suggests that:
- Delivery delays are governed by **clear, rule-like patterns**
- Non-linear feature interactions play a major role

---

#### 2. Interpreting Perfect Accuracy (Important Note)
The consistently perfect or near-perfect performance across multiple tree-based models indicates that **delivery timing variables strongly encode the outcome**.

In particular:
- `days_for_shipping_real`
- `days_for_shipment_scheduled`

These features are **highly predictive by nature** and effectively define whether a delivery is late. This explains the deterministic performance and highlights a real-world scenario where **post-shipment operational data makes delay detection trivial**.

> This is realistic for *delay detection* systems, but should be carefully handled if the goal is *early delay prediction*.

---

### Feature Importance Analysis

Across all high-performing models, the same dominant predictors consistently emerged:

#### Most Influential Features
1. **Days for Shipping (Actual)**  
2. **Days for Shipment (Scheduled)**  
3. **Shipping Mode**  
4. **Order Status**  

Secondary contributors included:
- Order and customer identifiers
- Geographic features (latitude, longitude, city, country)
- Financial indicators (benefit per order, profit metrics)

#### Example: Random Forest Top Features

<img width="984" height="584" alt="image" src="https://github.com/user-attachments/assets/298305c4-b7ad-4d19-8abd-a14ddba6356b" />


- `days_for_shipping_real` → **56.4%**
- `shipping_mode` → **15.6%**
- `days_for_shipment_scheduled` → **15.2%**
- `order_status` → **5.9%**

This confirms that **delivery delays are primarily driven by logistical execution and shipping policies rather than customer demographics**.

---

### Model Comparison Insights

- **Random Forest, Gradient Boosting, XGBoost, and LightGBM** provide both **high accuracy and stable cross-validation performance**, making them strong candidates for deployment.
- **AdaBoost** performs well but shows slightly reduced robustness.
- **KNN** struggles due to high dimensionality and categorical encoding, making it unsuitable for this dataset.

---

### Business Interpretation
- Faster shipping modes do not guarantee on-time delivery and may increase operational risk.
- Monitoring real vs scheduled shipping days enables **early identification of late deliveries**.
- ML-based delay detection systems can support:
  - Proactive customer communication
  - Dynamic logistics re-routing
  - Operational performance audits

---

### Final Takeaway
This study demonstrates that classical machine learning models, particularly **tree-based ensembles**, can accurately identify late delivery risks in supply chain data. Feature importance analysis highlights that **operational and logistical variables dominate delivery outcomes**, providing actionable insights for supply chain optimization.

## Installation and Usage

1. **Clone the repository:**
```bash
git clone https://github.com/raadsr15/Supply Chain Late Delivery Risk Prediction (ML).git
cd supplychain-late-delivery-risk

2. **Install dependencies:**

  ```bash
  pip install -r requirements.txt
  ```

3. Train the models and obtain results
   
```bash
jupyter notebook Supply Chain Late Delivery Risk Prediction (ML).ipynb
```
