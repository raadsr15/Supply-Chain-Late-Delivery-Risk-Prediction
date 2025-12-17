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
