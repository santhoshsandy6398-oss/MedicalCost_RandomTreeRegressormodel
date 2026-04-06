**📊 Insurance Cost Prediction Using Random Forest Regression
**
**📌 Project Overview**

This project focuses on predicting medical insurance charges for individuals based on key personal and lifestyle attributes. 
A Random Forest Regressor, an ensemble machine learning algorithm, is used to learn complex, non‑linear relationships within the data and generate accurate cost predictions.
The model helps understand how factors such as age, body mass index (BMI), and smoking status influence insurance charges.

**🎯 Objective**

To build and evaluate a machine learning model that predicts insurance charges using historical data, enabling data‑driven insights into healthcare cost estimation.

**🧠 Machine Learning Approach**

**Algorithm Used: **Random Forest Regressor
**Problem Type:** Supervised Regression

**Why Random Forest?**

Handles non‑linear relationships effectively
Robust to outliers and noise
Reduces overfitting through ensemble learning

**🗂️ Dataset Description**

The dataset contains information about individuals with the following attributes:

Age: Age of the insured person
BMI: Body Mass Index
Smoker: Smoking status (Yes / No)
Charges: Medical insurance cost (target variable)

Categorical variables are encoded into numerical form, and the dataset is split into training and testing sets for model evaluation.

**⚙️ Data Processing & Feature Engineering**

Removed unnecessary or low‑impact features
Converted categorical features into numerical representations
Split data into training (80%) and testing (20%)
Standardized numerical features for consistency


**📈 Model Evaluation**

The model performance is evaluated using the following metrics:

R² Score – Measures how well the model explains variance in the target variable
RMSE (Root Mean Squared Error) – Measures prediction error magnitude

Evaluation is performed on both training and test datasets to assess generalization and overfitting.

**✅ Results**

The Random Forest Regressor demonstrates strong predictive performance, effectively capturing the relationship between individual attributes and insurance charges. The results show a good balance between bias and variance, indicating a reliable regression model.

**🛠️ Technologies Used**

Python
NumPy
Pandas
Scikit‑learn


**🚀 Future Improvements**

Hyperparameter tuning for better accuracy
Feature importance analysis
Model comparison with Gradient Boosting and XGBoost
Deployment as a web application (Flask / Streamlit)


**👤 Author**

Santhosh P
Application Support Engineer | Aspiring ML Engineer
