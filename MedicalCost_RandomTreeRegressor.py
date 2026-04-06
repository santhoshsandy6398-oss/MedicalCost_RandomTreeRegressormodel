#Import Libraries

import numpy as np
import pandas as pd

#Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#Models
from sklearn.ensemble import RandomForestRegressor

#Evaluation
from sklearn.metrics import (mean_squared_error, r2_score)

#Load data
df_rf = pd.read_csv('insurance.csv')

# Feature Engineering (map function)
df_rf['sex']=df_rf['sex'].map({'male':1,'female':0})
df_rf['smoker']=df_rf['smoker'].map({'yes':1,'no':0})
df_rf['region']=df_rf['region'].map({'southeast':0,'southwest':1,'northeast':2,'northwest':3})

# Drop unnecessary columns (assuming these are not needed for RandomForest as per previous steps)
df_rf.drop(columns = ['sex', 'children', 'region'], inplace=True, errors='ignore')

#Split features and Target
X = df_rf.drop('charges', axis=1)
y = df_rf['charges']
X_train_rf, x_test_rf, y_train_rf, y_test_rf = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler_rf = StandardScaler()
X_train_rf = scaler_rf.fit_transform(X_train_rf)
x_test_rf = scaler_rf.transform(x_test_rf)

# Initialize and train RandomForestRegressor
rf_model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
rf_model.fit(X_train_rf, y_train_rf)

# Make predictions
y_pred_test_rf = rf_model.predict(x_test_rf)
y_pred_train_rf = rf_model.predict(X_train_rf)

# Evaluate the model
r2_rf_train = r2_score(y_train_rf, y_pred_train_rf)
rmse_rf_train = np.sqrt(mean_squared_error(y_train_rf, y_pred_train_rf))
r2_rf_test = r2_score(y_test_rf, y_pred_test_rf)
rmse_rf_test = np.sqrt(mean_squared_error(y_test_rf, y_pred_test_rf))

print("Random Forest Regressor Evaluation Results:")
print(f"  R2 Score test: {r2_rf_test:.4f}")
print(f"  RMSE test: {rmse_rf_test:.4f}")
print(f"  R2 Score Train: {r2_rf_train:.4f}")
print(f"  RMSE Train: {rmse_rf_train:.4f}")
