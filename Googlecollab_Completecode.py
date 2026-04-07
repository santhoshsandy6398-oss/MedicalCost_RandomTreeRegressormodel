
# Import Libraries
import numpy as np
import pandas as pd

# Preprocessing
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Models
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR, SVC

# Evaluation
from sklearn.metrics import mean_squared_error, r2_score

# Load data
df = pd.read_csv("insurance.csv")
print(df.head())

# Encoding categorical variables
df["sex"] = df["sex"].map({"male": 1, "female": 0})
df["smoker"] = df["smoker"].map({"yes": 1, "no": 0})
df["region"] = df["region"].map({
    "southeast": 0,
    "southwest": 1,
    "northeast": 2,
    "northwest": 3
})

print(df.head())

# Correlation check
print(df.corr()["charges"])

# Drop selected columns
df.drop(columns=["sex", "children", "region"], inplace=True)
print(df.head())

# Split features and target
X_train, x_test, y_train, y_test = train_test_split(
    df.drop("charges", axis=1),
    df["charges"],
    test_size=0.2,
    random_state=42
)

# Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
x_test = scaler.transform(x_test)

# Model selection
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree Regressor": DecisionTreeRegressor(
        max_depth=5,
        random_state=42
    ),
    "Random Forest Regressor": RandomForestRegressor(
        n_estimators=100,
        max_depth=5,
        random_state=42
    ),
    "KNN Regressor": KNeighborsRegressor(),
    "SVM": SVR(kernel="linear")
}

results = {}

# Train and evaluate models
for name, model in models.items():
    model.fit(X_train, y_train)

    # Test predictions
    y_pred_test = model.predict(x_test)
    r2_test = r2_score(y_test, y_pred_test)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))

    # Train predictions
    y_pred_train = model.predict(X_train)
    r2_train = r2_score(y_train, y_pred_train)
    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))

    results[name] = {
        "R2 Score (Test)": r2_test,
        "RMSE (Test)": rmse_test,
        "R2 Score (Train)": r2_train,
        "RMSE (Train)": rmse_train
    }

print("Model Evaluation Results:")
for name, metrics in results.items():
    print(name)
    for metric_name, value in metrics.items():
        print(f"  {metric_name}: {value:.4f}")
