
import pandas as pd
from MedicalCost_RandomTreeregressormodel import rf_model, scaler_rf

sample = pd.DataFrame({
    "age": [40],
    "bmi": [30.2],
    "smoker": [0]
})

sample_scaled = scaler_rf.transform(sample)
prediction = rf_model.predict(sample_scaled)

print("✅ Prediction successful")
print("Predicted insurance cost:", prediction[0])
