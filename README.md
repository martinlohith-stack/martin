import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
import shap
import matplotlib.pyplot as plt

# ------------------------------
# Step 1: Simulate Machine Sensor Data
# ------------------------------

np.random.seed(42)

data = pd.DataFrame({
    "temperature": np.random.normal(70, 5, 500),
    "vibration": np.random.normal(30, 3, 500),
    "pressure": np.random.normal(100, 10, 500),
    "motor_current": np.random.normal(15, 2, 500)
})

# Introduce abnormal behavior
data.iloc[480:500] = data.iloc[480:500] + np.array([20,15,-30,10])

print("Sample Data:")
print(data.head())

# ------------------------------
# Step 2: Train Anomaly Detection Model
# ------------------------------

model = IsolationForest(contamination=0.05, random_state=42)
model.fit(data)

# Predict anomalies
data["anomaly"] = model.predict(data)

# Convert -1 to anomaly
data["anomaly"] = data["anomaly"].apply(lambda x: 1 if x == -1 else 0)

print("\nDetected anomalies:")
print(data[data["anomaly"] == 1].head())

# ------------------------------
# Step 3: Root Cause Explanation
# ------------------------------

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(data.drop("anomaly", axis=1))

# Plot explanation
shap.summary_plot(shap_values, data.drop("anomaly", axis=1))

# ------------------------------
# Step 4: Generate Engineering Explanation
# ------------------------------

def explain_failure(row):

    reasons = []

    if row["vibration"] > 40:
        reasons.append("High vibration detected (possible bearing wear)")

    if row["temperature"] > 90:
        reasons.append("High temperature detected (possible overheating)")

    if row["pressure"] < 70:
        reasons.append("Low pressure detected (possible valve issue)")

    if row["motor_current"] > 25:
        reasons.append("High motor current detected (possible motor overload)")

    if reasons:
        return reasons
    else:
        return ["Unknown anomaly"]

print("\nFailure explanations:\n")

anomalies = data[data["anomaly"] == 1]

for i, row in anomalies.head(5).iterrows():

    print(f"Machine Event {i}")

    reasons = explain_failure(row)

    for r in reasons:
        print("-", r)

    print("Recommended Action: Inspect machine components\n")
