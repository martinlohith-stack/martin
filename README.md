import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
import shap
import matplotlib.pyplot as plt

# --- Step 1: Data Simulation ---
np.random.seed(42)
n_samples = 500
features = ["temperature", "vibration", "pressure", "motor_current"]

data = pd.DataFrame({
    "temperature": np.random.normal(70, 5, n_samples),
    "vibration": np.random.normal(30, 3, n_samples),
    "pressure": np.random.normal(100, 10, n_samples),
    "motor_current": np.random.normal(15, 2, n_samples)
})

# Simulate a specific failure: Bearing Degradation + Overheating
# Indices 480-500 will show high vibration and temp, low pressure
data.iloc[480:500] += np.array([25, 18, -35, 12])

# --- Step 2: Isolation Forest Model ---
model = IsolationForest(contamination=0.05, random_state=42)
model.fit(data)
data["anomaly_score"] = model.decision_function(data) # Lower = more anomalous
data["is_anomaly"] = model.predict(data.drop(columns="anomaly_score"))
data["is_anomaly"] = data["is_anomaly"].map({1: 0, -1: 1})

# --- Step 3: SHAP Explainer (The "Why") ---
# Isolation Forest output for SHAP is the anomaly score
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(data[features])

# --- Step 4: Autonomous Diagnostic Engine ---
def autonomous_diagnosis(idx, row, shap_vals):
    """
    Combines SHAP feature importance with engineering thresholds
    to provide a root cause diagnosis.
    """
    # Identify the top contributing feature from SHAP
    feature_importance = dict(zip(features, shap_vals[idx]))
    # For Isolation Forest, SHAP values are often negative for anomalies 
    # (contributing to a lower decision score)
    top_contributor = min(feature_importance, key=feature_importance.get)
    
    diagnostics = {
        "vibration": "Critical Bearing Wear / Misalignment",
        "temperature": "Thermal Overload / Cooling System Failure",
        "pressure": "Pneumatic Leak / Valve Seizure",
        "motor_current": "Electrical Short / Mechanical Obstruction"
    }

    print(f"--- Diagnostic Report: Event {idx} ---")
    print(f"Status: ANOMALY DETECTED")
    print(f"Primary Driver: {top_contributor.upper()} (SHAP: {feature_importance[top_contributor]:.4f})")
    print(f"Root Cause Analysis: {diagnostics.get(top_contributor)}")
    
    # Logic-based verification
    if row["temperature"] > 90 and row["vibration"] > 40:
        print("Confidence: HIGH - Multi-sensor correlation confirmed.")
    
    print(f"Recommended Action: Dispatch Technician for {top_contributor} inspection.\n")

# Run diagnosis on detected anomalies
anomaly_indices = data[data["is_anomaly"] == 1].index

for idx in anomaly_indices[:3]: # Review first 3 anomalies
    autonomous_diagnosis(idx, data.loc[idx], shap_values)

# Visualizing the SHAP contribution for a single anomaly
# This shows exactly how much each feature pushed the model toward an 'Anomaly' decision
shap.initjs()
print("Visualizing contribution for the first detected anomaly:")
shap.force_plot(explainer.expected_value, shap_values[anomaly_indices[0]], data[features].iloc[anomaly_indices[0]])
