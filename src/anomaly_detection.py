import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import joblib
import os

# ==============================
# Load Dataset
# ==============================

df = pd.read_csv("../data/city_activity_scores.csv")

# ==============================
# Features for Anomaly Detection
# ==============================

features = [
    "Traffic_Density",
    "Pedestrian_Count",
    "Vehicle_Count",
    "Public_Transport",
    "Activity_Score"
]

X = df[features]

# ==============================
# Standardize the Data
# ==============================

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ==============================
# Train Isolation Forest
# ==============================

model = IsolationForest(
    contamination=0.05,
    random_state=42
)

df["Anomaly"] = model.fit_predict(X_scaled)

# ==============================
# Save Isolation Forest Model
# ==============================

os.makedirs("../models", exist_ok=True)

joblib.dump(model, "../models/isolation_forest.pkl")

print("✅ Isolation Forest model saved successfully!")

# ==============================
# Convert Labels
# ==============================

df["Status"] = df["Anomaly"].map({
    1: "Normal",
    -1: "Anomaly"
})

# ==============================
# Save Final Dataset
# ==============================

df.to_csv("../data/city_final.csv", index=False)

# ==============================
# Display Results
# ==============================

print("\nAnomaly Detection Completed!\n")

print(df[["Location_ID", "Zone", "Activity_Score", "Status"]].head(20))

print("\nTotal Anomalies:", len(df[df["Status"] == "Anomaly"]))

print("\n✅ Final dataset saved successfully!")