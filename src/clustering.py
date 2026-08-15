import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import joblib
import os

# ==============================
# Load Dataset
# ==============================

df = pd.read_csv("../data/city_activity.csv")

# ==============================
# Features Used for Clustering
# ==============================

features = [
    "Traffic_Density",
    "Pedestrian_Count",
    "Vehicle_Count",
    "Public_Transport",
    "Average_Speed",
    "Commercial_Score",
    "Residential_Score",
    "Green_Space",
    "Morning_Activity",
    "Afternoon_Activity",
    "Evening_Activity",
    "Night_Activity"
]

X = df[features]

# ==============================
# Scale the Data
# ==============================

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ==============================
# Apply K-Means Clustering
# ==============================

kmeans = KMeans(
    n_clusters=5,
    random_state=42,
    n_init=10
)

df["Cluster"] = kmeans.fit_predict(X_scaled)

# ==============================
# Save K-Means Model
# ==============================

os.makedirs("../models", exist_ok=True)

joblib.dump(kmeans, "../models/kmeans_model.pkl")

print("✅ K-Means model saved successfully!")

# ==============================
# Display Results
# ==============================

print("\nCluster Counts:")
print(df["Cluster"].value_counts())

print("\nFirst 10 Records:")
print(df[["Location_ID", "Cluster"]].head(10))

# ==============================
# Save Clustered Dataset
# ==============================

df.to_csv("../data/city_clustered.csv", index=False)

print("\n✅ Clustering completed successfully!")