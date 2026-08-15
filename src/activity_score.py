import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load the dataset
df = pd.read_csv("../data/city_zones.csv")

# Features used to calculate activity
features = [
    "Traffic_Density",
    "Pedestrian_Count",
    "Vehicle_Count",
    "Public_Transport",
    "Commercial_Score",
    "Morning_Activity",
    "Afternoon_Activity",
    "Evening_Activity"
]

# Normalize values between 0 and 1
scaler = MinMaxScaler()

scaled = scaler.fit_transform(df[features])

scaled_df = pd.DataFrame(scaled, columns=features)

# Weighted Activity Score
df["Activity_Score"] = (
    0.20 * scaled_df["Traffic_Density"] +
    0.20 * scaled_df["Pedestrian_Count"] +
    0.15 * scaled_df["Vehicle_Count"] +
    0.10 * scaled_df["Public_Transport"] +
    0.15 * scaled_df["Commercial_Score"] +
    0.05 * scaled_df["Morning_Activity"] +
    0.05 * scaled_df["Afternoon_Activity"] +
    0.10 * scaled_df["Evening_Activity"]
) * 100

# Round to 2 decimal places
df["Activity_Score"] = df["Activity_Score"].round(2)

# Save the updated dataset
df.to_csv("../data/city_activity_scores.csv", index=False)

print("Urban Activity Scores Generated Successfully!\n")

print(df[[
    "Location_ID",
    "Zone",
    "Activity_Score"
]].head(15))