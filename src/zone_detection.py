import pandas as pd

# Load clustered data
df = pd.read_csv("../data/city_clustered.csv")

# Map clusters to meaningful zone names
zone_mapping = {
    0: "Commercial Zone",
    1: "Recreational Zone",
    2: "Mixed Zone",
    3: "Residential Zone",
    4: "Industrial Zone"
}

# Create new column
df["Zone"] = df["Cluster"].map(zone_mapping)

# Save updated dataset
df.to_csv("../data/city_zones.csv", index=False)

print("Zone Detection Completed Successfully!\n")

print(df[["Location_ID", "Cluster", "Zone"]].head(15))