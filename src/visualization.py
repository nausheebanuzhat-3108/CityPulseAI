import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("../data/city_activity.csv")

# Histogram of Traffic Density
plt.figure(figsize=(8,5))
plt.hist(df["Traffic_Density"], bins=20)

plt.title("Traffic Density Distribution")
plt.xlabel("Traffic Density")
plt.ylabel("Number of Locations")

plt.show()