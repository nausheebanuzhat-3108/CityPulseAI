import pandas as pd

# Load dataset
df = pd.read_csv("../data/city_activity.csv")

print("="*50)
print("CITY PULSE AI - DATA PREPROCESSING")
print("="*50)

print("\nFirst 5 Records")
print(df.head())

print("\nDataset Shape")
print(df.shape)

print("\nColumn Names")
print(df.columns.tolist())

print("\nMissing Values")
print(df.isnull().sum())

print("\nDuplicate Rows")
print(df.duplicated().sum())

print("\nStatistical Summary")
print(df.describe())

print("\nData Types")
print(df.dtypes)