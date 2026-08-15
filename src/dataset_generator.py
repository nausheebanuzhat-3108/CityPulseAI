import pandas as pd
import numpy as np
import os

np.random.seed(42)

# Create data folder if it doesn't exist
os.makedirs("../data", exist_ok=True)

zones = [
    "Commercial",
    "Residential",
    "Industrial",
    "Recreational",
    "Mixed"
]

rows = []

for i in range(1000):

    zone = np.random.choice(zones)

    if zone == "Commercial":
        traffic = np.random.randint(70,101)
        pedestrians = np.random.randint(600,1001)
        vehicles = np.random.randint(400,801)
        public_transport = np.random.randint(30,60)
        speed = np.random.randint(20,40)
        commercial = np.random.randint(80,101)
        residential = np.random.randint(10,31)
        green = np.random.randint(5,21)

    elif zone == "Residential":
        traffic = np.random.randint(20,51)
        pedestrians = np.random.randint(100,301)
        vehicles = np.random.randint(100,251)
        public_transport = np.random.randint(10,30)
        speed = np.random.randint(30,55)
        commercial = np.random.randint(10,31)
        residential = np.random.randint(80,101)
        green = np.random.randint(40,71)

    elif zone == "Industrial":
        traffic = np.random.randint(60,91)
        pedestrians = np.random.randint(50,151)
        vehicles = np.random.randint(500,901)
        public_transport = np.random.randint(10,25)
        speed = np.random.randint(25,45)
        commercial = np.random.randint(20,41)
        residential = np.random.randint(10,31)
        green = np.random.randint(5,21)

    elif zone == "Recreational":
        traffic = np.random.randint(15,36)
        pedestrians = np.random.randint(300,601)
        vehicles = np.random.randint(50,201)
        public_transport = np.random.randint(5,20)
        speed = np.random.randint(15,35)
        commercial = np.random.randint(20,41)
        residential = np.random.randint(30,51)
        green = np.random.randint(80,101)

    else:
        traffic = np.random.randint(40,71)
        pedestrians = np.random.randint(250,551)
        vehicles = np.random.randint(200,501)
        public_transport = np.random.randint(20,40)
        speed = np.random.randint(20,45)
        commercial = np.random.randint(40,71)
        residential = np.random.randint(40,71)
        green = np.random.randint(30,61)

    morning = np.random.randint(40,90)
    afternoon = np.random.randint(40,90)
    evening = np.random.randint(40,100)
    night = np.random.randint(10,70)

    latitude = round(17.20 + np.random.random()*0.40,6)
    longitude = round(78.20 + np.random.random()*0.40,6)

    rows.append([
        i+1,
        latitude,
        longitude,
        traffic,
        pedestrians,
        vehicles,
        public_transport,
        speed,
        commercial,
        residential,
        green,
        morning,
        afternoon,
        evening,
        night,
        zone
    ])

columns = [
    "Location_ID",
    "Latitude",
    "Longitude",
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
    "Night_Activity",
    "True_Zone"
]

df = pd.DataFrame(rows, columns=columns)

df.to_csv("../data/city_activity.csv", index=False)

print(df.head())

print("\nDataset Created Successfully!")
print("Total Records:", len(df))