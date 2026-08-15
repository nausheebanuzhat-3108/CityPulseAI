# 🏙️ City Pulse AI

## Intelligent Urban Activity Analysis and Zone Detection System

City Pulse AI is an AI/ML-based smart-city analytics project that analyzes urban activity, detects city zones, identifies unusual patterns, and presents results through an interactive Streamlit dashboard.

The project combines **machine learning**, **anomaly detection**, **data visualization**, **geospatial mapping**, and **computer vision**.

---

## 🎯 Project Objectives

- Analyze urban activity using multiple city-related features
- Group locations into meaningful urban zones
- Calculate an activity score for each location
- Detect anomalous or unusual urban activity patterns
- Visualize city activity using charts and an interactive map
- Analyze uploaded street images using computer vision
- Provide decision-support insights and recommendations

---

## 🚀 Key Features

- Urban Activity Analysis
- Zone Detection
- Activity Score Calculation
- Anomaly Detection
- Interactive Streamlit Dashboard
- Smart City Map
- Location Explorer
- Traffic and Pedestrian Analysis
- Visual AI Image Analysis
- YOLO-Based Object Detection
- Visual Risk Estimation
- AI-Assisted Recommendations
- CSV Data Export

---

## 🤖 Machine Learning Algorithms

### 1. K-Means Clustering

K-Means clustering is used to group urban locations based on similar activity patterns.

The clustering model uses features such as:

- Traffic Density
- Pedestrian Count
- Vehicle Count
- Public Transport
- Average Speed
- Commercial Score
- Residential Score
- Green Space
- Morning Activity
- Afternoon Activity
- Evening Activity
- Night Activity

The resulting clusters are interpreted as:

- Commercial Zone
- Residential Zone
- Industrial Zone
- Recreational Zone
- Mixed Zone

---

### 2. Isolation Forest

Isolation Forest is used for anomaly detection.

It identifies urban locations whose activity patterns are significantly different from normal patterns.

Each location is classified as:

- **Normal**
- **Anomaly**

---

## 📊 Activity Score

City Pulse AI calculates an urban activity score using multiple activity-related features.

The score is represented on a scale from:

```text
0 - 100
```

Activity levels are interpreted as:

- Low
- Moderate
- High
- Very High

---

## 📸 Visual AI Module

The project includes a computer vision module that analyzes uploaded street and urban images.

The Visual AI module uses **YOLO11m** for object detection.

It can detect supported urban objects such as:

- People
- Cars
- Buses
- Trucks
- Motorcycles
- Bicycles
- Traffic Lights
- Stop Signs

The module estimates:

- Pedestrian Activity
- Traffic Activity
- Visual Activity Score
- Urban Activity Profile
- Visual Risk Level
- Visual Recommendations

For large and crowded images, the app can use an enhanced tiled-image analysis to improve detection of smaller or distant objects.

> **Note:** The pretrained YOLO model has a fixed object vocabulary. Some local vehicle types, such as auto-rickshaws, may be classified under another vehicle category or missed. Visual counts are estimates, not certified traffic measurements.

---

## 🗺️ Smart City Map

The dashboard contains an interactive Folium map that displays urban locations using latitude and longitude.

The map shows information such as:

- Zone Type
- Activity Score
- Traffic Density
- Pedestrian Count
- Vehicle Count
- Anomaly Status

---

## 📍 Location Explorer

Users can search for a Location ID and view a detailed urban report containing:

- Detected Zone
- Activity Level
- Activity Score
- Traffic Density
- Pedestrian Count
- Vehicle Count
- Public Transport
- Green Space
- Commercial Score
- Residential Score
- Peak Activity Period
- Anomaly Status
- Risk Level
- Decision-Support Recommendations

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-Learn
- Streamlit
- Plotly
- Folium
- Streamlit-Folium
- Ultralytics YOLO
- OpenCV Headless
- Pillow
- Joblib

---

## 📂 Project Structure

```text
CityPulseAI/
│
├── dashboard/
│   └── app.py
│
├── data/
│   ├── city_activity.csv
│   ├── city_activity_scores.csv
│   ├── city_clustered.csv
│   ├── city_final.csv
│   └── city_zones.csv
│
├── models/
│   ├── kmeans_model.pkl
│   └── isolation_forest.pkl
│
├── src/
│   ├── dataset_generator.py
│   ├── preprocessing.py
│   ├── visualization.py
│   ├── clustering.py
│   ├── zone_detection.py
│   ├── activity_score.py
│   └── anomaly_detection.py
│
├── requirements.txt
└── README.md
```

---

## 📊 Dataset

The project uses a **synthetic urban activity dataset** containing approximately 1000 locations.

The dataset contains features related to:

- Traffic activity
- Pedestrian movement
- Vehicle movement
- Public transport
- Commercial activity
- Residential activity
- Green space
- Time-of-day activity

Synthetic data is used to safely simulate different urban environments for this internship project.

---

## ▶️ How to Run the Project

### 1. Clone the repository

```bash
git clone https://github.com/nausheebanuzhat-3108/CityPulseAI.git
```

### 2. Open the project directory

```bash
cd CityPulseAI
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Run the Streamlit dashboard

```bash
cd dashboard
python -m streamlit run app.py
```

---

## 🌐 Deployment

The project is designed to run locally and can also be deployed using **Streamlit Community Cloud**.

The deployed dashboard provides access to:

- Urban Activity Overview
- Location Explorer
- Visual AI
- Smart City Map
- Data Visualization
- CSV Export

---

## ⚠️ Limitations

- The urban dataset is synthetic and does not represent live sensor data.
- Zone detection depends on patterns learned from the generated dataset.
- Visual AI uses a pretrained object detector and may occasionally misclassify objects.
- Visual activity scores are estimates based on a single image.
- Exact geographic location recognition from uploaded images is not included.
- Real-time traffic feeds and IoT sensor integration are not currently implemented.

---

## 🔮 Future Scope

Possible future improvements include:

- Real-time traffic data integration
- IoT sensor connectivity
- CCTV video analysis
- Live pedestrian flow monitoring
- Custom-trained Indian road object detection
- Auto-rickshaw detection
- Geographic landmark recognition
- GPS and EXIF-based location analysis
- Weather-aware activity prediction
- Deep-learning-based urban forecasting
- Emergency congestion alerts
- Smart traffic signal optimization

---

## 🎓 Project Type

**AI / Machine Learning Internship Project**

---

## 👨‍💻 Project Summary

City Pulse AI demonstrates the practical use of:

- Unsupervised Machine Learning
- Anomaly Detection
- Data Preprocessing
- Data Visualization
- Computer Vision
- Geospatial Analytics
- Interactive AI Dashboards
- Smart City Decision Support
