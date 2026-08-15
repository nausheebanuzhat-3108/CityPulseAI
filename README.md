# 🏙️ City Pulse AI

## Intelligent Urban Activity Analysis and Zone Detection System

### 📌 Project Overview

City Pulse AI is an AI and Machine Learning based smart-city analytics system designed to analyze urban activity, identify city zones, calculate activity intensity, detect unusual patterns, and visualize results through an interactive dashboard.

The project combines traditional machine learning with computer vision to provide both structured urban-data analysis and image-based visual activity analysis.

---

## 🎯 Project Objectives

- Analyze urban activity using multiple city-related features
- Group locations into meaningful urban zones
- Calculate an activity score for each location
- Detect unusual or anomalous urban activity patterns
- Visualize spatial activity using an interactive smart-city map
- Analyze street images using computer vision
- Provide AI-assisted urban insights and recommendations

---

## 🚀 Key Features

- Urban Activity Analysis
- AI-Based Zone Detection
- Activity Score Calculation
- Anomaly Detection
- Interactive Streamlit Dashboard
- Smart City Map using Folium
- Location-Based Urban Reports
- Traffic and Pedestrian Analysis
- Visual AI Image Analysis
- YOLO-Based Object Detection
- Urban Risk Estimation
- AI-Assisted Recommendations
- CSV Data Export

---

## 🤖 AI / Machine Learning Techniques

### K-Means Clustering

K-Means clustering is used to group urban locations based on similar activity patterns.

The model analyzes features such as:

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

The clusters are interpreted into urban zones such as:

- Commercial Zone
- Residential Zone
- Industrial Zone
- Recreational Zone
- Mixed Zone

---

### Isolation Forest

Isolation Forest is used for anomaly detection.

It identifies locations whose urban activity patterns differ significantly from normal city behavior.

Each location is classified as:

- Normal
- Anomaly

---

## 📊 Activity Score

City Pulse AI calculates an activity score for each location using multiple urban indicators.

The score is represented on a scale from:

```text
0 - 100