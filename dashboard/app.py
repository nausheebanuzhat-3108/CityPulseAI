import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import folium
from streamlit_folium import st_folium
from pathlib import Path
from datetime import datetime
from PIL import Image
import numpy as np
from ultralytics import YOLO

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="City Pulse AI",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================================
# PATHS + DATA
# =========================================================

APP_DIR = Path(__file__).resolve().parent
DATA_PATH = APP_DIR.parent / "data" / "city_final.csv"

@st.cache_data
def load_data():
    data = pd.read_csv(DATA_PATH)
    return data

try:
    df = load_data()
except FileNotFoundError:
    st.error(
        f"Could not find the dataset at: {DATA_PATH}\n\n"
        "Keep this file inside CityPulseAI/dashboard and make sure "
        "CityPulseAI/data/city_final.csv exists."
    )
    st.stop()

# =========================================================
# DERIVED COLUMNS
# =========================================================

def get_activity_level(score):
    if score < 30:
        return "Low"
    if score < 60:
        return "Moderate"
    if score < 80:
        return "High"
    return "Very High"

df["Activity_Level"] = df["Activity_Score"].apply(get_activity_level)

ZONE_COLORS = {
    "Commercial Zone": "#FF5D73",
    "Residential Zone": "#2ED6A1",
    "Industrial Zone": "#FF9F43",
    "Recreational Zone": "#4DA3FF",
    "Mixed Zone": "#A679FF",
}

LEVEL_COLORS = {
    "Low": "#2ED6A1",
    "Moderate": "#F8D66D",
    "High": "#FF9F43",
    "Very High": "#FF5D73",
}


# =========================================================
# VISUAL AI MODEL + HELPERS
# =========================================================

SUPPORTED_URBAN_CLASSES = {
    "person",
    "car",
    "bus",
    "truck",
    "motorcycle",
    "bicycle",
    "traffic light",
    "stop sign",
}


@st.cache_resource
def load_visual_model():
    """
    Use YOLO11m for higher-quality crowded-scene detection.
    If the medium model cannot be loaded in a constrained deployment,
    fall back to YOLO11s automatically.
    """
    try:
        return YOLO("yolo11m.pt"), "YOLO11m"
    except Exception:
        return YOLO("yolo11s.pt"), "YOLO11s"


def extract_detection_summary(result):
    counts = {}
    confidences = []
    supported_confidences = []

    if result.boxes is not None and len(result.boxes) > 0:
        class_ids = result.boxes.cls.cpu().numpy().astype(int).tolist()
        confidences = result.boxes.conf.cpu().numpy().tolist()

        for class_id, confidence in zip(class_ids, confidences):
            label = result.names[class_id]
            counts[label] = counts.get(label, 0) + 1
            if label in SUPPORTED_URBAN_CLASSES:
                supported_confidences.append(float(confidence))

    return counts, confidences, supported_confidences


def run_visual_detection(image, model, use_crowded_scene_pass=True):
    """
    Primary inference uses a larger input size and a lower confidence threshold
    than the previous prototype so small / partially occluded urban objects are
    less likely to be missed.

    For large crowded images with relatively few initial detections, a second
    non-overlapping 2x2 tiled pass is used for analysis counts. This often helps
    with distant people and vehicles without double-counting overlapping tiles.
    """
    image_array = np.array(image)

    full_result = model.predict(
        source=image_array,
        conf=0.15,
        iou=0.45,
        imgsz=1280,
        max_det=500,
        verbose=False,
    )[0]

    full_counts, full_confidences, full_supported_conf = extract_detection_summary(full_result)
    analysis_counts = dict(full_counts)
    analysis_confidences = list(full_supported_conf)
    used_tiled_pass = False

    supported_full_total = sum(
        full_counts.get(label, 0)
        for label in SUPPORTED_URBAN_CLASSES
    )

    width, height = image.size

    if (
        use_crowded_scene_pass
        and width >= 1000
        and height >= 650
        and supported_full_total < 22
    ):
        used_tiled_pass = True
        x_mid = width // 2
        y_mid = height // 2

        crop_boxes = [
            (0, 0, x_mid, y_mid),
            (x_mid, 0, width, y_mid),
            (0, y_mid, x_mid, height),
            (x_mid, y_mid, width, height),
        ]

        tiled_counts = {}
        tiled_confidences = []

        for crop_box in crop_boxes:
            crop = image.crop(crop_box)
            tile_result = model.predict(
                source=np.array(crop),
                conf=0.12,
                iou=0.45,
                imgsz=960,
                max_det=300,
                verbose=False,
            )[0]

            tile_counts, _, tile_supported_conf = extract_detection_summary(tile_result)
            tiled_confidences.extend(tile_supported_conf)

            for label, count in tile_counts.items():
                tiled_counts[label] = tiled_counts.get(label, 0) + count

        # Use whichever pass found more objects for each class.
        # The tiles do not overlap, so their summed count does not create
        # duplicate detections from overlapping crops.
        all_labels = set(analysis_counts) | set(tiled_counts)
        analysis_counts = {
            label: max(full_counts.get(label, 0), tiled_counts.get(label, 0))
            for label in all_labels
        }

        if tiled_confidences:
            analysis_confidences.extend(tiled_confidences)

    return full_result, analysis_counts, full_confidences, analysis_confidences, used_tiled_pass


def visual_activity_level(score):
    if score < 30:
        return "Low"
    if score < 55:
        return "Moderate"
    if score < 78:
        return "High"
    return "Very High"


def visual_metrics_from_counts(counts):
    people = counts.get("person", 0)
    cars = counts.get("car", 0)
    buses = counts.get("bus", 0)
    trucks = counts.get("truck", 0)
    motorcycles = counts.get("motorcycle", 0)
    bicycles = counts.get("bicycle", 0)
    traffic_lights = counts.get("traffic light", 0)

    weighted_vehicle_units = (
        cars
        + 1.7 * buses
        + 1.8 * trucks
        + 0.8 * motorcycles
        + 0.5 * bicycles
    )

    vehicle_total = cars + buses + trucks + motorcycles + bicycles
    supported_total = people + vehicle_total + traffic_lights

    # Explainable visual activity estimate.
    # The score intentionally gives stronger weight to person/vehicle density
    # than the earlier prototype, which underestimated crowded street scenes.
    pedestrian_component = min(45.0, people * 3.0)
    traffic_component = min(42.0, weighted_vehicle_units * 4.6)
    crowd_bonus = min(8.0, max(0, supported_total - 8) * 0.8)
    transit_bonus = min(5.0, buses * 2.5 + traffic_lights * 0.8)

    visual_score = min(
        100.0,
        pedestrian_component + traffic_component + crowd_bonus + transit_bonus,
    )

    if weighted_vehicle_units >= 15:
        traffic_level = "Very High"
    elif weighted_vehicle_units >= 8:
        traffic_level = "High"
    elif weighted_vehicle_units >= 3.5:
        traffic_level = "Moderate"
    else:
        traffic_level = "Low"

    if people >= 20:
        pedestrian_level = "Very High"
    elif people >= 10:
        pedestrian_level = "High"
    elif people >= 4:
        pedestrian_level = "Moderate"
    else:
        pedestrian_level = "Low"

    return {
        "people": people,
        "cars": cars,
        "buses": buses,
        "trucks": trucks,
        "motorcycles": motorcycles,
        "bicycles": bicycles,
        "traffic_lights": traffic_lights,
        "vehicle_total": vehicle_total,
        "weighted_vehicle_units": weighted_vehicle_units,
        "visual_score": visual_score,
        "traffic_level": traffic_level,
        "pedestrian_level": pedestrian_level,
    }


def estimate_visual_zone(counts):
    """
    Activity-based urban profile only — not geographic place recognition.
    """
    metrics = visual_metrics_from_counts(counts)
    people = metrics["people"]
    vehicles = metrics["weighted_vehicle_units"]
    trucks = metrics["trucks"]
    buses = metrics["buses"]

    if trucks >= 3 and trucks >= max(2, people // 4):
        return "Industrial Zone"

    if people >= 10 and vehicles >= 4:
        return "Commercial Zone"

    if people >= 7 and vehicles <= 3.5:
        return "Recreational / Public Space"

    if people <= 5 and vehicles <= 4:
        return "Residential / Low-Activity Zone"

    if buses >= 2 and vehicles >= 5:
        return "Commercial / Transit Hub"

    return "Mixed Urban Zone"


def visual_recommendations(
    activity_score,
    traffic_level,
    pedestrian_level,
    people,
    vehicles,
    buses,
    trucks,
):
    recs = []

    if traffic_level in {"Very High", "High"}:
        recs.append("High vehicle concentration detected: optimize signal timing and congestion flow.")
    elif traffic_level == "Moderate":
        recs.append("Moderate traffic activity: monitor peak periods and junction performance.")
    else:
        recs.append("Vehicle activity appears manageable in this visual sample.")

    if pedestrian_level in {"Very High", "High"}:
        recs.append("Strong pedestrian presence: prioritize safe crossings, sidewalks and crowd movement.")
    elif pedestrian_level == "Moderate":
        recs.append("Moderate pedestrian activity: maintain clear and safe crossing access.")

    if buses == 0 and activity_score >= 70:
        recs.append("High visual activity with no bus detected: review public-transport availability.")

    if trucks >= 3:
        recs.append("Multiple heavy vehicles detected: consider freight-route and loading-zone monitoring.")

    if vehicles >= 10:
        recs.append("Parking, curb-space and lane-use management may help reduce local congestion.")

    return recs[:4]

# =========================================================
# PROFESSIONAL UI
# =========================================================

st.markdown(
    """
<style>
:root {
    --bg: #08111f;
    --panel: rgba(18, 32, 52, 0.82);
    --panel-2: rgba(22, 39, 63, 0.92);
    --border: rgba(150, 180, 220, 0.16);
    --text: #f4f8ff;
    --muted: #9fb0c7;
    --cyan: #47d7ff;
    --blue: #5b8cff;
    --green: #2ed6a1;
    --red: #ff5d73;
    --orange: #ff9f43;
}

.stApp {
    background:
        radial-gradient(circle at 15% 0%, rgba(71,215,255,.10), transparent 28%),
        radial-gradient(circle at 90% 8%, rgba(91,140,255,.12), transparent 24%),
        linear-gradient(180deg, #07101d 0%, #0a1423 48%, #08111f 100%);
    color: var(--text);
}

.block-container {
    max-width: 1500px;
    padding-top: 1.4rem;
    padding-bottom: 3rem;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0c1727 0%, #0a1320 100%);
    border-right: 1px solid var(--border);
}

.hero {
    position: relative;
    overflow: hidden;
    padding: 28px 30px;
    margin-bottom: 20px;
    border: 1px solid var(--border);
    border-radius: 24px;
    background:
        linear-gradient(120deg, rgba(35,64,102,.78), rgba(12,27,47,.86)),
        radial-gradient(circle at 80% 0%, rgba(71,215,255,.18), transparent 34%);
    box-shadow: 0 18px 50px rgba(0,0,0,.25);
}

.hero-title {
    font-size: clamp(2rem, 5vw, 3.25rem);
    font-weight: 850;
    letter-spacing: -0.04em;
    margin: 0;
    line-height: 1.05;
}

.hero-subtitle {
    color: #b7c6da;
    font-size: 1.02rem;
    margin-top: 10px;
    margin-bottom: 0;
}

.hero-badge {
    display: inline-block;
    margin-bottom: 12px;
    padding: 6px 10px;
    border-radius: 999px;
    background: rgba(71,215,255,.10);
    border: 1px solid rgba(71,215,255,.28);
    color: #8fe8ff;
    font-size: .78rem;
    font-weight: 700;
    letter-spacing: .08em;
    text-transform: uppercase;
}

.section-kicker {
    color: #8fe8ff;
    text-transform: uppercase;
    letter-spacing: .10em;
    font-size: .74rem;
    font-weight: 800;
    margin-bottom: 4px;
}

.section-title {
    font-size: 1.55rem;
    font-weight: 780;
    margin: 4px 0 14px 0;
}

.kpi {
    min-height: 142px;
    padding: 18px 18px 16px 18px;
    border: 1px solid var(--border);
    border-radius: 18px;
    background: linear-gradient(145deg, rgba(21,38,62,.94), rgba(13,25,43,.94));
    box-shadow: 0 10px 28px rgba(0,0,0,.18);
    transition: transform .18s ease, border-color .18s ease;
}

.kpi:hover {
    transform: translateY(-3px);
    border-color: rgba(71,215,255,.45);
}

.kpi-icon {
    font-size: 1.35rem;
}

.kpi-label {
    color: var(--muted);
    font-size: .78rem;
    font-weight: 750;
    letter-spacing: .07em;
    margin-top: 9px;
}

.kpi-value {
    font-size: 1.95rem;
    font-weight: 850;
    margin-top: 3px;
    letter-spacing: -0.03em;
}

.kpi-note {
    color: #8194ad;
    font-size: .75rem;
    margin-top: 4px;
}

.report-card {
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 20px;
    background: linear-gradient(145deg, rgba(20,37,60,.94), rgba(11,23,39,.96));
    min-height: 100%;
}

.report-label {
    color: var(--muted);
    font-size: .76rem;
    font-weight: 700;
    letter-spacing: .05em;
    text-transform: uppercase;
}

.report-value {
    font-size: 1.15rem;
    font-weight: 760;
    margin-top: 3px;
    margin-bottom: 14px;
}

.status-pill {
    display: inline-block;
    padding: 7px 11px;
    border-radius: 999px;
    font-size: .78rem;
    font-weight: 800;
}

.status-normal {
    color: #7ff4c9;
    background: rgba(46,214,161,.12);
    border: 1px solid rgba(46,214,161,.30);
}

.status-anomaly {
    color: #ff9aaa;
    background: rgba(255,93,115,.12);
    border: 1px solid rgba(255,93,115,.32);
}

.risk-low {
    color: #7ff4c9;
    background: rgba(46,214,161,.12);
    border: 1px solid rgba(46,214,161,.30);
}

.risk-medium {
    color: #ffd88e;
    background: rgba(255,159,67,.12);
    border: 1px solid rgba(255,159,67,.30);
}

.risk-high {
    color: #ff9aaa;
    background: rgba(255,93,115,.12);
    border: 1px solid rgba(255,93,115,.32);
}

.insight-box {
    padding: 16px 18px;
    border: 1px solid var(--border);
    border-radius: 16px;
    background: rgba(15,29,48,.78);
    margin-bottom: 10px;
}

.small-muted {
    color: var(--muted);
    font-size: .86rem;
}

div[data-testid="stMetric"] {
    background: rgba(17,31,50,.72);
    border: 1px solid var(--border);
    padding: 14px 16px;
    border-radius: 16px;
}

div[data-testid="stDataFrame"] {
    border: 1px solid var(--border);
    border-radius: 16px;
    overflow: hidden;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background: rgba(10,20,34,.58);
    padding: 7px;
    border: 1px solid var(--border);
    border-radius: 14px;
}

.stTabs [data-baseweb="tab"] {
    border-radius: 10px;
    padding: 8px 15px;
}

.stTabs [aria-selected="true"] {
    background: rgba(71,215,255,.10);
}

hr {
    border-color: rgba(150,180,220,.13);
}

.footer {
    text-align: center;
    color: #7890ab;
    padding: 22px 0 8px 0;
    font-size: .84rem;
}
</style>
""",
    unsafe_allow_html=True,
)

# =========================================================
# HELPERS
# =========================================================

def metric_card(icon, label, value, note=""):
    st.markdown(
        f"""
        <div class="kpi">
            <div class="kpi-icon">{icon}</div>
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def polish_figure(fig, height=360):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#DCE7F7"),
        margin=dict(l=28, r=22, t=60, b=36),
        height=height,
        legend_title_text="",
    )
    return fig

def peak_period(row):
    period_columns = {
        "Morning": "Morning_Activity",
        "Afternoon": "Afternoon_Activity",
        "Evening": "Evening_Activity",
        "Night": "Night_Activity",
    }
    available = {name: row[col] for name, col in period_columns.items() if col in row.index}
    if not available:
        return "Not available"
    return max(available, key=available.get)

def risk_level(row):
    if row.get("Status", "Normal") == "Anomaly":
        return "High"
    traffic = float(row.get("Traffic_Density", 0))
    activity = float(row.get("Activity_Score", 0))
    if traffic >= 80 or activity >= 80:
        return "High"
    if traffic >= 55 or activity >= 60:
        return "Medium"
    return "Low"

def build_recommendations(row):
    recs = []
    activity = float(row.get("Activity_Score", 0))
    traffic = float(row.get("Traffic_Density", 0))
    public_transport = float(row.get("Public_Transport", 0))
    green_space = float(row.get("Green_Space", 0))
    status = row.get("Status", "Normal")
    zone = row.get("Zone", "")

    if status == "Anomaly":
        recs.append("Investigate this location because the anomaly model flagged its activity pattern as unusual.")

    if traffic >= 75:
        recs.append("Prioritize traffic-flow optimization and adaptive signal timing.")
    elif traffic >= 55:
        recs.append("Monitor congestion during the location's busiest activity period.")
    else:
        recs.append("Traffic conditions are relatively manageable; maintain routine monitoring.")

    if activity >= 80:
        recs.append("Prepare for high service demand and increase operational monitoring.")
    elif activity >= 60:
        recs.append("Maintain capacity for moderate-to-high urban demand.")
    else:
        recs.append("Current activity demand is comparatively low.")

    if public_transport < 20 and activity >= 60:
        recs.append("Consider improving public-transport availability for this activity level.")

    if green_space < 20 and zone in {"Commercial Zone", "Industrial Zone", "Mixed Zone"}:
        recs.append("Explore additional green buffers or public-space improvements where feasible.")

    if zone == "Commercial Zone":
        recs.append("Review parking, pedestrian movement, and last-mile access around commercial areas.")
    elif zone == "Residential Zone":
        recs.append("Protect residential mobility, pedestrian safety, and local access.")
    elif zone == "Industrial Zone":
        recs.append("Monitor freight movement and separate heavy-vehicle flows from pedestrian routes.")
    elif zone == "Recreational Zone":
        recs.append("Preserve pedestrian comfort, green-space quality, and safe access.")
    elif zone == "Mixed Zone":
        recs.append("Balance residential, commercial, pedestrian, and vehicle demand.")

    # Keep the panel concise.
    return recs[:5]

def view_health(data):
    if data.empty:
        return "No data", "#9FB0C7", "No locations match the current view."
    anomaly_rate = (data["Status"].eq("Anomaly").mean() * 100) if "Status" in data else 0
    avg_traffic = data["Traffic_Density"].mean()
    if anomaly_rate >= 8 or avg_traffic >= 78:
        return "Critical", "#FF5D73", "High anomaly or traffic pressure in the current view."
    if anomaly_rate >= 3 or avg_traffic >= 58:
        return "Watch", "#FFB35C", "Conditions are manageable, but selected areas should be monitored."
    return "Stable", "#2ED6A1", "The current view shows generally stable urban conditions."

# =========================================================
# HEADER
# =========================================================

now = datetime.now()
st.markdown(
    f"""
    <div class="hero">
        <div class="hero-badge">AI-powered smart city analytics</div>
        <div class="hero-title">🏙️ City Pulse AI</div>
        <div class="hero-subtitle">
            Intelligent Urban Activity Analysis & Zone Detection System
            &nbsp;•&nbsp; Updated {now.strftime("%d %b %Y, %I:%M %p")}
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# SIDEBAR CONTROLS
# =========================================================

st.sidebar.title("🎛️ Control Center")
st.sidebar.caption("Search a location or filter the city-wide view.")

search_location = st.sidebar.text_input(
    "📍 Location ID",
    placeholder="Example: 100",
    help="Searching for a Location ID focuses the entire dashboard on that location.",
)

zones = ["All Zones"] + sorted(df["Zone"].dropna().astype(str).unique().tolist())
selected_zone = st.sidebar.selectbox("🏙️ Zone", zones)

min_score, max_score = st.sidebar.slider(
    "⭐ Activity Score",
    min_value=0,
    max_value=100,
    value=(0, 100),
)

status_option = st.sidebar.selectbox(
    "🚨 Status",
    ["All", "Normal", "Anomaly"],
)

activity_options = ["All", "Low", "Moderate", "High", "Very High"]
selected_activity = st.sidebar.selectbox(
    "📊 Activity Level",
    activity_options,
)

st.sidebar.divider()
st.sidebar.markdown("#### Model Stack")
st.sidebar.caption("K-Means • Isolation Forest • Rule-based activity interpretation")
st.sidebar.caption("Tip: Location search overrides the other filters for a clear single-location report.")

# =========================================================
# SEARCH + FILTER LOGIC
# =========================================================

searched_row = None
search_error = None

if search_location.strip():
    try:
        location_id = int(search_location.strip())
        match = df[df["Location_ID"] == location_id]
        if match.empty:
            search_error = f"Location ID {location_id} was not found."
            filtered_df = df.iloc[0:0].copy()
        else:
            searched_row = match.iloc[0]
            filtered_df = match.copy()
    except ValueError:
        search_error = "Enter a numeric Location ID, for example 100."
        filtered_df = df.iloc[0:0].copy()
else:
    filtered_df = df.copy()

    if selected_zone != "All Zones":
        filtered_df = filtered_df[filtered_df["Zone"] == selected_zone]

    filtered_df = filtered_df[
        (filtered_df["Activity_Score"] >= min_score)
        & (filtered_df["Activity_Score"] <= max_score)
    ]

    if status_option != "All":
        filtered_df = filtered_df[filtered_df["Status"] == status_option]

    if selected_activity != "All":
        filtered_df = filtered_df[filtered_df["Activity_Level"] == selected_activity]

if search_error:
    st.error(search_error)
elif searched_row is not None:
    st.success(
        f"Location {int(searched_row['Location_ID'])} found — "
        "the dashboard is now focused on this location."
    )

# =========================================================
# KPI STRIP
# =========================================================

total_locations = len(filtered_df)
avg_activity = filtered_df["Activity_Score"].mean() if total_locations else 0
total_anomalies = int(filtered_df["Status"].eq("Anomaly").sum()) if total_locations else 0
avg_traffic = filtered_df["Traffic_Density"].mean() if total_locations else 0
active_zones = filtered_df["Zone"].nunique() if total_locations else 0
health, health_color, health_note = view_health(filtered_df)

k1, k2, k3, k4, k5 = st.columns(5)
with k1:
    metric_card("📍", "LOCATIONS", f"{total_locations:,}", "Current view")
with k2:
    metric_card("⚡", "AVG ACTIVITY", f"{avg_activity:.1f}/100", "Urban activity index")
with k3:
    metric_card("🚦", "AVG TRAFFIC", f"{avg_traffic:.1f}/100", "Traffic density")
with k4:
    metric_card("🚨", "ANOMALIES", f"{total_anomalies:,}", "Isolation Forest")
with k5:
    metric_card("🧭", "ACTIVE ZONES", f"{active_zones}", health)

st.markdown("<br>", unsafe_allow_html=True)

# =========================================================
# TABS
# =========================================================

overview_tab, location_tab, visual_tab, map_tab, data_tab = st.tabs(
    ["📊 Overview", "📍 Location Explorer", "📸 Visual AI", "🗺️ Smart Map", "📋 Data & Export"]
)

# =========================================================
# OVERVIEW TAB
# =========================================================

with overview_tab:
    st.markdown('<div class="section-kicker">Current view</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">City Intelligence Overview</div>', unsafe_allow_html=True)

    if filtered_df.empty:
        st.warning("No data matches the current selection. Adjust the filters or clear the location search.")
    else:
        health_col, insight_col = st.columns([0.85, 1.65])

        with health_col:
            st.markdown(
                f"""
                <div class="report-card">
                    <div class="report-label">Urban health status</div>
                    <div style="font-size:2rem;font-weight:850;color:{health_color};margin:7px 0 8px 0;">
                        {health}
                    </div>
                    <div class="small-muted">{health_note}</div>
                    <hr>
                    <div class="report-label">Average activity</div>
                    <div class="report-value">{avg_activity:.1f}/100</div>
                    <div class="report-label">Average traffic density</div>
                    <div class="report-value">{avg_traffic:.1f}/100</div>
                    <div class="report-label">Anomaly share</div>
                    <div class="report-value">
                        {(total_anomalies / max(total_locations, 1) * 100):.1f}%
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with insight_col:
            highest = filtered_df.loc[filtered_df["Activity_Score"].idxmax()]
            dominant_zone = filtered_df["Zone"].value_counts().idxmax()
            avg_ped = filtered_df["Pedestrian_Count"].mean()
            peak_counts = filtered_df["Activity_Level"].value_counts()

            st.markdown(
                f"""
                <div class="report-card">
                    <div class="report-label">AI-assisted summary</div>
                    <div style="font-size:1.15rem;font-weight:760;margin:8px 0 14px 0;">
                        The current view contains {total_locations:,} location(s) across
                        {active_zones} zone type(s).
                    </div>
                    <div class="insight-box">
                        🏆 <b>Highest activity:</b> Location {int(highest["Location_ID"])}
                        at {highest["Activity_Score"]:.1f}/100
                    </div>
                    <div class="insight-box">
                        🏙️ <b>Dominant zone:</b> {dominant_zone}
                    </div>
                    <div class="insight-box">
                        🚶 <b>Average pedestrian count:</b> {avg_ped:.0f}
                    </div>
                    <div class="insight-box">
                        🔥 <b>High + Very High activity:</b>
                        {int(peak_counts.get("High", 0) + peak_counts.get("Very High", 0))} location(s)
                    </div>
                    <div class="insight-box">
                        🚨 <b>Model alerts:</b> {total_anomalies} anomalous location(s)
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)

        c1, c2 = st.columns(2)

        with c1:
            zone_counts = filtered_df["Zone"].value_counts().reset_index()
            zone_counts.columns = ["Zone", "Count"]
            fig_zone = px.pie(
                zone_counts,
                names="Zone",
                values="Count",
                hole=0.58,
                title="Urban Zone Distribution",
                color="Zone",
                color_discrete_map=ZONE_COLORS,
            )
            fig_zone.update_traces(textposition="inside", textinfo="percent+label")
            polish_figure(fig_zone)
            st.plotly_chart(fig_zone, use_container_width=True, config={"displayModeBar": False})

        with c2:
            by_zone = (
                filtered_df.groupby("Zone", as_index=False)["Activity_Score"]
                .mean()
                .sort_values("Activity_Score", ascending=False)
            )
            fig_bar = px.bar(
                by_zone,
                x="Zone",
                y="Activity_Score",
                color="Zone",
                color_discrete_map=ZONE_COLORS,
                title="Average Activity Score by Zone",
            )
            fig_bar.update_layout(showlegend=False)
            fig_bar.update_yaxes(range=[0, 100])
            polish_figure(fig_bar)
            st.plotly_chart(fig_bar, use_container_width=True, config={"displayModeBar": False})

        c3, c4 = st.columns(2)

        with c3:
            level_counts = (
                filtered_df["Activity_Level"]
                .value_counts()
                .reindex(["Low", "Moderate", "High", "Very High"], fill_value=0)
                .reset_index()
            )
            level_counts.columns = ["Activity Level", "Locations"]
            fig_levels = px.bar(
                level_counts,
                x="Activity Level",
                y="Locations",
                color="Activity Level",
                color_discrete_map=LEVEL_COLORS,
                title="Activity Level Distribution",
            )
            fig_levels.update_layout(showlegend=False)
            polish_figure(fig_levels)
            st.plotly_chart(fig_levels, use_container_width=True, config={"displayModeBar": False})

        with c4:
            fig_scatter = px.scatter(
                filtered_df,
                x="Traffic_Density",
                y="Activity_Score",
                color="Zone",
                color_discrete_map=ZONE_COLORS,
                hover_data=["Location_ID", "Status", "Activity_Level"],
                title="Traffic vs Activity",
            )
            fig_scatter.update_xaxes(range=[0, 100])
            fig_scatter.update_yaxes(range=[0, 100])
            polish_figure(fig_scatter)
            st.plotly_chart(fig_scatter, use_container_width=True, config={"displayModeBar": False})

        st.markdown('<div class="section-title">🏆 Top Active Locations</div>', unsafe_allow_html=True)
        top10 = filtered_df.nlargest(10, "Activity_Score").copy()
        top10["Location"] = "Location " + top10["Location_ID"].astype(int).astype(str)
        fig_top = px.bar(
            top10.sort_values("Activity_Score"),
            x="Activity_Score",
            y="Location",
            orientation="h",
            color="Zone",
            color_discrete_map=ZONE_COLORS,
            hover_data=["Status", "Traffic_Density", "Pedestrian_Count"],
            title="Top 10 Locations by Activity Score",
        )
        fig_top.update_xaxes(range=[0, 100])
        polish_figure(fig_top, height=430)
        st.plotly_chart(fig_top, use_container_width=True, config={"displayModeBar": False})

# =========================================================
# LOCATION EXPLORER TAB
# =========================================================

with location_tab:
    st.markdown('<div class="section-kicker">Search & explain</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Smart Location Report</div>', unsafe_allow_html=True)

    if searched_row is None:
        st.info(
            "Enter a Location ID in the sidebar (for example, 100). "
            "This tab will show the location's model result, urban profile, "
            "risk level, peak period, and recommendations."
        )

        sample_ids = df["Location_ID"].head(5).astype(int).tolist()
        st.caption("Example valid IDs: " + ", ".join(map(str, sample_ids)))
    else:
        row = searched_row
        risk = risk_level(row)
        risk_class = {"Low": "risk-low", "Medium": "risk-medium", "High": "risk-high"}[risk]
        status_class = "status-anomaly" if row["Status"] == "Anomaly" else "status-normal"
        peak = peak_period(row)
        recommendations = build_recommendations(row)

        report_left, report_right = st.columns([1.05, 1.25])

        with report_left:
            st.markdown(
                f"""
                <div class="report-card">
                    <div class="report-label">Location ID</div>
                    <div style="font-size:2.25rem;font-weight:850;margin-bottom:10px;">
                        {int(row["Location_ID"])}
                    </div>
                    <span class="status-pill {status_class}">{row["Status"]}</span>
                    &nbsp;
                    <span class="status-pill {risk_class}">{risk} Risk</span>
                    <hr>
                    <div class="report-label">Detected zone</div>
                    <div class="report-value">{row["Zone"]}</div>
                    <div class="report-label">Activity level</div>
                    <div class="report-value">{row["Activity_Level"]}</div>
                    <div class="report-label">Peak activity period</div>
                    <div class="report-value">{peak}</div>
                    <div class="report-label">Coordinates</div>
                    <div class="report-value">
                        {row["Latitude"]:.5f}, {row["Longitude"]:.5f}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with report_right:
            gauge1, gauge2 = st.columns(2)

            with gauge1:
                fig_activity_gauge = go.Figure(
                    go.Indicator(
                        mode="gauge+number",
                        value=float(row["Activity_Score"]),
                        number={"suffix": "/100"},
                        title={"text": "Activity Score"},
                        gauge={
                            "axis": {"range": [0, 100]},
                            "bar": {"color": "#47D7FF"},
                            "bgcolor": "rgba(255,255,255,.04)",
                            "borderwidth": 0,
                            "steps": [
                                {"range": [0, 30], "color": "rgba(46,214,161,.16)"},
                                {"range": [30, 60], "color": "rgba(248,214,109,.13)"},
                                {"range": [60, 80], "color": "rgba(255,159,67,.13)"},
                                {"range": [80, 100], "color": "rgba(255,93,115,.14)"},
                            ],
                        },
                    )
                )
                polish_figure(fig_activity_gauge, height=280)
                st.plotly_chart(
                    fig_activity_gauge,
                    use_container_width=True,
                    config={"displayModeBar": False},
                )

            with gauge2:
                fig_traffic_gauge = go.Figure(
                    go.Indicator(
                        mode="gauge+number",
                        value=float(row["Traffic_Density"]),
                        number={"suffix": "/100"},
                        title={"text": "Traffic Density"},
                        gauge={
                            "axis": {"range": [0, 100]},
                            "bar": {"color": "#FF9F43"},
                            "bgcolor": "rgba(255,255,255,.04)",
                            "borderwidth": 0,
                            "steps": [
                                {"range": [0, 40], "color": "rgba(46,214,161,.14)"},
                                {"range": [40, 70], "color": "rgba(248,214,109,.12)"},
                                {"range": [70, 100], "color": "rgba(255,93,115,.13)"},
                            ],
                        },
                    )
                )
                polish_figure(fig_traffic_gauge, height=280)
                st.plotly_chart(
                    fig_traffic_gauge,
                    use_container_width=True,
                    config={"displayModeBar": False},
                )

        st.markdown('<div class="section-title">Urban Profile</div>', unsafe_allow_html=True)

        p1, p2, p3, p4 = st.columns(4)
        with p1:
            st.metric("🚶 Pedestrians", f"{int(row['Pedestrian_Count']):,}")
        with p2:
            st.metric("🚗 Vehicles", f"{int(row['Vehicle_Count']):,}")
        with p3:
            st.metric("🚌 Public Transport", f"{int(row['Public_Transport']):,}")
        with p4:
            st.metric("🌳 Green Space", f"{float(row['Green_Space']):.0f}/100")

        p5, p6, p7, p8 = st.columns(4)
        with p5:
            st.metric("🏪 Commercial", f"{float(row['Commercial_Score']):.0f}/100")
        with p6:
            st.metric("🏠 Residential", f"{float(row['Residential_Score']):.0f}/100")
        with p7:
            st.metric("🚘 Avg Speed", f"{float(row['Average_Speed']):.0f} km/h")
        with p8:
            cluster_value = int(row["Cluster"]) if "Cluster" in row.index else "—"
            st.metric("🧩 Cluster", cluster_value)

        st.markdown('<div class="section-title">🤖 Decision-Support Recommendations</div>', unsafe_allow_html=True)
        st.caption(
            "These recommendations are rule-based interpretations of the machine-learning results "
            "and the location's measured features."
        )
        for rec in recommendations:
            st.markdown(f'<div class="insight-box">✓ {rec}</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-title">Time-of-Day Activity</div>', unsafe_allow_html=True)
        period_df = pd.DataFrame(
            {
                "Period": ["Morning", "Afternoon", "Evening", "Night"],
                "Activity": [
                    row["Morning_Activity"],
                    row["Afternoon_Activity"],
                    row["Evening_Activity"],
                    row["Night_Activity"],
                ],
            }
        )
        fig_period = px.line(
            period_df,
            x="Period",
            y="Activity",
            markers=True,
            title=f"Location {int(row['Location_ID'])} Activity Pattern",
        )
        fig_period.update_yaxes(range=[0, 100])
        polish_figure(fig_period, height=330)
        st.plotly_chart(fig_period, use_container_width=True, config={"displayModeBar": False})

# =========================================================
# VISUAL AI TAB
# =========================================================

with visual_tab:
    st.markdown('<div class="section-kicker">Computer vision</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Visual Urban Intelligence</div>', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="report-card">
            <div style="font-size:1.08rem;font-weight:760;margin-bottom:8px;">
                Upload a city or street image for enhanced visual activity analysis
            </div>
            <div class="small-muted">
                City Pulse AI uses a stronger YOLO detector, higher-resolution inference and an
                optional crowded-scene pass to improve detection of small and partially hidden
                people and vehicles. The urban profile and activity score are visual estimates;
                exact place/location recognition is intentionally not included in this version.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    control_col, info_col = st.columns([1, 2])

    with control_col:
        use_crowded_scene_pass = st.checkbox(
            "Enhanced crowded-scene detection",
            value=True,
            help=(
                "For large, crowded images the app may run an additional 2×2 tiled pass "
                "to improve small-object counts. It is slower but usually more accurate."
            ),
        )

    with info_col:
        st.caption(
            "Accuracy mode: YOLO11m · 1280 px full-frame inference · lower confidence threshold "
            "for small/occluded urban objects."
        )

    uploaded_image = st.file_uploader(
        "Upload an urban image",
        type=["jpg", "jpeg", "png", "webp"],
        help=(
            "Best results come from clear street, junction, market, neighborhood or public-space images. "
            "Very distant or heavily occluded objects can still be missed."
        ),
        key="visual_ai_image",
    )

    if uploaded_image is None:
        st.info(
            "Upload an image to start. For the demo, use a street scene with visible people "
            "and vehicles so the activity analysis has enough evidence."
        )
    else:
        try:
            image = Image.open(uploaded_image).convert("RGB")
            preview_col, analysis_col = st.columns([1.2, 1])

            with preview_col:
                st.markdown("#### Uploaded image")
                st.image(image, caption=uploaded_image.name, width="stretch")

            with st.spinner(
                "City Pulse AI is running high-accuracy visual analysis. "
                "Crowded images may take a little longer..."
            ):
                model, model_name = load_visual_model()
                (
                    result,
                    detected_counts,
                    full_confidences,
                    analysis_confidences,
                    used_tiled_pass,
                ) = run_visual_detection(
                    image,
                    model,
                    use_crowded_scene_pass=use_crowded_scene_pass,
                )

            metrics = visual_metrics_from_counts(detected_counts)

            people = metrics["people"]
            cars = metrics["cars"]
            buses = metrics["buses"]
            trucks = metrics["trucks"]
            motorcycles = metrics["motorcycles"]
            bicycles = metrics["bicycles"]
            traffic_lights = metrics["traffic_lights"]
            vehicle_total = metrics["vehicle_total"]
            visual_score = metrics["visual_score"]
            traffic_level = metrics["traffic_level"]
            pedestrian_level = metrics["pedestrian_level"]

            estimated_zone = estimate_visual_zone(detected_counts)
            visual_level = visual_activity_level(visual_score)

            if traffic_level == "Very High" or visual_score >= 82:
                visual_risk = "High"
                visual_risk_color = "#FF5D73"
            elif traffic_level == "High" or visual_score >= 58:
                visual_risk = "Medium"
                visual_risk_color = "#FFB648"
            else:
                visual_risk = "Low"
                visual_risk_color = "#2ED6A1"

            avg_conf = (
                sum(analysis_confidences) / len(analysis_confidences) * 100
                if analysis_confidences
                else 0
            )

            with analysis_col:
                st.markdown("#### Visual AI summary")
                st.markdown(
                    f"""
                    <div class="report-card">
                        <div class="report-label">Estimated urban profile</div>
                        <div style="font-size:1.55rem;font-weight:850;margin:4px 0 14px 0;">
                            {estimated_zone}
                        </div>

                        <div class="report-label">Visual activity score</div>
                        <div class="report-value">{visual_score:.0f}/100 · {visual_level}</div>

                        <div class="report-label">Traffic level</div>
                        <div class="report-value">{traffic_level}</div>

                        <div class="report-label">Pedestrian activity</div>
                        <div class="report-value">{pedestrian_level}</div>

                        <div class="report-label">Visual risk</div>
                        <div style="font-size:1.2rem;font-weight:800;color:{visual_risk_color};margin-top:4px;">
                            {visual_risk}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.progress(min(1.0, visual_score / 100.0))
                st.caption(
                    f"Detector: {model_name} · analysis confidence avg: {avg_conf:.0f}%"
                    + (" · enhanced tiled pass used" if used_tiled_pass else " · full-frame pass sufficient")
                )

            st.markdown("<br>", unsafe_allow_html=True)

            metric_cols = st.columns(6)
            visual_metrics = [
                ("🚶", "People", people),
                ("🚗", "Cars", cars),
                ("🚌", "Buses", buses),
                ("🚚", "Trucks", trucks),
                ("🏍️", "Motorcycles", motorcycles),
                ("🚲", "Bicycles", bicycles),
            ]

            for col, (icon, label, value) in zip(metric_cols, visual_metrics):
                with col:
                    metric_card(icon, label.upper(), str(value), "Detected")

            st.markdown("<br>", unsafe_allow_html=True)

            annotated = result.plot()
            # Ultralytics plot output is BGR; Streamlit expects RGB.
            annotated_rgb = annotated[:, :, ::-1]

            annotated_col, rec_col = st.columns([1.25, 1])

            with annotated_col:
                st.markdown("#### Detected objects")
                st.image(
                    annotated_rgb,
                    caption=(
                        f"{model_name} full-frame detections"
                        + (
                            " · enhanced counting also used tiled analysis"
                            if used_tiled_pass
                            else ""
                        )
                    ),
                    width="stretch",
                )

                if used_tiled_pass:
                    st.caption(
                        "The displayed boxes come from the full-frame pass. Analysis counts may be higher "
                        "because the enhanced tiled pass can recover small distant objects."
                    )

            with rec_col:
                st.markdown("#### 🤖 Visual recommendations")

                recommendations = visual_recommendations(
                    visual_score,
                    traffic_level,
                    pedestrian_level,
                    people,
                    vehicle_total,
                    buses,
                    trucks,
                )

                for recommendation in recommendations:
                    st.markdown(
                        f'<div class="insight-box">✓ {recommendation}</div>',
                        unsafe_allow_html=True,
                    )

                if visual_risk == "High":
                    st.error(
                        "High visual activity detected. Treat this as a screening alert and verify "
                        "with live traffic/sensor data before operational decisions."
                    )
                elif visual_risk == "Medium":
                    st.warning(
                        "Moderate-to-high activity detected. Monitoring during peak periods is recommended."
                    )
                else:
                    st.success(
                        "The visible activity appears manageable in this single image."
                    )

                st.caption(
                    "This module estimates urban activity from a single image. It does not measure speed, "
                    "flow over time, or exact geographic location."
                )

            st.markdown("<br>", unsafe_allow_html=True)

            supported = {
                key: value
                for key, value in detected_counts.items()
                if key in SUPPORTED_URBAN_CLASSES and value > 0
            }

            if supported:
                detection_df = pd.DataFrame(
                    {
                        "Object": list(supported.keys()),
                        "Count": list(supported.values()),
                    }
                ).sort_values("Count", ascending=False)

                fig_detect = px.bar(
                    detection_df,
                    x="Object",
                    y="Count",
                    color="Object",
                    title="Detected Urban Objects",
                )
                fig_detect.update_layout(showlegend=False)
                polish_figure(fig_detect, height=360)

                st.plotly_chart(
                    fig_detect,
                    use_container_width=True,
                    config={"displayModeBar": False},
                )

                detail_left, detail_right = st.columns(2)

                with detail_left:
                    st.markdown("#### Detection quality")
                    st.write(f"**Model:** {model_name}")
                    st.write("**Full-frame image size:** 1280 px inference")
                    st.write("**Confidence threshold:** 0.15")
                    st.write(
                        f"**Crowded-scene enhancement:** {'Used' if used_tiled_pass else 'Not required'}"
                    )

                with detail_right:
                    st.markdown("#### Interpretation")
                    st.write(f"**People detected:** {people}")
                    st.write(f"**Road vehicles detected:** {vehicle_total}")
                    st.write(f"**Traffic estimate:** {traffic_level}")
                    st.write(f"**Pedestrian estimate:** {pedestrian_level}")

            else:
                st.warning(
                    "No supported urban objects were detected. Try another image with clearer people, "
                    "vehicles or street infrastructure."
                )

            st.caption(
                "Important: pretrained YOLO models use a fixed object vocabulary. Some local vehicle types "
                "(for example auto-rickshaws) may be classified as another vehicle class or missed. "
                "Counts are therefore estimates rather than certified traffic measurements."
            )

        except Exception as exc:
            st.error(
                "Visual analysis could not be completed for this image. "
                "Try a standard JPG/PNG street image."
            )
            st.caption(f"Technical detail: {exc}")


# =========================================================
# MAP TAB
# =========================================================

with map_tab:
    st.markdown('<div class="section-kicker">Spatial intelligence</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Interactive Urban Activity Map</div>', unsafe_allow_html=True)

    if filtered_df.empty:
        st.warning("No locations are available to map with the current selection.")
    else:
        if searched_row is not None:
            center_lat = float(searched_row["Latitude"])
            center_lon = float(searched_row["Longitude"])
            zoom = 15
        else:
            center_lat = float(filtered_df["Latitude"].mean())
            center_lon = float(filtered_df["Longitude"].mean())
            zoom = 11

        city_map = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=zoom,
            tiles="CartoDB dark_matter",
            control_scale=True,
        )

        for _, row in filtered_df.iterrows():
            zone_color = ZONE_COLORS.get(row["Zone"], "#9FB0C7")
            base_radius = 4.5 + float(row["Activity_Score"]) / 18
            is_anomaly = row["Status"] == "Anomaly"
            is_searched = searched_row is not None and int(row["Location_ID"]) == int(searched_row["Location_ID"])

            radius = base_radius + (4 if is_anomaly else 0) + (4 if is_searched else 0)
            border_color = "#FFFFFF" if is_searched else ("#FF5D73" if is_anomaly else zone_color)
            border_weight = 3 if is_searched else (2 if is_anomaly else 1)

            popup_html = f"""
            <div style="width:260px;font-family:Arial,sans-serif;">
                <div style="font-size:18px;font-weight:800;margin-bottom:8px;">🏙️ City Pulse AI</div>
                <div><b>Location:</b> {int(row["Location_ID"])}</div>
                <div><b>Zone:</b> {row["Zone"]}</div>
                <div><b>Activity:</b> {row["Activity_Score"]:.1f}/100 ({row["Activity_Level"]})</div>
                <div><b>Traffic:</b> {row["Traffic_Density"]}/100</div>
                <div><b>Pedestrians:</b> {int(row["Pedestrian_Count"])}</div>
                <div><b>Vehicles:</b> {int(row["Vehicle_Count"])}</div>
                <div><b>Status:</b> {row["Status"]}</div>
                <div><b>Coordinates:</b> {row["Latitude"]:.5f}, {row["Longitude"]:.5f}</div>
            </div>
            """

            folium.CircleMarker(
                location=[row["Latitude"], row["Longitude"]],
                radius=radius,
                color=border_color,
                weight=border_weight,
                fill=True,
                fill_color=zone_color,
                fill_opacity=0.78,
                popup=folium.Popup(popup_html, max_width=320),
                tooltip=f"Location {int(row['Location_ID'])} • {row['Zone']}",
            ).add_to(city_map)

        legend_html = """
        <div style="
            position: fixed;
            bottom: 28px;
            left: 28px;
            width: 205px;
            z-index: 9999;
            padding: 12px 14px;
            border-radius: 12px;
            background: rgba(8,17,31,.92);
            color: white;
            border: 1px solid rgba(255,255,255,.22);
            font-size: 12px;
            box-shadow: 0 8px 20px rgba(0,0,0,.30);
        ">
            <b>ZONE LEGEND</b><br><br>
            <span style="color:#FF5D73">●</span> Commercial<br>
            <span style="color:#2ED6A1">●</span> Residential<br>
            <span style="color:#FF9F43">●</span> Industrial<br>
            <span style="color:#4DA3FF">●</span> Recreational<br>
            <span style="color:#A679FF">●</span> Mixed<br><br>
            Larger marker = higher activity<br>
            Red outline = anomaly<br>
            White outline = searched location
        </div>
        """
        city_map.get_root().html.add_child(folium.Element(legend_html))

        st_folium(city_map, width=None, height=650)

        if searched_row is not None:
            st.caption(
                f"Map focused on Location {int(searched_row['Location_ID'])}. "
                "Click the highlighted marker to inspect its details."
            )

# =========================================================
# DATA + EXPORT TAB
# =========================================================

with data_tab:
    st.markdown('<div class="section-kicker">Inspect & export</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Location Data</div>', unsafe_allow_html=True)

    if filtered_df.empty:
        st.warning("No rows are available for the current selection.")
    else:
        display_columns = [
            "Location_ID",
            "Zone",
            "Cluster",
            "Activity_Score",
            "Activity_Level",
            "Status",
            "Traffic_Density",
            "Pedestrian_Count",
            "Vehicle_Count",
            "Public_Transport",
            "Green_Space",
            "Latitude",
            "Longitude",
        ]
        display_columns = [col for col in display_columns if col in filtered_df.columns]

        table_df = filtered_df[display_columns].copy()
        table_df["Activity_Score"] = table_df["Activity_Score"].round(2)

        def highlight_anomaly(row):
            if row.get("Status") == "Anomaly":
                return ["background-color: rgba(255, 93, 115, 0.18)"] * len(row)
            return [""] * len(row)

        st.dataframe(
            table_df.style.apply(highlight_anomaly, axis=1),
            use_container_width=True,
            height=480,
        )

        st.markdown('<div class="section-title">Export Current View</div>', unsafe_allow_html=True)

        export_col, info_col = st.columns([0.55, 1.45])
        with export_col:
            csv_data = filtered_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Download CSV",
                data=csv_data,
                file_name="city_pulse_results.csv",
                mime="text/csv",
                use_container_width=True,
            )

        with info_col:
            st.info(
                f"The export contains {len(filtered_df):,} row(s) and reflects the current "
                "location search or sidebar filters."
            )

# =========================================================
# FOOTER
# =========================================================

st.divider()
st.markdown(
    """
    <div class="footer">
        <b>🏙️ City Pulse AI</b><br>
        Intelligent Urban Activity Analysis & Zone Detection System<br>
        K-Means Clustering • Isolation Forest • Streamlit • Plotly • Folium
    </div>
    """,
    unsafe_allow_html=True,
)
