import os
import sys
sys.path.append(os.path.abspath("."))

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
import shap
import matplotlib.pyplot as plt

from src.data_preprocessing import preprocess_data

# --------------------------------------------------
# Page configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Australian Rainfall Prediction",
    layout="wide"
)

st.title("🌧️ Australian Rainfall Prediction Dashboard")
st.caption(
    "End-to-end ML system with probabilistic prediction, spatial insights, and explainability"
)

# --------------------------------------------------
# Paths
# --------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")

MODEL_PATHS = {
    "Random Forest": os.path.join(MODELS_DIR, "random_forest.pkl"),
    "Logistic Regression": os.path.join(MODELS_DIR, "logistic_regression.pkl"),
}

FEATURES_PATH = os.path.join(MODELS_DIR, "feature_columns.pkl")

# --------------------------------------------------
# Load feature schema
# --------------------------------------------------
feature_columns = joblib.load(FEATURES_PATH)

# --------------------------------------------------
# Sidebar controls
# --------------------------------------------------
st.sidebar.header("⚙️ Controls")

model_choice = st.sidebar.selectbox(
    "Select Model",
    list(MODEL_PATHS.keys())
)

model = joblib.load(MODEL_PATHS[model_choice])

st.sidebar.markdown("---")
st.sidebar.subheader("Data Input")

data_mode = st.sidebar.radio(
    "Choose input method",
    ["Use sample data", "Upload CSV", "Manual input (single prediction)"]
)

# --------------------------------------------------
# Load data based on mode
# --------------------------------------------------
df = None

if data_mode == "Use sample data":
    df = pd.read_csv(os.path.join(DATA_DIR, "weatherAUS.csv"))
    st.info("Using built-in sample dataset")

elif data_mode == "Upload CSV":
    uploaded_file = st.file_uploader("Upload Australian Weather CSV", type=["csv"])
    if uploaded_file is None:
        st.stop()
    df = pd.read_csv(uploaded_file)

elif data_mode == "Manual input (single prediction)":
    st.subheader("📝 Manual Weather Input")

    humidity = st.slider("Humidity (%)", 0, 100, 50)
    temp = st.slider("Temperature (°C)", -5, 45, 25)
    wind = st.slider("Wind Speed (km/h)", 0, 100, 10)
    pressure = st.slider("Pressure (hPa)", 980, 1050, 1010)

    input_dict = {
        "Humidity9am": humidity,
        "Humidity3pm": humidity,
        "Temp9am": temp,
        "Temp3pm": temp,
        "WindSpeed9am": wind,
        "WindSpeed3pm": wind,
        "Pressure9am": pressure,
        "Pressure3pm": pressure,
    }

    df = pd.DataFrame([input_dict])

# --------------------------------------------------
# Proceed only if data exists
# --------------------------------------------------
if df is not None:
    st.subheader("📄 Data Preview")
    st.dataframe(df.head())

# --------------------------------------------------
# Preprocess & align features
# --------------------------------------------------
# 1. Convert Streamlit input to DataFrame
input_df = pd.DataFrame([user_input])

# 2. Preprocess
df_processed = preprocess_data(input_df)

# 3. Enforce feature schema
df_processed = df_processed.reindex(
    columns=feature_columns,
    fill_value=0
)

# 4. Predict
preds = model.predict(df_processed)
probs = model.predict_proba(df_processed)[:, 1]


df["RainTomorrow_Pred"] = preds
df["Rain_Probability"] = probs

# --------------------------------------------------
# KPI Summary
# --------------------------------------------------
st.subheader("📌 Prediction Summary")

c1, c2, c3 = st.columns(3)
c1.metric("🌧️ Rain Expected (%)", f"{preds.mean() * 100:.1f}%")
c2.metric("🎯 Avg Rain Probability", f"{probs.mean():.2f}")
c3.metric("📁 Records", len(df))

# --------------------------------------------------
# Confidence Gauge
# --------------------------------------------------
st.subheader("🎯 Prediction Confidence")

fig_gauge = go.Figure(go.Indicator(
    mode="gauge+number",
    value=float(probs.mean()),
    gauge={"axis": {"range": [0, 1]}},
    title={"text": "Average Rain Probability"}
))
st.plotly_chart(fig_gauge, use_container_width=True)

# --------------------------------------------------
# Trend
# --------------------------------------------------
st.subheader("📈 Rain Probability Trend")

df["Index"] = range(len(df))
fig_trend = px.line(
    df,
    x="Index",
    y="Rain_Probability",
    labels={"Rain_Probability": "Probability of Rain"}
)
st.plotly_chart(fig_trend, use_container_width=True)

# --------------------------------------------------
# Australia Map
# --------------------------------------------------
if "Location" in df.columns:
    st.subheader("🗺️ Rainfall Intensity Map (Australia)")

    map_df = (
        df.groupby("Location", as_index=False)["Rain_Probability"]
        .mean()
    )

    fig_map = px.scatter_geo(
        map_df,
        locations="Location",
        locationmode="country names",
        scope="australia",
        size="Rain_Probability",
        color="Rain_Probability",
        color_continuous_scale="Blues"
    )
    st.plotly_chart(fig_map, use_container_width=True)

# --------------------------------------------------
# SHAP Explainability
# --------------------------------------------------
st.subheader("🧠 SHAP Explainability")

shap_sample = df_processed.sample(
    n=min(200, len(df_processed)),
    random_state=42
)

if model_choice == "Random Forest":
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(shap_sample)[1]
else:
    explainer = shap.LinearExplainer(model, shap_sample)
    shap_values = explainer.shap_values(shap_sample)

fig, ax = plt.subplots()
shap.summary_plot(
    shap_values,
    shap_sample,
    plot_type="bar",
    show=False
)
st.pyplot(fig)

# --------------------------------------------------
# Download results
# --------------------------------------------------
st.subheader("⬇️ Download Predictions")

st.download_button(
    "Download CSV",
    df.to_csv(index=False),
    file_name="rainfall_predictions.csv",
    mime="text/csv"
)