import os
import streamlit as st
import pandas as pd
import joblib

from src.data_preprocessing import preprocess_data

# ----------------------------
# Config
# ----------------------------
st.set_page_config(
    page_title="Rainfall Prediction",
    layout="centered"
)

st.title("🌧️ Rainfall Prediction")
st.caption("Simple ML-based probability of rainfall")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

MODEL_PATH = os.path.join(MODELS_DIR, "random_forest.pkl")
FEATURES_PATH = os.path.join(MODELS_DIR, "feature_columns.pkl")

model = joblib.load(MODEL_PATH)
feature_columns = joblib.load(FEATURES_PATH)

# ----------------------------
# Input
# ----------------------------
st.subheader("📥 Input Data")

input_mode = st.radio(
    "Choose input method",
    ["Sample data", "Upload CSV", "Manual input"]
)

df = None

if input_mode == "Sample data":
    df = pd.read_csv("data/weatherAUS.csv")
    st.info("Using sample dataset")

elif input_mode == "Upload CSV":
    file = st.file_uploader("Upload CSV", type=["csv"])
    if file:
        df = pd.read_csv(file)

elif input_mode == "Manual input":
    humidity = st.slider("Humidity (%)", 0, 100, 50)
    temp = st.slider("Temperature (°C)", -5, 45, 25)
    pressure = st.slider("Pressure (hPa)", 980, 1050, 1010)

    df = pd.DataFrame([{
        "Humidity9am": humidity,
        "Humidity3pm": humidity,
        "Temp9am": temp,
        "Temp3pm": temp,
        "Pressure9am": pressure,
        "Pressure3pm": pressure,
    }])

# ----------------------------
# Prediction
# ----------------------------
if df is not None:
    df_processed = preprocess_data(df)

    df_processed = df_processed.reindex(
        columns=feature_columns,
        fill_value=0
    )

    probs = model.predict_proba(df_processed)[:, 1]
    df["Rain_Probability"] = probs

    st.subheader("🔮 Prediction")

    st.metric(
        label="Probability of Rain",
        value=f"{probs.mean() * 100:.1f}%"
    )

    st.dataframe(df[["Rain_Probability"]])

    # ----------------------------
    # Download
    # ----------------------------
    st.download_button(
        "⬇️ Download Results",
        df.to_csv(index=False),
        file_name="rainfall_predictions.csv",
        mime="text/csv"
    )
