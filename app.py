import os
import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go

from src.data_preprocessing import preprocess_data

# ----------------------------
# Config
# ----------------------------
st.set_page_config(
    page_title="Rainfall Prediction",
    layout="wide"
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
with st.sidebar:
    st.header("🔧 Configuration")
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
        st.markdown("---")
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

    # Metrics
    m1, m2, m3 = st.columns(3)
    avg_prob = probs.mean()

    with m1:
        st.metric(
            label="Average Probability",
            value=f"{avg_prob * 100:.1f}%"
        )

    with m2:
        high_risk_count = (probs > 0.5).sum()
        st.metric(
            label="High Risk Days (>50%)",
            value=high_risk_count
        )

    with m3:
        st.metric(
            label="Total Records",
            value=len(df)
        )

    st.markdown("---")

    # Visualizations
    col_viz1, col_viz2 = st.columns(2)

    if len(df) == 1:
        # Gauge Chart for single prediction
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = avg_prob * 100,
            title = {'text': "Rain Probability"},
            gauge = {
                'axis': {'range': [0, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 30], 'color': "lightgreen"},
                    {'range': [30, 70], 'color': "yellow"},
                    {'range': [70, 100], 'color': "salmon"}],
            }
        ))
        st.plotly_chart(fig, use_container_width=True)

    else:
        # Histogram
        with col_viz1:
            fig_hist = px.histogram(
                df,
                x="Rain_Probability",
                nbins=20,
                title="Probability Distribution",
                labels={"Rain_Probability": "Probability"},
                color_discrete_sequence=['#1f77b4']
            )
            st.plotly_chart(fig_hist, use_container_width=True)

        # Scatter Plot
        with col_viz2:
            if "Humidity3pm" in df.columns:
                fig_scatter = px.scatter(
                    df,
                    x="Humidity3pm",
                    y="Rain_Probability",
                    color="Rain_Probability",
                    title="Humidity vs Rain Probability",
                    color_continuous_scale="Bluered"
                )
                st.plotly_chart(fig_scatter, use_container_width=True)

    st.markdown("### Detailed Data")

    # Configure columns for dataframe
    cols_to_show = ["Rain_Probability"]
    # Add other useful columns if they exist
    potential_cols = ["Date", "Location", "Humidity3pm", "Temp3pm"]
    for c in potential_cols:
        if c in df.columns:
            cols_to_show.insert(0, c)

    st.dataframe(
        df[cols_to_show],
        column_config={
            "Rain_Probability": st.column_config.ProgressColumn(
                "Rain Probability",
                help="The probability of rain",
                format="%.2f",
                min_value=0,
                max_value=1,
            ),
        },
        use_container_width=True
    )

    # ----------------------------
    # Download
    # ----------------------------
    st.download_button(
        "⬇️ Download Results",
        df.to_csv(index=False),
        file_name="rainfall_predictions.csv",
        mime="text/csv"
    )
