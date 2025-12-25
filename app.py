import os
import sys
sys.path.append(os.path.abspath("."))

import streamlit as st
import pandas as pd
import joblib
import numpy as np
import shap
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, auc, confusion_matrix
from src.data_preprocessing import preprocess_data


# --------------------------------------------------
# Page config
# --------------------------------------------------
st.set_page_config(page_title="Australian Rainfall ML Dashboard", layout="wide")
st.title("🌧️ Australian Rainfall Prediction & Explainability Dashboard")

# --------------------------------------------------
# Load models
# --------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATHS = {
    "Random Forest": os.path.join(BASE_DIR, "models", "random_forest.pkl"),
    "Logistic Regression": os.path.join(BASE_DIR, "models", "logistic_regression.pkl"),
}

model_choice = st.sidebar.selectbox("Select Model", list(MODEL_PATHS.keys()))
model = joblib.load(MODEL_PATHS[model_choice])

# --------------------------------------------------
# Upload data
# --------------------------------------------------
uploaded_file = st.file_uploader("Upload Weather CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    X = preprocess_data(df)

    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]

    df["RainTomorrow_Pred"] = y_pred
    df["Rain_Probability"] = y_prob

    # ==================================================
    # KPI SECTION
    # ==================================================
    st.subheader("Key Metrics")

    c1, c2, c3 = st.columns(3)
    c1.metric("Rain Expected (%)", f"{df['RainTomorrow_Pred'].mean()*100:.1f}%")
    c2.metric("Avg Rain Probability", f"{df['Rain_Probability'].mean():.2f}")
    c3.metric("Observations", len(df))

    # ==================================================
    # CONFIDENCE GAUGE
    # ==================================================
    st.subheader("Prediction Confidence")

    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=df["Rain_Probability"].mean(),
        gauge={"axis": {"range": [0, 1]}},
        title={"text": "Average Rain Probability"}
    ))
    st.plotly_chart(fig_gauge, use_container_width=True)

    # ==================================================
    # TREND
    # ==================================================
    st.subheader("Rain Probability Trend")

    df["Index"] = range(len(df))
    fig_trend = px.line(df, x="Index", y="Rain_Probability")
    st.plotly_chart(fig_trend, use_container_width=True)

    # ==================================================
    # AUSTRALIA MAP
    # ==================================================
    if "Location" in df.columns:
        st.subheader("Australia Rainfall Intensity Map")

        loc_df = df.groupby("Location", as_index=False)["Rain_Probability"].mean()
        fig_map = px.scatter_geo(
            loc_df,
            locations="Location",
            locationmode="country names",
            scope="australia",
            size="Rain_Probability",
            color="Rain_Probability",
            color_continuous_scale="Blues"
        )
        st.plotly_chart(fig_map, use_container_width=True)

    # ==================================================
    # ROC & CONFUSION MATRIX
    # ==================================================
    st.subheader("Model Evaluation")

    fpr, tpr, _ = roc_curve(df["RainTomorrow_Pred"], y_prob)
    roc_auc = auc(fpr, tpr)

    fig_roc = px.line(x=fpr, y=tpr, title=f"ROC Curve (AUC={roc_auc:.2f})")
    st.plotly_chart(fig_roc, use_container_width=True)

    cm = confusion_matrix(df["RainTomorrow_Pred"], y_pred)
    fig_cm = px.imshow(cm, text_auto=True, title="Confusion Matrix")
    st.plotly_chart(fig_cm, use_container_width=True)

    # ==================================================
    # SHAP EXPLAINABILITY
    # ==================================================
    st.subheader("SHAP Explainability")

    st.markdown("### Global Feature Impact")

    if model_choice == "Random Forest":
        explainer = shap.TreeExplainer(model.named_steps["rf"])
        shap_values = explainer.shap_values(X)[1]
    else:
        explainer = shap.LinearExplainer(model.named_steps["lr"], X)
        shap_values = explainer.shap_values(X)

    fig, ax = plt.subplots()
    shap.summary_plot(shap_values, X, plot_type="bar", show=False)
    st.pyplot(fig)

    st.markdown("### Local Explanation (Single Prediction)")
    idx = st.slider("Select observation", 0, len(X)-1, 0)

    fig2, ax2 = plt.subplots()
    shap.force_plot(
        explainer.expected_value if model_choice=="Random Forest" else explainer.expected_value,
        shap_values[idx],
        X.iloc[idx],
        matplotlib=True,
        show=False
    )
    st.pyplot(fig2)

    # ==================================================
    # DOWNLOAD
    # ==================================================
    st.subheader("Download Predictions")

    st.download_button(
        "Download CSV",
        data=df.to_csv(index=False),
        file_name="rainfall_predictions.csv",
        mime="text/csv"
    )
