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
# Page config
# --------------------------------------------------
st.set_page_config(
    page_title="Australian Rainfall Prediction",
    layout="wide"
)

st.title("🌧️ Australian Rainfall Prediction Dashboard")
st.caption("Prediction • Confidence • Trends • Maps • Explainability")

# --------------------------------------------------
# Paths
# --------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

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
# Sidebar
# --------------------------------------------------
model_choice = st.sidebar.selectbox(
    "Select Model",
    list(MODEL_PATHS.keys())
)

model = joblib.load(MODEL_PATHS[model_choice])

# --------------------------------------------------
# File upload
# --------------------------------------------------
uploaded
