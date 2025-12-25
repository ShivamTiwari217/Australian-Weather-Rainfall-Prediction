import streamlit as st
import pandas as pd
import joblib
from src.data_preprocessing import preprocess_data

st.set_page_config(page_title="Rainfall Prediction", layout="centered")
st.title("🌧️ Rainfall Prediction App")

model_choice = st.selectbox(
    "Choose Model",
    ["Random Forest", "Logistic Regression"]
)

model_path = (
    "models/random_forest.pkl"
    if model_choice == "Random Forest"
    else "models/logistic_regression.pkl"
)

model = joblib.load(model_path)

uploaded_file = st.file_uploader("Upload Weather CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Data Preview", df.head())

    df_processed = preprocess_data(df)
    predictions = model.predict(df_processed)

    df["Rain Prediction"] = predictions
    st.success("Prediction Completed")
    st.write(df.head())
