import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# --------------------------------------------------
# Paths
# --------------------------------------------------
DATA_PATH = "data/weatherAUS.csv"
MODELS_DIR = "models"

os.makedirs(MODELS_DIR, exist_ok=True)

# --------------------------------------------------
# Load & preprocess data
# --------------------------------------------------
df = pd.read_csv(DATA_PATH)
df = df.dropna()

# Encode categorical variables
for col in df.select_dtypes(include="object").columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

X = df.drop("RainTomorrow", axis=1)
y = df["RainTomorrow"]

# 🔑 SAVE FEATURE SCHEMA (CRITICAL)
joblib.dump(
    X.columns.tolist(),
    os.path.join(MODELS_DIR, "feature_columns.pkl")
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# --------------------------------------------------
# RANDOM FOREST (light, stable)
# --------------------------------------------------
print("Training Random Forest...")

rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=1
)

rf_model.fit(X_train, y_train)

joblib.dump(
    rf_model,
    os.path.join(MODELS_DIR, "random_forest.pkl")
)

print("✅ Random Forest saved")

# --------------------------------------------------
# LOGISTIC REGRESSION (baseline)
# --------------------------------------------------
print("Training Logistic Regression...")

lr_model = LogisticRegression(
    C=1.0,
    penalty="l2",
    solver="liblinear",
    max_iter=1000
)

lr_model.fit(X_train, y_train)

joblib.dump(
    lr_model,
    os.path.join(MODELS_DIR, "logistic_regression.pkl")
)

print("✅ Logistic Regression saved")

print("\n🎉 ALL MODELS & FEATURE SCHEMA SAVED SUCCESSFULLY")