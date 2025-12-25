import pandas as pd

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Drop target if present
    df = df.drop(columns=["RainTomorrow"], errors="ignore")

    # Encode categorical variables
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype("category").cat.codes

    # Ensure numeric
    df = df.apply(pd.to_numeric, errors="coerce")

    # Handle missing values
    df = df.fillna(0)

    return df
