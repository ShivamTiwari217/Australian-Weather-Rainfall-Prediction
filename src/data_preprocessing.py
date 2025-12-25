import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def load_data(path):
    return pd.read_csv(path)

def preprocess_data(df):
    df = df.dropna()
    
    label_encoders = {}
    for col in df.select_dtypes(include='object').columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    return df, label_encoders

def split_data(df, target='RainTomorrow'):
    X = df.drop(columns=[target])
    y = df[target]

    return train_test_split(X, y, test_size=0.2, random_state=42)
