import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def load_and_prepare_data(csv_path):
    df = pd.read_csv(csv_path)

    # Drop identifier columns
    df = df.drop(columns=["UDI", "Product ID"])

    # Encode categorical variable
    df["Type"] = df["Type"].map({"L": 0, "M": 1, "H": 2})

    target = "Machine failure"
    features = [c for c in df.columns if c != target]

    X = df[features].values
    y = df[target].values

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X, y, features
