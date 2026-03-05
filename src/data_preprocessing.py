import os
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


RAW_DATA_PATH = os.path.join("data", "raw", "ai4i2020.csv")


def load_raw_dataframe(csv_path: str = RAW_DATA_PATH) -> pd.DataFrame:
    """
    Load the raw AI4I 2020 dataset.
    """
    return pd.read_csv(csv_path)


def build_features_and_target(
    df: pd.DataFrame, target_col: str = "Machine failure"
) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
    """
    Apply feature engineering consistent with the original implementation:
    - Drop identifier columns
    - Encode the categorical 'Type' column
    - Separate features and target
    """
    # Drop identifier columns if they exist
    drop_cols = [c for c in ["UDI", "Product ID"] if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    # Encode categorical variable
    if "Type" in df.columns:
        if df["Type"].dtype == object:
            df["Type"] = df["Type"].map({"L": 0, "M": 1, "H": 2})

    features = [c for c in df.columns if c != target_col]

    X_df = df[features]
    y = df[target_col].values

    return X_df, y, features


def fit_scaler(X_df: pd.DataFrame) -> Tuple[np.ndarray, StandardScaler]:
    """
    Fit a StandardScaler on the feature dataframe and return the
    transformed array and the fitted scaler.
    """
    scaler = StandardScaler()
    X = scaler.fit_transform(X_df.values)
    return X, scaler


def prepare_training_data(
    csv_path: str = RAW_DATA_PATH, target_col: str = "Machine failure"
) -> Tuple[np.ndarray, np.ndarray, List[str], StandardScaler]:
    """
    High-level helper that:
    - Loads the raw dataset
    - Builds features and target
    - Fits the scaler

    Returns:
    - X: scaled feature matrix
    - y: target array
    - features: list of feature names
    - scaler: fitted StandardScaler instance
    """
    df = load_raw_dataframe(csv_path)
    X_df, y, features = build_features_and_target(df, target_col=target_col)
    X, scaler = fit_scaler(X_df)
    return X, y, features, scaler


def preprocess_input_records(
    records, features: List[str], scaler: StandardScaler
) -> np.ndarray:
    """
    Preprocess one or more input records for inference using the
    trained scaler and feature ordering.

    Parameters
    ----------
    records : dict or list of dict
        Input records where keys correspond to column names.
    features : list of str
        Feature names in the order expected by the model.
    scaler : StandardScaler
        Fitted scaler from training.
    """
    if isinstance(records, dict):
        data = [records]
    else:
        data = list(records)

    df = pd.DataFrame(data)

    # Drop identifier columns if present
    drop_cols = [c for c in ["UDI", "Product ID"] if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    # Encode categorical 'Type' if present and still non-numeric
    if "Type" in df.columns and df["Type"].dtype == object:
        df["Type"] = df["Type"].map({"L": 0, "M": 1, "H": 2})

    missing = [f for f in features if f not in df.columns]
    if missing:
        raise ValueError(f"Missing required features in input: {missing}")

    X_df = df[features]
    X = scaler.transform(X_df.values)
    return X

