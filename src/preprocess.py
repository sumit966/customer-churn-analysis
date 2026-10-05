"""Preprocess churn data: encode categoricals, split, scale."""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


NUMERIC_COLS = ["tenure", "monthly_charges", "total_charges", "senior_citizen"]
CATEGORICAL_COLS = [
    "gender", "partner", "dependents", "contract",
    "internet_service", "tech_support", "payment_method", "paperless_billing"
]
TARGET = "churn"


def build_preprocessor() -> ColumnTransformer:
    """Return a ColumnTransformer for numeric + categorical features."""
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_COLS),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_COLS),
        ]
    )


def prepare_data(df: pd.DataFrame, test_size: float = 0.2, seed: int = 42):
    """Split into train and test."""
    X = df[NUMERIC_COLS + CATEGORICAL_COLS]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    preprocessor = build_preprocessor()
    X_train_enc = preprocessor.fit_transform(X_train)
    X_test_enc = preprocessor.transform(X_test)

    feature_names = preprocessor.get_feature_names_out()

    return X_train_enc, X_test_enc, y_train, y_test, preprocessor, list(feature_names)


if __name__ == "__main__":
    df = pd.read_csv("data/churn.csv")
    X_train, X_test, y_train, y_test, preprocessor, features = prepare_data(df)
    print(f"Train shape: {X_train.shape}")
    print(f"Test shape: {X_test.shape}")
    print(f"Features: {len(features)}")
