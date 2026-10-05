"""Evaluate trained churn model."""
import json
import joblib
import pandas as pd
from preprocess import prepare_data
from train import evaluate


def evaluate_saved():
    model = joblib.load("models/churn_model.joblib")
    preprocessor = joblib.load("models/preprocessor.joblib")

    with open("models/metadata.json") as f:
        metadata = json.load(f)

    df = pd.read_csv("data/churn.csv")
    X_train, X_test, y_train, y_test, _, _ = prepare_data(df)

    metrics = evaluate(model, X_test, y_test)

    print(f"\n[Eval] Model: {metadata['best_model']}")
    for k, v in metrics.items():
        print(f"   {k}: {v:.4f}")


if __name__ == "__main__":
    evaluate_saved()
