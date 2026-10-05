"""Train churn prediction model + rank churn drivers."""
import os
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)

from preprocess import prepare_data


MODELS = {
    "logistic_regression": LogisticRegression(max_iter=1000, random_state=42),
    "random_forest": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
}


def evaluate(model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred)),
        "auc_roc": float(roc_auc_score(y_test, y_proba)),
    }


def train_all():
    os.makedirs("models", exist_ok=True)
    df = pd.read_csv("data/churn.csv")
    X_train, X_test, y_train, y_test, preprocessor, features = prepare_data(df)

    best_auc = 0
    best_name = None
    best_model = None
    results = {}

    for name, model in MODELS.items():
        print(f"\n[Training] {name}...")
        model.fit(X_train, y_train)
        metrics = evaluate(model, X_test, y_test)
        results[name] = metrics
        print(f"   Accuracy:  {metrics['accuracy']:.4f}")
        print(f"   Precision: {metrics['precision']:.4f}")
        print(f"   Recall:    {metrics['recall']:.4f}")
        print(f"   F1:        {metrics['f1']:.4f}")
        print(f"   AUC-ROC:   {metrics['auc_roc']:.4f}")

        if metrics["auc_roc"] > best_auc:
            best_auc = metrics["auc_roc"]
            best_name = name
            best_model = model

    joblib.dump(best_model, "models/churn_model.joblib")
    joblib.dump(preprocessor, "models/preprocessor.joblib")

    # Churn drivers (feature importance)
    if best_name == "random_forest":
        importances = best_model.feature_importances_
    else:
        importances = np.abs(best_model.coef_[0])

    drivers = sorted(
        zip(features, importances.tolist()),
        key=lambda x: -x[1]
    )[:15]

    metadata = {
        "best_model": best_name,
        "auc_roc": best_auc,
        "all_results": results,
        "top_churn_drivers": [{"feature": f, "importance": round(v, 4)} for f, v in drivers],
        "feature_names": features,
    }

    with open("models/metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n[Best] {best_name} (AUC-ROC: {best_auc:.4f})")
    print(f"\n[Top churn drivers]")
    for f, v in drivers[:8]:
        print(f"   {f}: {v:.4f}")

    return metadata


if __name__ == "__main__":
    train_all()
