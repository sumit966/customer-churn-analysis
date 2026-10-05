"""Generate synthetic telecom customer churn dataset."""
import os
import numpy as np
import pandas as pd


def generate_churn_data(n_customers: int = 5000, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)

    # Contract type heavily influences churn
    contract = np.random.choice(
        ["Month-to-month", "One year", "Two year"],
        n_customers, p=[0.55, 0.25, 0.20]
    )

    tenure = np.random.randint(0, 72, n_customers)
    monthly_charges = np.random.uniform(20, 120, n_customers).round(2)
    total_charges = (tenure * monthly_charges).round(2)

    internet_service = np.random.choice(
        ["DSL", "Fiber optic", "No"],
        n_customers, p=[0.35, 0.45, 0.20]
    )
    payment_method = np.random.choice(
        ["Electronic check", "Mailed check", "Bank transfer", "Credit card"],
        n_customers, p=[0.35, 0.20, 0.20, 0.25]
    )
    tech_support = np.random.choice(["Yes", "No"], n_customers, p=[0.30, 0.70])
    paperless_billing = np.random.choice(["Yes", "No"], n_customers, p=[0.60, 0.40])

    gender = np.random.choice(["Male", "Female"], n_customers)
    senior_citizen = np.random.choice([0, 1], n_customers, p=[0.85, 0.15])
    partner = np.random.choice(["Yes", "No"], n_customers, p=[0.50, 0.50])
    dependents = np.random.choice(["Yes", "No"], n_customers, p=[0.30, 0.70])

    # Churn probability logic
    churn_score = (
        0.5 * (contract == "Month-to-month").astype(float)
        + 0.3 * (internet_service == "Fiber optic").astype(float)
        + 0.4 * (payment_method == "Electronic check").astype(float)
        + 0.3 * (tech_support == "No").astype(float)
        + 0.2 * (tenure < 12).astype(float)
        + 0.2 * (monthly_charges > 80).astype(float)
        - 0.15 * (partner == "Yes").astype(float)
        - 0.15 * (dependents == "Yes").astype(float)
    )
    probability = 1 / (1 + np.exp(-(churn_score - 1.0)))
    churn = (np.random.random(n_customers) < probability).astype(int)

    df = pd.DataFrame({
        "customer_id": [f"C{str(i).zfill(5)}" for i in range(1, n_customers + 1)],
        "gender": gender,
        "senior_citizen": senior_citizen,
        "partner": partner,
        "dependents": dependents,
        "tenure": tenure,
        "contract": contract,
        "internet_service": internet_service,
        "tech_support": tech_support,
        "payment_method": payment_method,
        "paperless_billing": paperless_billing,
        "monthly_charges": monthly_charges,
        "total_charges": total_charges,
        "churn": churn,
    })

    return df


def save_dataset(output_dir: str = "data"):
    os.makedirs(output_dir, exist_ok=True)
    df = generate_churn_data()
    path = f"{output_dir}/churn.csv"
    df.to_csv(path, index=False)
    print(f"[OK] Generated {len(df)} customers -> {path}")
    print(f"[OK] Churn rate: {df['churn'].mean():.2%}")
    return df


if __name__ == "__main__":
    df = save_dataset()
    print(df.head())
