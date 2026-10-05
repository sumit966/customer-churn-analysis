"""Generate churn charts and a Markdown report."""
import os
import json
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="darkgrid")


def save_charts(df: pd.DataFrame, output_dir: str = "reports"):
    os.makedirs(output_dir, exist_ok=True)

    # 1. Churn by contract type
    fig, ax = plt.subplots(figsize=(7, 4))
    ct = df.groupby("contract")["churn"].mean().reset_index()
    sns.barplot(data=ct, x="contract", y="churn", ax=ax, palette="Reds_r")
    ax.set_title("Churn Rate by Contract Type")
    ax.set_ylabel("Churn Rate")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/churn_by_contract.png", dpi=100)
    plt.close()

    # 2. Churn by tenure buckets
    df = df.copy()
    df["tenure_bucket"] = pd.cut(
        df["tenure"], bins=[-1, 12, 24, 48, 100],
        labels=["<1y", "1-2y", "2-4y", "4y+"]
    )
    fig, ax = plt.subplots(figsize=(7, 4))
    tt = df.groupby("tenure_bucket", observed=True)["churn"].mean().reset_index()
    sns.barplot(data=tt, x="tenure_bucket", y="churn", ax=ax, palette="Oranges_r")
    ax.set_title("Churn Rate by Tenure")
    ax.set_ylabel("Churn Rate")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/churn_by_tenure.png", dpi=100)
    plt.close()

    # 3. Churn by payment method
    fig, ax = plt.subplots(figsize=(8, 4))
    pm = df.groupby("payment_method")["churn"].mean().reset_index()
    sns.barplot(data=pm, x="payment_method", y="churn", ax=ax, palette="Purples_r")
    ax.set_title("Churn Rate by Payment Method")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/churn_by_payment.png", dpi=100)
    plt.close()

    # 4. Top churn drivers
    try:
        with open("models/metadata.json") as f:
            metadata = json.load(f)
        drivers = metadata["top_churn_drivers"][:10]
        fig, ax = plt.subplots(figsize=(8, 5))
        names = [d["feature"] for d in drivers][::-1]
        vals = [d["importance"] for d in drivers][::-1]
        ax.barh(names, vals, color="#ef4444")
        ax.set_title("Top Churn Drivers")
        ax.set_xlabel("Importance")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/top_churn_drivers.png", dpi=100)
        plt.close()
    except Exception as e:
        print(f"[WARN] Skipped driver chart: {e}")

    print(f"[OK] Charts saved to {output_dir}/")


def generate_report(df: pd.DataFrame, output_path: str = "reports/report.md"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open("models/metadata.json") as f:
        metadata = json.load(f)

    churn_rate = df["churn"].mean()
    n_churn = int(df["churn"].sum())
    n_total = len(df)

    lines = []
    lines.append("# Customer Churn Analysis Report\n")
    lines.append(f"Generated on {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}\n")

    lines.append("## Summary\n")
    lines.append(f"- Total customers: {n_total:,}")
    lines.append(f"- Churned customers: {n_churn:,}")
    lines.append(f"- Overall churn rate: {churn_rate:.2%}")
    lines.append(f"- Best model: {metadata['best_model']}")
    lines.append(f"- Best AUC-ROC: {metadata['auc_roc']:.4f}\n")

    lines.append("## Model Performance\n")
    lines.append("| Model | Accuracy | Precision | Recall | F1 | AUC-ROC |")
    lines.append("|-------|----------|-----------|--------|----|---------|")
    for name, m in metadata["all_results"].items():
        lines.append(
            f"| {name} | {m['accuracy']:.4f} | {m['precision']:.4f} | "
            f"{m['recall']:.4f} | {m['f1']:.4f} | {m['auc_roc']:.4f} |"
        )

    lines.append("\n## Top 10 Churn Drivers\n")
    lines.append("| Feature | Importance |")
    lines.append("|---------|-----------|")
    for d in metadata["top_churn_drivers"][:10]:
        lines.append(f"| {d['feature']} | {d['importance']:.4f} |")

    lines.append("\n## Churn Rate by Contract\n")
    ct = df.groupby("contract")["churn"].agg(["mean", "count"]).round(4).reset_index()
    ct.columns = ["Contract", "Churn Rate", "Customers"]
    lines.append(ct.to_markdown(index=False))

    lines.append("\n\n## Charts\n")
    lines.append("![Churn by Contract](churn_by_contract.png)\n")
    lines.append("![Churn by Tenure](churn_by_tenure.png)\n")
    lines.append("![Churn by Payment](churn_by_payment.png)\n")
    lines.append("![Top Churn Drivers](top_churn_drivers.png)\n")

    lines.append("\n## Retention Recommendations\n")
    lines.append("1. Target month-to-month customers with loyalty discounts or annual plan offers.")
    lines.append("2. Increase tech support adoption — support users churn at lower rates.")
    lines.append("3. Move electronic-check payers to auto-pay methods (credit card, bank transfer).")
    lines.append("4. Focus retention efforts on first-year customers — highest churn risk.")
    lines.append("5. Bundle premium services for high-monthly-charge customers to increase stickiness.")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"[OK] Report saved to {output_path}")


if __name__ == "__main__":
    df = pd.read_csv("data/churn.csv")
    save_charts(df)
    generate_report(df)
