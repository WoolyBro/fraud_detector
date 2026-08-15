# Fraud Detector

A simple machine learning project that detects potentially fraudulent financial transactions, with a Streamlit app to test it interactively.

## What's in here

- **`analysis_data.ipynb`** — exploratory data analysis and model training
- **`fraud_detect.py`** — Streamlit app for testing transactions
- **`fraud_detection.pkl`** — the trained model
- **`AIML Dataset.csv`** — the transaction dataset (tracked via Git LFS, ~470MB)

## The dataset

Each row is a simulated financial transaction with these fields:

| Column | Meaning |
|---|---|
| `type` | Transaction type: `PAYMENT`, `TRANSFER`, `CASH_OUT`, `CASH_IN`, `DEBIT` |
| `amount` | Transaction amount |
| `oldbalanceOrg` / `newbalanceOrig` | Sender's balance before / after |
| `oldbalanceDest` / `newbalanceDest` | Receiver's balance before / after |
| `isFraud` | 1 if the transaction was fraudulent |

## The model

A Logistic Regression pipeline (`StandardScaler` + `OneHotEncoder`) trained on the numeric balance fields and transaction type. On held-out test data it catches **94% of actual fraud cases** (recall), though it flags some legitimate transactions too (low precision) — it favors not missing fraud over avoiding false alarms.

Since the model can miss fraud that doesn't fit patterns it learned (e.g. a transaction type it rarely saw as fraudulent), the app also runs a simple **balance consistency check**: it verifies the sender's balance actually changed by the stated `amount`. A transaction is flagged if *either* the model or this check considers it suspicious.

## Running it

Install the dependencies:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn streamlit joblib
```

Launch the app:

```bash
python -m streamlit run fraud_detect.py
```

Then open the local URL it prints, fill in a transaction's details, and click **Prediction**.

## Notes

- The dataset only contains real fraud examples for `TRANSFER` and `CASH_OUT` types — the model leans heavily on transaction type as a result.
- `git lfs` is required to pull the dataset file (`git lfs install` once per machine, then a normal `git clone`).
