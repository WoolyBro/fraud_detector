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

The model learns from thousands of past transactions to guess whether a new one looks fraudulent. On test data it's **94.5% accurate**, though it also raises some false alarms on legitimate transactions — it's tuned to avoid missing fraud rather than avoiding false alarms.

Since it can miss fraud that doesn't match patterns it learned (e.g. a transaction type it rarely saw as fraudulent before), the app also runs a simple **balance consistency check**: it verifies the sender's balance actually changed by the stated `amount`. A transaction is flagged if *either* the model or this check considers it suspicious.

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
