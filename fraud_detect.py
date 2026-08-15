import streamlit as st 
import pandas as pd
import joblib 

model = joblib.load("fraud_detection.pkl")

st.title("Fraud Detector Model")
st.markdown("Please enter the transaction detils and use the predictor")
st.divider()
transaction_type = st.selectbox("Transaction Type", ["PAYMENT", "TRANSFER", "CASH_OUT", "DEPOSIT"])
amount = st.number_input("Amount", min_value=0, value=1000)
oldbalanceOrg = st.number_input("Old Balance (Sender)", min_value = 0.0, value = 10000.0)
newbalanceOrig = st.number_input("New Balance (Sender)", min_value=0, value = 9000)
oldbalanceDest = st.number_input("Old Balance (Receiver)", min_value=0, value = 0)
newbalanceDest = st.number_input("New Balance (Receiver)", min_value=0, value = 0)

if st.button("Prediction"):
    input_data = pd.DataFrame([{
        "type" : transaction_type,
        "amount" : amount,
        "oldbalanceOrg" : oldbalanceOrg,
        "newbalanceOrig" : newbalanceOrig,
        "oldbalanceDest" : oldbalanceDest,
        "newbalanceDest" : newbalanceDest
    }])
    prediction = model.predict(input_data)[0]

    # Rule-based sanity check: does the sender's balance actually move by `amount`?
    # Legitimate transactions satisfy oldbalanceOrg - newbalanceOrig == amount;
    # a large mismatch is a strong tampering/fraud signal the model alone can miss.
    tolerance = 1.0
    orig_diff = oldbalanceOrg - newbalanceOrig
    orig_mismatch = abs(orig_diff - amount) > tolerance
    drained_beyond_amount = newbalanceOrig == 0 and oldbalanceOrg > amount + tolerance

    rule_flagged = orig_mismatch or drained_beyond_amount
    final_flag = bool(prediction == 1) or rule_flagged

    st.subheader(f"Model Prediction : '{int(prediction)}'")

    st.markdown("**Balance consistency check**")
    st.write(f"Sender balance change: {orig_diff:,.2f} (expected ≈ amount: {amount:,.2f})")
    if rule_flagged:
        st.warning(f"Balance mismatch of {abs(orig_diff - amount):,.2f} detected — sender's balance moved more than the stated amount.")
    else:
        st.write("Sender balance change matches the stated amount.")

    st.divider()
    if final_flag:
        st.error("Final verdict: This transaction is FLAGGED as fraud" + (" (balance mismatch)" if rule_flagged and prediction != 1 else ""))
    else:
        st.success("Final verdict: This transaction looks legal")