import streamlit as st
import joblib

model = joblib.load("spam_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

st.title("Email Scam Detector")

email_text = st.text_area("Enter email text")

if st.button("Check"):
    if email_text.strip():
        X = vectorizer.transform([email_text])
        prediction = model.predict(X)[0]
        result = "🚨 Scam / Spam" if prediction == 1 else "✅ Safe / Not Spam"
        st.write(result)
    else:
        st.write("Please enter some text")

