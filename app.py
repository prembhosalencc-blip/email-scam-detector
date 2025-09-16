import streamlit as st
import joblib
import numpy as np
import pandas as pd
import altair as alt

model = joblib.load("spam_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

st.set_page_config(page_title="📧 Email Scam Detector", layout="centered")

st.title("📧 Email Scam Detector")
st.markdown("Detect whether an email is **Spam/Scam** or **Safe** using Machine Learning!")

email_text = st.text_area("Enter the email text below:")

if st.button("Check"):
    if email_text.strip():
        X = vectorizer.transform([email_text])
        prediction = model.predict(X)[0]
        probability = model.predict_proba(X)[0]
        
        spam_prob = probability[1] * 100
        safe_prob = probability[0] * 100

        result = "🚨 Spam / Scam" if prediction == 1 else "✅ Safe / Not Spam"
        st.markdown(f"### Prediction: {result}")
        st.markdown(f"**Probability:** Safe: {safe_prob:.2f}% | Spam: {spam_prob:.2f}%")

        # Progress bars for probabilities
        st.progress(int(safe_prob) if prediction==0 else int(spam_prob))

        # Pie chart
        prob_df = pd.DataFrame({
            'Category': ['Safe', 'Spam'],
            'Probability': [safe_prob, spam_prob]
        })

        chart = alt.Chart(prob_df).mark_arc(innerRadius=50).encode(
            theta=alt.Theta(field="Probability", type="quantitative"),
            color=alt.Color(field="Category", type="nominal"),
            tooltip=['Category', 'Probability']
        )

        st.altair_chart(chart, use_container_width=True)

    else:
        st.warning("⚠️ Please enter some text to analyze.")

