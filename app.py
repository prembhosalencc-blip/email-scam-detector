import streamlit as st
import joblib
import pandas as pd
import altair as alt

st.set_page_config(page_title="📧 Email Scam Detector", layout="wide")

model = joblib.load("spam_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

st.markdown("<h1 style='text-align:center; color:#4B0082;'>📧 Email Scam Detector Application</h1>", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["🔍 Prediction", "📊 Dashboard"])

with tab1:
    st.subheader("Email Scam Prediction")
    email_text = st.text_area("Enter the email text below:")
    if st.button("Analyze", key="predict_btn"):
        if email_text.strip():
            X = vectorizer.transform([email_text])
            prediction = model.predict(X)[0]
            probability = model.predict_proba(X)[0]
            safe_prob = probability[0] * 100
            spam_prob = probability[1] * 100
            if prediction == 1:
                st.markdown(f"<h2 style='color:red; text-align:center;'>🚨 Spam / Scam</h2>", unsafe_allow_html=True)
            else:
                st.markdown(f"<h2 style='color:green; text-align:center;'>✅ Safe / Not Spam</h2>", unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            col1.metric("✅ Safe Probability", f"{safe_prob:.2f}%")
            col2.metric("🚨 Spam Probability", f"{spam_prob:.2f}%")
        else:
            st.warning("⚠️ Please enter some text to analyze.")

with tab2:
    st.subheader("Email Scam Detector Dashboard")
    data = {
        "Emails Checked": 150,
        "Spam Detected": 65,
        "Safe Emails": 85,
        "Accuracy": "96.6%"
    }
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("📧 Emails Checked", data["Emails Checked"])
    col2.metric("🚨 Spam Detected", data["Spam Detected"])
    col3.metric("✅ Safe Emails", data["Safe Emails"])
    col4.metric("🎯 Model Accuracy", data["Accuracy"])
    st.markdown("---")
    st.subheader("Spam vs Safe Distribution")
    df = pd.DataFrame({"Category": ["Spam", "Safe"], "Count": [data["Spam Detected"], data["Safe Emails"]]})
    pie_chart = alt.Chart(df).mark_arc(innerRadius=50).encode(
        theta="Count:Q",
        color=alt.Color("Category:N", scale=alt.Scale(range=["red","green"])),
        tooltip=["Category", "Count"]
    )
    st.altair_chart(pie_chart, use_container_width=True)
    st.markdown("---")
    st.subheader("Trend of Emails Analyzed (Sample Data)")
    trend_df = pd.DataFrame({
        "Day": ["Mon", "Tue", "Wed", "Thu", "Fri"],
        "Spam": [5, 12, 8, 15, 25],
        "Safe": [20, 18, 22, 25, 30]
    })
    trend_chart = alt.Chart(trend_df).transform_fold(
        ["Spam", "Safe"],
        as_=["Category", "Count"]
    ).mark_line(point=True).encode(
        x="Day:N",
        y="Count:Q",
        color="Category:N"
    )
    st.altair_chart(trend_chart, use_container_width=True)
