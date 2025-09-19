import streamlit as st
import joblib
import pandas as pd
import altair as alt

st.set_page_config(page_title="📧 Email Scam Detector", layout="wide")

model = joblib.load("spam_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

st.markdown("<h1 style='text-align:center; color:#4B0082;'>📧 Email Scam Detector Dashboard</h1>", unsafe_allow_html=True)

email_text = st.sidebar.text_area("Enter Email Text Here")
section = st.sidebar.selectbox("Select Section to View", ["Prediction", "Probabilities", "Pie Chart", "Bar Graph"])
analyze_btn = st.sidebar.button("Analyze")

if analyze_btn and email_text.strip():
    X = vectorizer.transform([email_text])
    prediction = model.predict(X)[0]
    probability = model.predict_proba(X)[0]

    safe_prob = probability[0] * 100
    spam_prob = probability[1] * 100
    prob_df = pd.DataFrame({
        "Category": ["Safe", "Spam"],
        "Probability": [safe_prob, spam_prob]
    })

    if section == "Prediction":
        st.markdown("## Prediction")
        if prediction == 1:
            st.markdown(f"<h2 style='color:red; text-align:center;'>🚨 Spam / Scam</h2>", unsafe_allow_html=True)
        else:
            st.markdown(f"<h2 style='color:green; text-align:center;'>✅ Safe / Not Spam</h2>", unsafe_allow_html=True)

    elif section == "Probabilities":
        st.markdown("## Probabilities")
        col1, col2 = st.columns(2)
        col1.metric("✅ Safe", f"{safe_prob:.2f}%")
        col2.metric("🚨 Spam", f"{spam_prob:.2f}%")

    elif section == "Pie Chart":
        st.markdown("## Pie Chart")
        pie_chart = alt.Chart(prob_df).mark_arc(innerRadius=50).encode(
            theta="Probability:Q",
            color=alt.Color("Category:N", scale=alt.Scale(range=["green","red"])),
            tooltip=["Category","Probability"]
        )
        st.altair_chart(pie_chart, use_container_width=True)

    elif section == "Bar Graph":
        st.markdown("## Probability Bar Graph")
        bar_chart = alt.Chart(prob_df).mark_bar().encode(
            x=alt.X("Probability:Q", title="Probability (%)"),
            y=alt.Y("Category:N", sort="-x"),
            color=alt.Color("Category:N", scale=alt.Scale(range=["green","red"])),
            tooltip=["Category","Probability"]
        )
        st.altair_chart(bar_chart, use_container_width=True)

elif analyze_btn and not email_text.strip():
    st.sidebar.warning("⚠️ Please enter some text to analyze.") 
