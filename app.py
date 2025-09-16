import streamlit as st
import joblib
import pandas as pd
import altair as alt

st.set_page_config(
    page_title="📧 Email Scam Detector",
    page_icon="📧",
    layout="wide"
)

model = joblib.load("spam_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

st.markdown("<h1 style='text-align:center; color:#4B0082;'>📧 Email Scam Detector Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; font-size:18px;'>Enter email text in the sidebar to detect Spam/Not Spam with probabilities.</p>", unsafe_allow_html=True)

email_text = st.sidebar.text_area("Enter Email Text Here")
st.sidebar.markdown("### Click below to analyze your email")
analyze_btn = st.sidebar.button("Analyze")

if analyze_btn:
    if email_text.strip():
        X = vectorizer.transform([email_text])
        prediction = model.predict(X)[0]
        probability = model.predict_proba(X)[0]

        safe_prob = probability[0] * 100
        spam_prob = probability[1] * 100

        col1, col2, col3 = st.columns([1,2,1])

        with col1:
            st.markdown("### Probabilities")
            st.markdown(f"<div style='background-color:#d4edda; padding:15px; border-radius:10px; text-align:center;'>✅ Safe<br><h2 style='color:green;'>{safe_prob:.2f}%</h2></div>", unsafe_allow_html=True)
            st.markdown(f"<div style='background-color:#f8d7da; padding:15px; border-radius:10px; text-align:center;'>🚨 Spam<br><h2 style='color:red;'>{spam_prob:.2f}%</h2></div>", unsafe_allow_html=True)

        with col2:
            if prediction == 1:
                st.markdown(f"<h2 style='color:red; text-align:center; font-size:32px;'>🚨 Spam / Scam</h2>", unsafe_allow_html=True)
            else:
                st.markdown(f"<h2 style='color:green; text-align:center; font-size:32px;'>✅ Safe / Not Spam</h2>", unsafe_allow_html=True)

            st.markdown("---")
            st.markdown("### Probability Bars")
            st.progress(int(safe_prob) if prediction == 0 else int(spam_prob))

        with col3:
            prob_df = pd.DataFrame({
                "Category": ["Safe", "Spam"],
                "Probability": [safe_prob, spam_prob]
            })

            st.markdown("### Visual Charts")

            pie_chart = alt.Chart(prob_df).mark_arc(innerRadius=50).encode(
                theta="Probability:Q",
                color=alt.Color("Category:N", scale=alt.Scale(range=["green","red"])),
                tooltip=["Category","Probability"]
            )
            st.altair_chart(pie_chart, use_container_width=True)

            bar_chart = alt.Chart(prob_df).mark_bar().encode(
                x=alt.X("Probability:Q", title="Probability (%)"),
                y=alt.Y("Category:N", sort="-x"),
                color=alt.Color("Category:N", scale=alt.Scale(range=["green","red"])),
                tooltip=["Category","Probability"]
            )
            st.altair_chart(bar_chart, use_container_width=True)

    else:
        st.sidebar.warning("⚠️ Please enter some text to analyze.")
