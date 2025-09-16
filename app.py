import streamlit as st
import joblib
import pandas as pd
import altair as alt

st.set_page_config(
    page_title="📧 Email Scam Detector",
    page_icon="📧",
    layout="wide",
)

model = joblib.load("spam_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

st.title("📧 Email Scam Detector Dashboard")
st.markdown("Enter email text in the sidebar to detect whether it is **Spam/Scam** or **Safe**.")

email_text = st.sidebar.text_area("Enter Email Text Here")

if st.sidebar.button("Analyze"):
    if email_text.strip():
        X = vectorizer.transform([email_text])
        prediction = model.predict(X)[0]
        probability = model.predict_proba(X)[0]

        safe_prob = probability[0] * 100
        spam_prob = probability[1] * 100

        col1, col2, col3 = st.columns([1,2,1])

        with col1:
            st.markdown("### Probabilities")
            st.metric(label="Safe", value=f"{safe_prob:.2f}%")
            st.metric(label="Spam", value=f"{spam_prob:.2f}%")

        with col2:
            st.markdown("### Prediction")
            if prediction == 1:
                st.markdown(f"<h2 style='color:red; text-align:center;'>🚨 Spam / Scam</h2>", unsafe_allow_html=True)
            else:
                st.markdown(f"<h2 style='color:green; text-align:center;'>✅ Safe / Not Spam</h2>", unsafe_allow_html=True)

        with col3:
            st.markdown("### Probability Charts")
            prob_df = pd.DataFrame({
                "Category": ["Safe", "Spam"],
                "Probability": [safe_prob, spam_prob]
            })

            pie_chart = alt.Chart(prob_df).mark_arc(innerRadius=50).encode(
                theta="Probability:Q",
                color=alt.Color("Category:N", scale=alt.Scale(range=["green", "red"])),
                tooltip=["Category", "Probability"]
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

