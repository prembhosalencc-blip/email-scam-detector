import streamlit as st
import joblib
import pandas as pd
import altair as alt

model = joblib.load("spam_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

st.set_page_config(page_title="📧 Email Scam Detector", layout="wide")

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

        if prediction == 1:
            st.markdown(f"<h2 style='color:red; text-align:center;'>🚨 Spam / Scam</h2>", unsafe_allow_html=True)
        else:
            st.markdown(f"<h2 style='color:green; text-align:center;'>✅ Safe / Not Spam</h2>", unsafe_allow_html=True)

        st.markdown(f"<p style='text-align:center;'>**Probability:** Safe: {safe_prob:.2f}% | Spam: {spam_prob:.2f}%</p>", unsafe_allow_html=True)

        st.progress(int(spam_prob) if prediction == 1 else int(safe_prob))

        prob_df = pd.DataFrame({
            'Category': ['Safe', 'Spam'],
            'Probability': [safe_prob, spam_prob]
        })

        col1, col2 = st.columns(2)

        with col1:
            pie_chart = alt.Chart(prob_df).mark_arc(innerRadius=50).encode(
                theta=alt.Theta(field="Probability", type="quantitative"),
                color=alt.Color(field="Category", type="nominal"),
                tooltip=['Category', 'Probability']
            )
            st.altair_chart(pie_chart, use_container_width=True)

        with col2:
            bar_chart = alt.Chart(prob_df).mark_bar().encode(
                x=alt.X('Probability', title='Probability (%)'),
                y=alt.Y('Category', sort='-x'),
                color=alt.Color('Category', scale=alt.Scale(range=['green','red'])),
                tooltip=['Category', 'Probability']
            )
            st.altair_chart(bar_chart, use_container_width=True)

    else:
        st.warning("⚠️ Please enter some text to analyze.")
