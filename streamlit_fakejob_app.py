import streamlit as st
import joblib
import re

MODEL_PATH = "fakejob_pipeline.joblib"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def decide_label(fake_prob):
    if fake_prob >= 0.6:
        return "Fake Job"
    elif fake_prob <= 0.4:
        return "Real Job"
    else:
        return "Unsure"

st.set_page_config(page_title="Fake Job Detector", layout="centered")

st.title("Fake Job Posting Detection")
st.write("Enter job details to check if the posting is fake or real")

title = st.text_input("Job Title")
company_profile = st.text_area("Company Profile")
description = st.text_area("Job Description")
requirements = st.text_area("Requirements")
benefits = st.text_area("Benefits (Optional)")

if st.button("Predict"):

    required_fields = {
        "Job Title": title,
        "Company Profile": company_profile,
        "Job Description": description,
        "Requirements": requirements
    }

    missing_fields = [
        name for name, value in required_fields.items()
        if not value.strip()
    ]

    st.markdown("###Prediction Result")

    if missing_fields:
        st.error("**FAKE JOB POSTING**")
        st.caption(
            "Reason: Missing required fields → "
            + ", ".join(missing_fields)
        )
    else:
        combined_text = " ".join([
            title,
            company_profile,
            description,
            requirements,
            benefits
        ])

        cleaned_text = clean_text(combined_text)

        fake_prob = model.predict_proba([cleaned_text])[0][1]
        result = decide_label(fake_prob)

        if result == "Fake Job":
            st.error("**FAKE JOB POSTING**")
        elif result == "Real Job":
            st.success("**REAL JOB POSTING**")
        else:
            st.warning("**UNSURE — NEEDS MANUAL REVIEW**")

        st.caption(
            "Predictions are probability-based. "
            "Borderline cases are marked as UNSURE to reduce false accusations."
        )

