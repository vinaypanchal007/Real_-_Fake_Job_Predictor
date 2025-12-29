import streamlit as st # type: ignore
import joblib
import re

# ---------------- CONFIG ----------------
MODEL_PATH = "fakejob_pipeline.joblib"

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

# ---------------- CLEAN TEXT (SAME AS TRAINING) ----------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ---------------- DECISION LOGIC ----------------
def decide_label(fake_prob):
    if fake_prob >= 0.6:
        return "Fake Job"
    elif fake_prob <= 0.4:
        return "Real Job"
    else:
        return "Unsure"

# ---------------- STREAMLIT UI ----------------
st.set_page_config(page_title="Fake Job Detector", layout="centered")

st.title("Fake Job Posting Detection")
st.write("Enter job details to check if the posting is fake or real")

title = st.text_input("Job Title")
company_profile = st.text_area("Company Profile")
description = st.text_area("Job Description")
requirements = st.text_area("Requirements")
benefits = st.text_area("Benefits")

if st.button("Predict"):
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

    st.markdown("### 🔍 Prediction Result")

    if result == "Fake Job":
        st.error(f"**FAKE JOB POSTING**")
    elif result == "Real Job":
        st.success(f"**REAL JOB POSTING**")
    else:
        st.warning(f"**UNSURE — NEEDS MANUAL REVIEW**")

    st.caption(
        "Predictions are probability-based. Borderline cases are marked as UNSURE "
        "to reduce false accusations."
    )
