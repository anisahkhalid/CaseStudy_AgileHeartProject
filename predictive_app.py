import time
import pandas as pd
import streamlit as st
from joblib import load

from log_utils import log_prediction

st.set_page_config(page_title="Heart Disease Prediction (V1 vs V2)", layout="centered")

@st.cache_resource
def load_models():
    model_v1 = load("models/model_v1.pkl")
    model_v2 = load("models/model_v2.pkl")
    return model_v1, model_v2

def make_input_df(inputs: dict) -> pd.DataFrame:
    return pd.DataFrame([inputs])

st.title("❤️ Heart Disease Prediction App")
st.write("Compare predictions from **Model V1 (Logistic Regression)** and **Model V2 (Random Forest)**.")

model_v1, model_v2 = load_models()

st.subheader("Patient Inputs")

# --- Inputs (keys must match training columns) ---
age = st.number_input("Age", min_value=1, max_value=120, value=45)
dataset = st.text_input("Dataset / place of study", value="Cleveland")
sex = st.selectbox("Sex", ["Male", "Female"])
cp = st.selectbox("Chest Pain Type (cp)", ["typical angina", "atypical angina", "non-anginal", "asymptomatic"])

trestbps = st.number_input("Resting Blood Pressure (trestbps)", min_value=50, max_value=250, value=130)
chol = st.number_input("Serum Cholesterol (chol)", min_value=50, max_value=700, value=200)
fbs = st.selectbox("Fasting Blood Sugar > 120 (fbs)", [0, 1])

restecg = st.selectbox("Resting ECG (restecg)", ["normal", "stt abnormality", "lv hypertrophy"])
thalch = st.number_input("Max Heart Rate Achieved (thalch)", min_value=50, max_value=250, value=150)
exang = st.selectbox("Exercise-induced Angina (exang)", [0, 1])

oldpeak = st.number_input("Oldpeak (ST depression)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
slope = st.number_input("Slope", min_value=0, max_value=3, value=1)
ca = st.number_input("Number of major vessels (ca)", min_value=0, max_value=3, value=0)
thal = st.selectbox("Thal", ["normal", "fixed defect", "reversible defect"])

inputs = {
    "age": age,
    "dataset": dataset,
    "sex": sex,
    "cp": cp,
    "trestbps": trestbps,
    "chol": chol,
    "fbs": fbs,
    "restecg": restecg,
    "thalch": thalch,
    "exang": exang,
    "oldpeak": oldpeak,
    "slope": slope,
    "ca": ca,
    "thal": thal,
}
input_df = make_input_df(inputs)

st.divider()
st.subheader("Step 1: Generate Prediction")

# --- Prediction step ---
if st.button("Predict with both models"):
    # Model V1
    start = time.time()
    pred_v1 = int(model_v1.predict(input_df)[0])
    prob_v1 = float(model_v1.predict_proba(input_df)[0][1])
    latency_v1 = (time.time() - start) * 1000

    # Model V2
    start = time.time()
    pred_v2 = int(model_v2.predict(input_df)[0])
    prob_v2 = float(model_v2.predict_proba(input_df)[0][1])
    latency_v2 = (time.time() - start) * 1000

    # Store results in session_state (so they persist)
    st.session_state["results"] = {
        "pred_v1": pred_v1,
        "prob_v1": prob_v1,
        "latency_v1": latency_v1,
        "pred_v2": pred_v2,
        "prob_v2": prob_v2,
        "latency_v2": latency_v2,
    }

# --- Display results if available ---
if "results" in st.session_state:
    r = st.session_state["results"]

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Model V1 (Logistic Regression)")
        st.write("Prediction:", "Heart Disease" if r["pred_v1"] == 1 else "No Heart Disease")
        st.write("Probability (disease):", round(r["prob_v1"], 4))
        st.write("Latency (ms):", round(r["latency_v1"], 2))

    with col2:
        st.markdown("### Model V2 (Random Forest)")
        st.write("Prediction:", "Heart Disease" if r["pred_v2"] == 1 else "No Heart Disease")
        st.write("Probability (disease):", round(r["prob_v2"], 4))
        st.write("Latency (ms):", round(r["latency_v2"], 2))

    st.divider()
    st.subheader("Step 2: Submit Feedback & Log")

    feedback_score = st.slider("Feedback Score (1=bad, 5=good)", 1, 5, 4)
    feedback_comment = st.text_area("Feedback Comment (optional)", "")

    if st.button("Submit Feedback & Log"):
        # Log both predictions with the same feedback
        log_prediction("v2_ui", "v1", r["latency_v1"], r["pred_v1"], r["prob_v1"], feedback_score, feedback_comment)
        log_prediction("v2_ui", "v2", r["latency_v2"], r["pred_v2"], r["prob_v2"], feedback_score, feedback_comment)

        st.success("Saved feedback + predictions to logs/monitoring_logs.csv ✅")

        # Optional: clear results so user does a new prediction next time
        # del st.session_state["results"]
else:
    st.info("Click **Predict with both models** to generate results first.")

