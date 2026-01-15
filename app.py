import streamlit as st

st.set_page_config(page_title="Agile Heart Disease Project", layout="wide")

st.title("❤️ Agile Heart Disease Prediction System")

st.markdown("""
### Navigation
Use the sidebar to access:

- **Prediction (UI V2)** – Compare Model V1 vs Model V2
- **Old UI (UI V1)** – First iteration (for Agile evidence)
- **Monitoring Dashboard** – Latency, feedback, logs, charts

This application demonstrates:
- Model iteration (Logistic Regression → Random Forest)
- UI iteration based on user feedback
- Monitoring and MLOps principles
""")
