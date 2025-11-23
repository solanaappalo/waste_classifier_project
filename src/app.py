"""
Streamlit App for Waste Classifier
Run: streamlit run src/app.py
"""

import streamlit as st
from pathlib import Path
import joblib
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
MODELS = BASE / "models"
DATA = BASE / "data"

@st.cache_resource
def load_models():
    pipeline = joblib.load(MODELS / "pipeline_full.joblib")
    return pipeline

st.set_page_config(page_title="EcoSortify Waste Classifier", layout="centered")

st.markdown(
    """
    <style>
    /* Main app background */
    .stApp {
        background-color: #e6ece8; 
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("♻️ EcoSortify - A Waste Classifier ")
st.markdown("Type a waste description and the model will predict which bin/category it belongs to.")


pipeline = load_models()

st.text_area("Enter waste description:", key="input_text", height=120)
if st.button("Predict"):
    text = st.session_state.input_text.strip()
    if not text:
        st.warning("Please enter a description.")
    else:
        pred = pipeline.predict([text])[0]
        probs = pipeline.predict_proba([text])[0]
        labels = pipeline.classes_
        # show results
        st.success(f"Predicted category: **{pred}**")
        
        st.info("Tip: Try short natural descriptions like 'one broken glass bottle from kitchen' or 'old laptop battery'.")

st.markdown("""
<hr>
<div style='text-align:center; font-size:12px; color:#2e7d32;'>
<b>Done by Solana – Solana Appalo A M 🌿</b>
</div>
""", unsafe_allow_html=True)
