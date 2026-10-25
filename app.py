"""Interactive Streamlit demo for the educational Heart Age Engine."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from heart_age_engine.config import MODEL_FEATURES
from heart_age_engine.data import clean_data
from heart_age_engine.explainability import local_contributions
from heart_age_engine.pipeline import train_pipeline

st.set_page_config(page_title="Heart Age Engine", page_icon="❤️", layout="wide")

st.title("❤️ Heart Age Engine")
st.caption("Explainable ML demo — educational use only, not a medical diagnostic tool.")

bundle_path = Path("artifacts/model_bundle.joblib")
bundle = joblib.load(bundle_path) if bundle_path.exists() else None

if bundle is None:
    st.info(
        "No saved model bundle was found. Upload the public dataset below to train the demo "
        "in memory, or run `python scripts/train.py --data ...` locally first."
    )
    uploaded = st.file_uploader(
        "Upload cardio_train.csv or cardio_train.xlsx",
        type=["csv", "xlsx"],
    )
    if uploaded is None:
        st.stop()

    if uploaded.name.lower().endswith(".xlsx"):
        raw = pd.read_excel(uploaded)
    else:
        raw = pd.read_csv(uploaded, sep=";")
        if raw.shape[1] == 1:
            uploaded.seek(0)
            raw = pd.read_csv(uploaded)

    with st.spinner("Cleaning data and training models..."):
        cleaned = clean_data(raw)
        _, bundle = train_pipeline(cleaned)

heart_model = bundle.heart_age_model
cluster_model = bundle.cluster_model

with st.sidebar:
    st.header("Inputs")
    age = st.slider("Chronological age", 30, 70, 45)
    gender = st.radio("Sex code", options=[1, 2], horizontal=True)
    systolic = st.slider("Systolic BP", 80, 200, 130)
    diastolic = st.slider("Diastolic BP", 40, 140, 85)
    bmi = st.slider("BMI", 14.0, 55.0, 25.0, 0.5)
    cholesterol = st.selectbox("Cholesterol category", [1, 2, 3])
    glucose = st.selectbox("Glucose category", [1, 2, 3])
    smoke = int(st.checkbox("Smoker"))
    alcohol = int(st.checkbox("Alcohol use"))
    active = int(st.checkbox("Physically active", value=True))

pulse_pressure = systolic - diastolic
row = pd.DataFrame(
    [
        {
            "gender": gender,
            "bmi": bmi,
            "ap_hi": systolic,
            "ap_lo": diastolic,
            "pulse_pressure": pulse_pressure,
            "cholesterol": cholesterol,
            "gluc": glucose,
            "smoke": smoke,
            "alco": alcohol,
            "active": active,
        }
    ]
)

heart_age = float(heart_model.predict(row)[0])
age_gap = heart_age - age
cluster_row = row.assign(age_gap=age_gap, age_years=age)
cluster_id = int(cluster_model.predict(cluster_row)[0])

c1, c2, c3 = st.columns(3)
c1.metric("Chronological age", f"{age} yrs")
c2.metric("Model-derived heart age", f"{heart_age:.1f} yrs")
c3.metric("Model-derived age gap", f"{age_gap:+.1f} yrs")

st.subheader("Local model explanation")
contrib = local_contributions(heart_model, row)
st.dataframe(contrib, use_container_width=True, hide_index=True)
st.bar_chart(contrib.set_index("feature")["contribution_years"])

st.subheader("Exploratory cluster")
st.write(f"This input maps to **cluster {cluster_id}** under the fitted K-Means model.")

with st.expander("How to interpret this demo"):
    st.markdown(
        """
        - The heart-age value is a **model-derived proxy**, not a biological measurement.
        - SHAP values describe how the fitted model moved its prediction for this input.
        - The K-Means label is an exploratory grouping, not a medical phenotype.
        - Do not use this application for diagnosis or treatment decisions.
        """
    )

st.caption(f"Predictor count: {len(MODEL_FEATURES)} · Model: GradientBoostingRegressor")
