"""Streamlit demo: estimate stroke risk with the model trained in notebooks/stroke_prediction.ipynb.

Run:  streamlit run app.py
Educational demo only - NOT a medical device and not for clinical use.
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "models" / "stroke_model.joblib"
CARD_PATH = ROOT / "models" / "model_card.json"

st.set_page_config(page_title="Stroke risk demo", page_icon="🧠", layout="centered")


@st.cache_resource
def load_artifacts():
    return joblib.load(MODEL_PATH), json.loads(CARD_PATH.read_text())


if not MODEL_PATH.exists():
    st.error("Model not found. Run notebooks/stroke_prediction.ipynb first to train and save it.")
    st.stop()

model, card = load_artifacts()
threshold = card["threshold"]
prevalence = card["training_prevalence"]
metrics = card["test_metrics"]

st.title("🧠 Stroke risk estimator")
st.warning("Educational demo trained on a public Kaggle dataset. **Not a medical device** - do not use for health decisions.")

with st.form("patient"):
    c1, c2 = st.columns(2)
    age = c1.number_input("Age (years)", 0.0, 100.0, 60.0, 1.0)
    gender = c2.selectbox("Gender", ["Female", "Male"])
    glucose = c1.number_input("Average glucose level (mg/dL)", 50.0, 300.0, 105.0, 1.0)
    bmi_unknown = c2.checkbox("BMI unknown")
    bmi = c2.number_input("BMI (kg/m²)", 10.0, 100.0, 28.0, 0.1, disabled=bmi_unknown)
    hypertension = c1.selectbox("Hypertension", ["No", "Yes"])
    heart_disease = c2.selectbox("Heart disease", ["No", "Yes"])
    ever_married = c1.selectbox("Ever married", ["Yes", "No"])
    work_type = c2.selectbox("Work type", ["Private", "Self-employed", "Govt_job", "children", "Never_worked"])
    residence = c1.selectbox("Residence", ["Urban", "Rural"])
    smoking = c2.selectbox("Smoking status", ["never smoked", "formerly smoked", "smokes", "Unknown"])
    submitted = st.form_submit_button("Estimate risk")

if submitted:
    row = pd.DataFrame([{
        "age": age, "avg_glucose_level": glucose, "bmi": np.nan if bmi_unknown else bmi,
        "gender": gender, "ever_married": ever_married, "work_type": work_type,
        "Residence_type": residence, "smoking_status": smoking,
        "hypertension": int(hypertension == "Yes"), "heart_disease": int(heart_disease == "Yes"),
    }])[card["features"]]
    risk = float(model.predict_proba(row)[0, 1])
    flagged = risk >= threshold

    st.metric("Estimated stroke risk", f"{risk:.1%}", delta=f"{risk / prevalence:.1f}x the dataset average ({prevalence:.1%})", delta_color="off")
    if flagged:
        st.error(f"Above the screening threshold ({threshold:.1%}): this profile would be **flagged** for follow-up.")
    else:
        st.success(f"Below the screening threshold ({threshold:.1%}): not flagged.")

    st.caption(
        f"The threshold was chosen to catch about {metrics['Recall (sensitivity)']['estimate']:.0%} of strokes in held-out data, "
        f"at the cost of a low precision ({metrics['Precision (PPV)']['estimate']:.0%}) - most flagged patients do **not** have a stroke."
    )

with st.expander("About this model"):
    st.write(
        f"Model: **{card['model_family']}** with sigmoid calibration. Held-out test ROC-AUC "
        f"{metrics['ROC-AUC']['estimate']:.2f} (95% CI {metrics['ROC-AUC']['ci95'][0]:.2f}-{metrics['ROC-AUC']['ci95'][1]:.2f}), "
        f"PR-AUC {metrics['PR-AUC']['estimate']:.2f}. Trained on {card['n_train']:,} patients with only ~5% stroke prevalence. "
        "Associations are observational, and the data comes from a single public source with unverified provenance."
    )
