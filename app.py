from pathlib import Path
import pickle

import numpy as np
import streamlit as st


FEATURES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal",
]

MODEL_CANDIDATES = [
    Path("heart_model.pkl"),
    Path("model.pkl"),
    Path("heart_disease_model.pkl"),
    Path("models/heart_model.pkl"),
]


def inject_css():
    st.markdown(
        """
        <style>
        .main-title {
            font-size: 3rem;
            font-weight: 800;
            line-height: 1.1;
            margin-bottom: 0.25rem;
        }
        .subtitle {
            color: #64748b;
            font-size: 1.05rem;
            margin-bottom: 1.5rem;
        }
        .hero {
            padding: 1.5rem 1.75rem;
            border-radius: 22px;
            background: linear-gradient(135deg, #fff1f2 0%, #ffffff 55%, #eff6ff 100%);
            border: 1px solid #e2e8f0;
            margin-bottom: 1.5rem;
        }
        .section-card {
            padding: 1rem 1.2rem;
            border-radius: 16px;
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            margin: 0.5rem 0 1rem;
        }
        .result-card {
            padding: 1.4rem;
            border-radius: 20px;
            border: 1px solid #e2e8f0;
            background: white;
            box-shadow: 0 8px 28px rgba(15, 23, 42, 0.08);
        }
        .risk-high {
            color: #b91c1c;
            font-size: 1.7rem;
            font-weight: 800;
        }
        .risk-low {
            color: #15803d;
            font-size: 1.7rem;
            font-weight: 800;
        }
        .small-muted {
            color: #64748b;
            font-size: 0.9rem;
        }
        div.stButton > button,
        div.stFormSubmitButton > button {
            border-radius: 12px;
            font-weight: 700;
            min-height: 3rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def load_model():
    """Load a saved pickle model if one is present."""
    for model_path in MODEL_CANDIDATES:
        if model_path.exists():
            try:
                with open(model_path, "rb") as file:
                    return pickle.load(file), model_path.name
            except Exception as exc:
                st.warning(f"Could not load {model_path.name}: {exc}")
    return None, None


def clinical_risk_probability(patient):
    """Fallback estimator used only when no trained model is available."""
    score = 0.0

    if patient["age"] >= 60:
        score += 2.0
    elif patient["age"] >= 45:
        score += 1.0

    if patient["sex"] == 1:
        score += 1.0

    if patient["cp"] in (1, 2):
        score += 1.5
    elif patient["cp"] == 3:
        score += 2.0

    if patient["trestbps"] > 140:
        score += 1.5
    if patient["chol"] > 200:
        score += 1.0
    if patient["fbs"] == 1:
        score += 1.0

    if patient["restecg"] == 1:
        score += 0.8
    elif patient["restecg"] == 2:
        score += 1.2

    if patient["thalach"] < 120:
        score += 1.5
    elif patient["thalach"] < 150:
        score += 0.5

    if patient["exang"] == 1:
        score += 2.0

    if patient["oldpeak"] > 2.0:
        score += 2.0
    elif patient["oldpeak"] > 1.0:
        score += 1.0

    if patient["slope"] in (1, 2):
        score += 1.0
    if patient["ca"] > 0:
        score += 2.0
    if patient["thal"] in (2, 3):
        score += 1.5
    elif patient["thal"] == 1:
        score += 0.5

    z = (score - 7.0) / 2.8
    return float(1 / (1 + np.exp(-z)))


def predict_heart_disease(patient, model=None):
    """Return prediction and probability."""
    if model is not None:
        values = np.array([patient[name] for name in FEATURES], dtype=float).reshape(1, -1)
        result = int(model.predict(values)[0])
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(values)[0]
            probability = float(probabilities[1]) if len(probabilities) > 1 else float(probabilities[0])
        else:
            probability = float(result)
        return result, float(np.clip(probability, 0.0, 1.0))

    probability = clinical_risk_probability(patient)
    return int(probability >= 0.5), probability


def prepare_patient_for_model(patient):
    prepared = patient.copy()
    prepared["sex"] = 1 if prepared["sex"] == "Male" else 0
    prepared["fbs"] = 1 if prepared["fbs"] == "Yes" else 0
    prepared["exang"] = 1 if prepared["exang"] == "Yes" else 0
    return prepared


def build_patient_data():
    st.markdown("### 🧑‍⚕️ Patient Information")
    st.caption("Enter the patient's clinical measurements and ECG-related information.")

    with st.container(border=True):
        st.markdown("**Basic information**")
        col1, col2, col3 = st.columns(3)
        with col1:
            age = st.number_input("Age", 18, 120, 52, 1)
        with col2:
            sex = st.selectbox("Sex", ["Male", "Female"])
        with col3:
            cp = st.selectbox(
                "Chest Pain Type",
                [(0, "Typical angina"), (1, "Atypical angina"), (2, "Non-anginal pain"), (3, "Asymptomatic")],
                format_func=lambda item: item[1],
                index=2,
            )[0]

    with st.container(border=True):
        st.markdown("**Vital signs & blood tests**")
        col1, col2, col3 = st.columns(3)
        with col1:
            trestbps = st.number_input("Resting BP (mmHg)", 80, 220, 130, 1)
        with col2:
            chol = st.number_input("Cholesterol (mg/dL)", 100, 500, 210, 1)
        with col3:
            fbs = st.selectbox("Fasting Blood Sugar > 120", ["No", "Yes"])

    with st.container(border=True):
        st.markdown("**ECG & exercise information**")
        col1, col2, col3 = st.columns(3)
        with col1:
            restecg = st.selectbox(
                "Resting ECG",
                [(0, "Normal"), (1, "ST-T wave abnormality"), (2, "Left ventricular hypertrophy")],
                format_func=lambda item: item[1],
            )[0]
        with col2:
            thalach = st.number_input("Maximum Heart Rate", 50, 220, 150, 1)
        with col3:
            exang = st.selectbox("Exercise-induced Angina", ["No", "Yes"])

    with st.container(border=True):
        st.markdown("**Additional clinical indicators**")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            oldpeak = st.number_input("Oldpeak (ST depression)", 0.0, 10.0, 1.5, 0.1)
        with col2:
            slope = st.selectbox(
                "ST Segment Slope",
                [(0, "Upsloping"), (1, "Flat"), (2, "Downsloping")],
                format_func=lambda item: item[1],
                index=1,
            )[0]
        with col3:
            ca = st.number_input("Major Vessels (0–4)", 0, 4, 0, 1)
        with col4:
            thal = st.selectbox(
                "Thalassemia",
                [(0, "Normal"), (1, "Fixed defect"), (2, "Reversible defect"), (3, "Unknown")],
                format_func=lambda item: item[1],
                index=2,
            )[0]

    return {
        "age": age,
        "sex": sex,
        "cp": cp,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs,
        "restecg": restecg,
        "thalach": thalach,
        "exang": exang,
        "oldpeak": oldpeak,
        "slope": slope,
        "ca": ca,
        "thal": thal,
    }


def show_result(prediction, probability, model_name):
    risk_percent = probability * 100
    confidence = max(probability, 1 - probability) * 100

    st.markdown("## 📊 Prediction Result")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Risk Level", "HIGH ⚠️" if prediction else "LOW ✅")
    with col2:
        st.metric("Estimated Risk", f"{risk_percent:.1f}%")
    with col3:
        st.metric("Model Confidence", f"{confidence:.1f}%")

    st.markdown("### Risk Probability")
    st.progress(float(np.clip(probability, 0.0, 1.0)), text=f"Estimated probability: {risk_percent:.1f}%")

    if prediction:
        st.error("🚨 The model indicates an elevated likelihood of heart disease. Please consult a qualified healthcare professional for proper evaluation.")
    else:
        st.success("✅ The model indicates a lower likelihood of heart disease based on the supplied inputs.")

    if model_name:
        st.caption(f"Prediction generated using: `{model_name}`")
    else:
        st.warning("No trained pickle model was found. This result uses the app's fallback scoring estimator and should not be treated as a medical diagnosis.")

    st.info("ℹ️ This application is for educational and demonstration purposes only. It does not replace professional medical advice, diagnosis, or treatment.")


def main():
    st.set_page_config(
        page_title="Heart Disease Prediction",
        page_icon="❤️",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_css()

    with st.sidebar:
        st.markdown("## ❤️ Heart Health")
        st.markdown("**Heart Disease Prediction**")
        st.divider()
        st.markdown("### About")
        st.write("Enter patient measurements to estimate the probability of heart disease using the configured prediction model.")
        st.markdown("### Features")
        st.write("Age, sex, chest pain, blood pressure, cholesterol, ECG, heart rate, exercise angina, ST depression, vessels and thalassemia.")
        st.divider()
        st.caption("Educational tool • Not a medical diagnosis")

    st.markdown(
        """
        <div class="hero">
            <div class="main-title">❤️ Heart Disease Prediction</div>
            <div class="subtitle">A clean, simple interface for estimating heart disease risk from clinical inputs.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    model, model_name = load_model()
    if model_name:
        st.success(f"Model ready: {model_name}")
    else:
        st.info("Running in fallback mode because no saved model file was found.")

    with st.form("heart_prediction_form"):
        patient = build_patient_data()
        st.markdown("###")
        submitted = st.form_submit_button("🔍  Predict Heart Disease Risk", use_container_width=True, type="primary")

    if submitted:
        with st.spinner("Analyzing patient data..."):
            prepared = prepare_patient_for_model(patient)
            prediction, probability = predict_heart_disease(prepared, model=model)
        st.divider()
        show_result(prediction, probability, model_name)


if __name__ == "__main__":
    main()
