from pathlib import Path
import pickle

import numpy as np
import streamlit as st

FEATURES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal",
]

MODEL_CANDIDATES = [
    Path("heart_model.pkl"), Path("model.pkl"),
    Path("heart_disease_model.pkl"), Path("models/heart_model.pkl"),
]


def inject_css():
    st.markdown("""
    <style>
    @keyframes heartbeat { 0%, 100% { transform: scale(1); } 12% { transform: scale(1.08); } 24% { transform: scale(1); } 36% { transform: scale(1.05); } 48% { transform: scale(1); } }
    @keyframes floatIn { from { opacity: 0; transform: translateY(18px); } to { opacity: 1; transform: translateY(0); } }
    @keyframes glow { 0%,100% { box-shadow: 0 0 0 rgba(239,68,68,0); } 50% { box-shadow: 0 0 28px rgba(239,68,68,.18); } }
    @keyframes pulseRing { 0% { transform: scale(.92); opacity: .65; } 100% { transform: scale(1.25); opacity: 0; } }
    @keyframes shimmer { 0% { background-position: -500px 0; } 100% { background-position: 500px 0; } }
    .hero { padding: 2rem; border-radius: 24px; background: linear-gradient(135deg,#fff1f2,#fff,#eff6ff); border: 1px solid #e2e8f0; margin-bottom: 1.5rem; animation: floatIn .65s ease-out, glow 3s ease-in-out infinite; position: relative; overflow: hidden; }
    .heart-icon { display:inline-block; font-size:4rem; animation: heartbeat 1.25s ease-in-out infinite; transform-origin:center; filter: drop-shadow(0 8px 14px rgba(239,68,68,.2)); }
    .main-title { font-size: 3rem; font-weight: 850; line-height:1.05; margin:.2rem 0; }
    .subtitle { color:#64748b; font-size:1.05rem; }
    .section-card { animation: floatIn .5s ease-out both; }
    .result-card { padding:1.5rem; border-radius:22px; border:1px solid #e2e8f0; background:rgba(255,255,255,.96); box-shadow:0 12px 35px rgba(15,23,42,.10); animation:floatIn .65s ease-out; }
    .result-title { font-size:1.8rem; font-weight:800; }
    .pulse-dot { width:12px; height:12px; border-radius:50%; background:#ef4444; display:inline-block; margin-right:8px; position:relative; }
    .pulse-dot:after { content:''; position:absolute; inset:-5px; border:2px solid #ef4444; border-radius:50%; animation:pulseRing 1.4s infinite; }
    div.stButton > button, div.stFormSubmitButton > button { border-radius:14px; font-weight:750; min-height:3.2rem; transition:transform .2s, box-shadow .2s; }
    div.stButton > button:hover, div.stFormSubmitButton > button:hover { transform:translateY(-2px); box-shadow:0 8px 20px rgba(15,23,42,.12); }
    .model-ready { padding:.65rem 1rem; border-radius:12px; background:linear-gradient(90deg,#ecfdf5,#f0fdf4,#ecfdf5); background-size:700px 100%; animation:shimmer 3s linear infinite; border:1px solid #bbf7d0; }
    @media (max-width: 700px) { .main-title {font-size:2.1rem;} .heart-icon {font-size:3rem;} .hero {padding:1.25rem;} }
    </style>
    """, unsafe_allow_html=True)


def play_sound(kind):
    """Play a short browser sound after the user submits the form."""
    urls = {
        "success": "https://assets.mixkit.co/active_storage/sfx/2869/2869-preview.mp3",
        "warning": "https://assets.mixkit.co/active_storage/sfx/2633/2633-preview.mp3",
    }
    url = urls[kind]
    st.markdown(f"""
    <audio autoplay>
      <source src="{url}" type="audio/mpeg">
    </audio>
    """, unsafe_allow_html=True)


def load_model():
    for model_path in MODEL_CANDIDATES:
        if model_path.exists():
            try:
                with open(model_path, "rb") as file:
                    return pickle.load(file), model_path.name
            except Exception as exc:
                st.warning(f"Could not load {model_path.name}: {exc}")
    return None, None


def clinical_risk_probability(patient):
    score = 0.0
    if patient["age"] >= 60: score += 2.0
    elif patient["age"] >= 45: score += 1.0
    if patient["sex"] == 1: score += 1.0
    if patient["cp"] in (1, 2): score += 1.5
    elif patient["cp"] == 3: score += 2.0
    if patient["trestbps"] > 140: score += 1.5
    if patient["chol"] > 200: score += 1.0
    if patient["fbs"] == 1: score += 1.0
    if patient["restecg"] == 1: score += .8
    elif patient["restecg"] == 2: score += 1.2
    if patient["thalach"] < 120: score += 1.5
    elif patient["thalach"] < 150: score += .5
    if patient["exang"] == 1: score += 2.0
    if patient["oldpeak"] > 2.0: score += 2.0
    elif patient["oldpeak"] > 1.0: score += 1.0
    if patient["slope"] in (1, 2): score += 1.0
    if patient["ca"] > 0: score += 2.0
    if patient["thal"] in (2, 3): score += 1.5
    elif patient["thal"] == 1: score += .5
    z = (score - 7.0) / 2.8
    return float(1 / (1 + np.exp(-z)))


def predict_heart_disease(patient, model=None):
    if model is not None:
        values = np.array([patient[name] for name in FEATURES], dtype=float).reshape(1, -1)
        result = int(model.predict(values)[0])
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(values)[0]
            probability = float(probabilities[1]) if len(probabilities) > 1 else float(probabilities[0])
        else:
            probability = float(result)
        return result, float(np.clip(probability, 0, 1))
    probability = clinical_risk_probability(patient)
    return int(probability >= .5), probability


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
        c1,c2,c3 = st.columns(3)
        with c1: age = st.number_input("Age",18,120,52,1)
        with c2: sex = st.selectbox("Sex",["Male","Female"])
        with c3: cp = st.selectbox("Chest Pain Type",[(0,"Typical angina"),(1,"Atypical angina"),(2,"Non-anginal pain"),(3,"Asymptomatic")],format_func=lambda x:x[1],index=2)[0]
    with st.container(border=True):
        st.markdown("**Vital signs & blood tests**")
        c1,c2,c3 = st.columns(3)
        with c1: trestbps = st.number_input("Resting BP (mmHg)",80,220,130,1)
        with c2: chol = st.number_input("Cholesterol (mg/dL)",100,500,210,1)
        with c3: fbs = st.selectbox("Fasting Blood Sugar > 120",["No","Yes"])
    with st.container(border=True):
        st.markdown("**ECG & exercise information**")
        c1,c2,c3 = st.columns(3)
        with c1: restecg = st.selectbox("Resting ECG",[(0,"Normal"),(1,"ST-T wave abnormality"),(2,"Left ventricular hypertrophy")],format_func=lambda x:x[1])[0]
        with c2: thalach = st.number_input("Maximum Heart Rate",50,220,150,1)
        with c3: exang = st.selectbox("Exercise-induced Angina",["No","Yes"])
    with st.container(border=True):
        st.markdown("**Additional clinical indicators**")
        c1,c2,c3,c4 = st.columns(4)
        with c1: oldpeak = st.number_input("Oldpeak (ST depression)",0.0,10.0,1.5,0.1)
        with c2: slope = st.selectbox("ST Segment Slope",[(0,"Upsloping"),(1,"Flat"),(2,"Downsloping")],format_func=lambda x:x[1],index=1)[0]
        with c3: ca = st.number_input("Major Vessels (0–4)",0,4,0,1)
        with c4: thal = st.selectbox("Thalassemia",[(0,"Normal"),(1,"Fixed defect"),(2,"Reversible defect"),(3,"Unknown")],format_func=lambda x:x[1],index=2)[0]
    return {"age":age,"sex":sex,"cp":cp,"trestbps":trestbps,"chol":chol,"fbs":fbs,"restecg":restecg,"thalach":thalach,"exang":exang,"oldpeak":oldpeak,"slope":slope,"ca":ca,"thal":thal}


def show_result(prediction, probability, model_name):
    risk_percent = probability * 100
    confidence = max(probability, 1 - probability) * 100
    st.markdown('<div class="result-card">', unsafe_allow_html=True)
    st.markdown('<div class="result-title"><span class="pulse-dot"></span>Prediction Result</div>', unsafe_allow_html=True)
    st.write("")
    c1,c2,c3 = st.columns(3)
    with c1: st.metric("Risk Level", "HIGH ⚠️" if prediction else "LOW ✅")
    with c2: st.metric("Estimated Risk", f"{risk_percent:.1f}%")
    with c3: st.metric("Model Confidence", f"{confidence:.1f}%")
    st.markdown("### Risk Probability")
    st.progress(float(np.clip(probability,0,1)), text=f"Estimated probability: {risk_percent:.1f}%")
    if prediction:
        st.error("🚨 The model indicates an elevated likelihood of heart disease. Please consult a qualified healthcare professional for proper evaluation.")
        play_sound("warning")
    else:
        st.success("✅ The model indicates a lower likelihood of heart disease based on the supplied inputs.")
        play_sound("success")
        st.balloons()
    if model_name: st.caption(f"Prediction generated using: `{model_name}`")
    else: st.warning("No trained pickle model was found. This result uses the app's fallback scoring estimator and should not be treated as a medical diagnosis.")
    st.info("ℹ️ This application is for educational and demonstration purposes only. It does not replace professional medical advice, diagnosis, or treatment.")
    st.markdown('</div>', unsafe_allow_html=True)


def main():
    st.set_page_config(page_title="Heart Disease Prediction",page_icon="❤️",layout="wide",initial_sidebar_state="expanded")
    inject_css()
    with st.sidebar:
        st.markdown("## ❤️ Heart Health")
        st.markdown("**Heart Disease Prediction**")
        st.divider()
        st.markdown("### About")
        st.write("Estimate heart disease risk from clinical measurements using the configured prediction model.")
        st.markdown("### ✨ Experience")
        st.write("Animated interface, live risk visualization and result sound effects.")
        st.divider()
        st.caption("Educational tool • Not a medical diagnosis")

    st.markdown('''<div class="hero"><div><span class="heart-icon">❤️</span></div><div class="main-title">Heart Disease Prediction</div><div class="subtitle">A modern, interactive interface for estimating heart disease risk from clinical inputs.</div></div>''',unsafe_allow_html=True)
    model, model_name = load_model()
    if model_name:
        st.markdown(f'<div class="model-ready">🟢 <b>Model ready:</b> {model_name}</div>',unsafe_allow_html=True)
    else:
        st.info("Running in fallback mode because no saved model file was found.")
    st.write("")
    with st.form("heart_prediction_form"):
        patient = build_patient_data()
        st.markdown("###")
        submitted = st.form_submit_button("🔍  Predict Heart Disease Risk",use_container_width=True,type="primary")
    if submitted:
        with st.spinner("🫀 Analyzing patient data..."):
            prepared = prepare_patient_for_model(patient)
            prediction, probability = predict_heart_disease(prepared,model=model)
        st.divider()
        show_result(prediction,probability,model_name)


if __name__ == "__main__":
    main()
