import os
from pathlib import Path

import numpy as np
import streamlit as st

# Recommended feature order for a standard heart disease dataset
FEATURES = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
]

MODEL_CANDIDATES = [
    Path("heart_model.pkl"),
    Path("model.pkl"),
    Path("heart_disease_model.pkl"),
    Path("models/heart_model.pkl"),
]


def load_model():
    """Try to load a trained pickle model if one exists.
    If not found, return None and the app falls back to a built-in risk estimator.
    """
    for model_path in MODEL_CANDIDATES:
        if model_path.exists():
            try:
                import pickle

                with open(model_path, "rb") as f:
                    model = pickle.load(f)
                st.success(f"Loaded model from: {model_path}")
                return model
            except Exception as exc:
                st.warning(f"Failed to load {model_path}: {exc}")
    return None


def clinical_risk_probability(patient):
    """Fallback risk estimator.
    This is a lightweight scoring model to keep the app working even before a real
    trained model is provided. Replace it with your actual saved model when ready.
    """
    age = patient["age"]
    sex = patient["sex"]
    cp = patient["cp"]
    trestbps = patient["trestbps"]
    chol = patient["chol"]
    fbs = patient["fbs"]
    restecg = patient["restecg"]
    thalach = patient["thalach"]
    exang = patient["exang"]
    oldpeak = patient["oldpeak"]
    slope = patient["slope"]
    ca = patient["ca"]
    thal = patient["thal"]

    score = 0.0

    if age >= 60:
        score += 2.0
    elif age >= 45:
        score += 1.0

    if sex == 1:
        score += 1.0

    if cp in (1, 2):
        score += 1.5
    elif cp == 3:
        score += 2.0

    if trestbps > 140:
        score += 1.5
    if chol > 200:
        score += 1.0
    if fbs > 120:
        score += 1.0
    if restecg == 1:
        score += 0.8
    elif restecg == 2:
        score += 1.2

    if thalach < 120:
        score += 1.5
    elif thalach < 150:
        score += 0.5

    if exang == 1:
        score += 2.0
    if oldpeak > 2.0:
        score += 2.0
    elif oldpeak > 1.0:
        score += 1.0

    if slope in (1, 2):
        score += 1.0
    if ca > 0:
        score += 2.0
    if thal in (2, 3):
        score += 1.5
    elif thal == 1:
        score += 0.5

    # Convert score to probability between 0 and 1
    z = (score - 7.0) / 2.8
    probability = 1 / (1 + np.exp(-z))
    return float(probability)


def predict_heart_disease(patient, model=None):
    """Predict using a real sklearn model if available, otherwise use fallback scoring."""
    if model is not None:
        try:
            values = np.array([patient[name] for name in FEATURES], dtype=float).reshape(1, -1)
            result = model.predict(values)[0]
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(values)[0]
                probability = float(proba[1]) if len(proba) > 1 else float(proba[0])
            else:
                probability = float(result)
            return int(result), probability
        except Exception:
            pass

    probability = clinical_risk_probability(patient)
    return 1 if probability >= 0.5 else 0, probability


def build_patient_data():
    return {
        "age": st.number_input("Age", min_value=18, max_value=120, value=52, step=1),
        "sex": st.selectbox("Sex", ["Male", "Female"], index=0),
        "cp": st.selectbox(
            "Chest Pain Type",
            [
                (0, "Typical angina"),
                (1, "Atypical angina"),
                (2, "Non-anginal pain"),
                (3, "Asymptomatic"),
            ],
            format_func=lambda x: x[1],
            index=2,
        )[0],
        "trestbps": st.number_input("Resting Blood Pressure (mmHg)", min_value=80, max_value=220, value=130, step=1),
        "chol": st.number_input("Cholesterol (mg/dl)", min_value=100, max_value=500, value=210, step=1),
        "fbs": st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"], index=0),
        "restecg": st.selectbox(
            "Resting ECG",
            [(0, "Normal"), (1, "ST-T wave abnormality"), (2, "Left ventricular hypertrophy")],
            format_func=lambda x: x[1],
            index=0,
        )[0],
        "thalach": st.number_input("Maximum Heart Rate", min_value=50, max_value=220, value=150, step=1),
        "exang": st.selectbox("Exercise-induced Angina", ["No", "Yes"], index=0),
        "oldpeak": st.number_input("Oldpeak (ST depression)", min_value=0.0, max_value=10.0, value=1.5, step=0.1),
        "slope": st.selectbox(
            "Slope of ST segment",
            [(0, "Upsloping"), (1, "Flat"), (2, "Downsloping")],
            format_func=lambda x: x[1],
            index=1,
        )[0],
        "ca": st.number_input("Number of major vessels colored by fluoroscopy", min_value=0, max_value=4, value=0, step=1),
        "thal": st.selectbox(
            "Thalassemia",
            [(0, "Normal"), (1, "Fixed defect"), (2, "Reversible defect"), (3, "Unknown")],
            format_func=lambda x: x[1],
            index=2,
        )[0],
    }


def prepare_patient_for_model(patient):
    prepared = patient.copy()
    prepared["sex"] = 1 if prepared["sex"] == "Male" else 0
    prepared["fbs"] = 1 if prepared["fbs"] == "Yes" else 0
    prepared["exang"] = 1 if prepared["exang"] == "Yes" else 0
    return prepared


def main():
    st.set_page_config(page_title="Heart Disease Prediction", page_icon="❤️", layout="wide")
    st.title("Heart Disease Prediction App")
    st.caption("Input patient information and estimate heart disease risk.")

    model = load_model()

    with st.form("heart_prediction_form"):
        patient = build_patient_data()
        submitted = st.form_submit_button("Predict")

    if submitted:
        patient = prepare_patient_for_model(patient)
        prediction, probability = predict_heart_disease(patient, model=model)

        st.subheader("Prediction Result")
        risk_percent = round(probability * 100, 2)

        if prediction == 1:
            st.error(f"High risk: {risk_percent}% chance of heart disease")
        else:
            st.success(f"Low risk: {risk_percent}% chance of heart disease")

        st.progress(probability)

        st.write("Model probability of heart disease:", f"{risk_percent}%")

        st.info(
            "Tip: If you have a trained .pkl model, save it as heart_model.pkl in the same folder "
            "and the app will automatically use it instead of the built-in fallback estimator."
        )


if __name__ == "__main__":
    main()
