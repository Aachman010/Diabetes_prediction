import os
import joblib
import pandas as pd
import streamlit as st


MODEL_PATH = "diabetes_model.pkl"


st.set_page_config(
    page_title="Diabetes Prediction",
    page_icon="🩺",
    layout="centered",
)


@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            "diabetes_model.pkl was not found. Run `python train.py` first "
            "and keep the generated model file in the project folder."
        )
    return joblib.load(MODEL_PATH)


try:
    model = load_model()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()


st.title("🩺 Diabetes Prediction")
st.write(
    "Enter the patient's information and the trained Logistic Regression "
    "model will return a prediction."
)
st.info("Educational ML project — this output is not a medical diagnosis.")


with st.form("patient_form"):
    st.subheader("Patient Information")

    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox(
            "Gender",
            ["Female", "Male", "Other"],
        )

        age = st.number_input(
            "Age",
            min_value=0.0,
            max_value=120.0,
            value=45.0,
            step=1.0,
        )

        hypertension = st.selectbox(
            "Hypertension",
            [0, 1],
            format_func=lambda x: "No" if x == 0 else "Yes",
        )

        heart_disease = st.selectbox(
            "Heart Disease",
            [0, 1],
            format_func=lambda x: "No" if x == 0 else "Yes",
        )

    with col2:
        smoking_history = st.selectbox(
            "Smoking History",
            ["never", "No Info", "current", "former", "ever", "not current"],
        )

        bmi = st.number_input(
            "BMI",
            min_value=5.0,
            max_value=80.0,
            value=27.5,
            step=0.1,
        )

        hba1c = st.number_input(
            "HbA1c Level",
            min_value=3.0,
            max_value=20.0,
            value=5.8,
            step=0.1,
        )

        blood_glucose = st.number_input(
            "Blood Glucose Level",
            min_value=40,
            max_value=500,
            value=120,
            step=1,
        )

    submitted = st.form_submit_button(
        "🔍 Predict Diabetes Risk",
        use_container_width=True,
    )


if submitted:
    new_patient = pd.DataFrame(
        {
            "gender": [gender],
            "age": [age],
            "hypertension": [hypertension],
            "heart_disease": [heart_disease],
            "smoking_history": [smoking_history],
            "bmi": [bmi],
            "HbA1c_level": [hba1c],
            "blood_glucose_level": [blood_glucose],
        }
    )

    prediction = int(model.predict(new_patient)[0])
    probability = float(model.predict_proba(new_patient)[0, 1])

    st.divider()

    if prediction == 1:
        st.error("Model prediction: Diabetes (class 1)")
    else:
        st.success("Model prediction: No Diabetes (class 0)")

    st.metric("Estimated class-1 probability", f"{probability * 100:.2f}%")

    st.subheader("Input Used")
    st.dataframe(new_patient, use_container_width=True)

    st.caption(
        "The probability shown is the model's estimated probability for "
        "class 1. It should not be interpreted as a clinical diagnosis."
    )
