import streamlit as st
import pandas as pd
import joblib
import shap
import numpy as np
import matplotlib.pyplot as plt
from utils.explainer import get_explainer

# Page configuration
st.set_page_config(
    page_title="CardioWise",
    page_icon="❤️",
    layout="centered"
)

# Title
st.title("❤️ CardioWise - Interpretable Heart Risk Prediction")
st.markdown("""
Welcome to **CardioWise**, a simple tool that uses machine learning to estimate your 10-year heart disease risk based on medical data.  
It also provides interpretable insights using SHAP (Shapley values) to explain the factors influencing your result.
""")

# Load model
model = joblib.load("model/rf_model.pkl")

# Get SHAP explainer (pre-initialized with training data inside explainer.py)
explainer = get_explainer(model)

# Sidebar for user input
st.sidebar.header("Enter your medical details")

def user_input():
    gender = st.sidebar.selectbox("Sex", ("Male", "Female"))
    male = 1 if gender == "Male" else 0
    age = st.sidebar.slider("Age", 30, 80, 50)
    education = st.sidebar.selectbox("Education Level", [1, 2, 3, 4])
    currentSmoker = st.sidebar.selectbox("Currently Smokes?", ("Yes", "No"))
    smoker = 1 if currentSmoker == "Yes" else 0
    cigsPerDay = st.sidebar.slider("Cigarettes per Day", 0, 50, 10)
    BPMeds = st.sidebar.selectbox("On BP Medication?", ("Yes", "No"))
    bpm = 1 if BPMeds == "Yes" else 0
    prevalentStroke = st.sidebar.selectbox("Had a Stroke Before?", ("Yes", "No"))
    stroke = 1 if prevalentStroke == "Yes" else 0
    prevalentHyp = st.sidebar.selectbox("Has Hypertension?", ("Yes", "No"))
    hyp = 1 if prevalentHyp == "Yes" else 0
    diabetes = st.sidebar.selectbox("Diabetic?", ("Yes", "No"))
    diab = 1 if diabetes == "Yes" else 0
    totChol = st.sidebar.slider("Total Cholesterol", 100, 400, 200)
    sysBP = st.sidebar.slider("Systolic BP", 90, 200, 120)
    diaBP = st.sidebar.slider("Diastolic BP", 60, 140, 80)
    BMI = st.sidebar.slider("BMI", 15.0, 50.0, 25.0)
    heartRate = st.sidebar.slider("Heart Rate", 40, 120, 70)
    glucose = st.sidebar.slider("Glucose", 50, 300, 80)

    user_data = pd.DataFrame([[
        male, age, education, smoker, cigsPerDay, bpm, stroke,
        hyp, diab, totChol, sysBP, diaBP, BMI, heartRate, glucose
    ]], columns=[
        "male", "age", "education", "currentSmoker", "cigsPerDay", "BPMeds",
        "prevalentStroke", "prevalentHyp", "diabetes", "totChol", "sysBP",
        "diaBP", "BMI", "heartRate", "glucose"
    ])
    
    return user_data

# Collect input
input_df = user_input()

# Predict and explain
if st.button("🔍 Predict Risk and Explain"):
    prob = model.predict_proba(input_df)[0][1]
    prediction = model.predict(input_df)[0]
    
    st.subheader("🧠 Prediction Result")
    st.write(f"**Predicted 10-year CHD Risk:** {prob:.2%}")
    st.write(f"**Risk Classification:** {'High Risk' if prediction == 1 else 'Low Risk'}")

    st.markdown("---")
    st.subheader("🔎 What influenced this prediction?")

    shap_values = explainer.shap_values(input_df)
    st.set_option('deprecation.showPyplotGlobalUse', False)
    shap.initjs()
    shap.force_plot(
        explainer.expected_value[1],
        shap_values[1],
        input_df,
        matplotlib=True,
        show=False
    )
    st.pyplot(bbox_inches='tight')
