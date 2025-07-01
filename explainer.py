import shap
import pandas as pd
import joblib

def get_explainer(model):
    # Sample background data for SHAP (use the same data you trained on or clean subset)
    background = pd.read_csv("data/dataset.csv")
    features = [
        "male", "age", "education", "currentSmoker", "cigsPerDay", "BPMeds",
        "prevalentStroke", "prevalentHyp", "diabetes", "totChol", "sysBP",
        "diaBP", "BMI", "heartRate", "glucose"
    ]
    background = background[features].dropna()
    background_sample = background.sample(100, random_state=42)

    explainer = shap.TreeExplainer(model, data=background_sample)
    return explainer