# 🫀 CardioWise – Interpretable Heart Risk Classifier

**CardioWise** is a lightweight, end-to-end AI tool that predicts a patient's 10-year risk of coronary heart disease (CHD) using clinical data — and explains each prediction using SHAP interpretability.

This app demonstrates the power of combining tabular machine learning with model explainability, presented through a simple Streamlit interface.

---

## 🚀 Live Features

- 🔎 Upload or enter patient features (e.g. age, cholesterol, glucose, BMI, smoking).
- 🤖 Predict 10-year CHD risk using a trained Random Forest model.
- 🧠 Visualize SHAP values showing which features influenced the prediction.
- ⚖️ Balanced model with class weighting and threshold tuning to detect rare CHD cases.

---

## 🧠 Model Details

- Algorithm: `RandomForestClassifier` (with `class_weight='balanced'`)
- Dataset: [Framingham Heart Study](https://www.kaggle.com/datasets/dileep070/heart-disease-prediction-using-logistic-regression)
- Preprocessing:
  - Null value imputation
  - MinMaxScaler
  - Class imbalance handled via RandomOverSampler
- Evaluation:
  - Accuracy: ~80%
  - AUC-ROC: ~0.63
  - Threshold-tuned F1 on minority class (CHD=1): ~0.30

---

## 📂 Project Structure

```
CardioWise/
├── app.py                  # Streamlit application
├── model/
│   └── rf_model.pkl        # Trained Random Forest model
├── data/
│   └── dataset.csv         # Input clinical dataset (optional to share)
├── utils/
│   └── explainer.py        # SHAP functions and helper utilities
├── notebooks/
│   └── 01_EDA_and_Model.ipynb  # Local training + analysis (not published)
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

---

## ⚙️ How to Run Locally

1. **Clone the repo**:

```bash
git clone https://github.com/mufithamajeed/CardioWise.git
cd CardioWise
```

2. **Create virtual environment**:

```bash
python -m venv .venv
# On Windows
.venv\Scripts\activate
# On Mac/Linux
source .venv/bin/activate
```

3. **Install dependencies**:

```bash
pip install -r requirements.txt
```

4. **Run Streamlit app**:

```bash
streamlit run app.py
```

Then open `http://localhost:8501` in your browser.

---

## 📈 Example Use Case

```json
{
  "age": 61,
  "sex": 0,
  "currentSmoker": 1,
  "cigsPerDay": 30,
  "totChol": 225,
  "sysBP": 150,
  "diaBP": 95,
  "glucose": 103,
  "BMI": 28.5,
  "heartRate": 65,
  "diabetes": 0
}
```

**Predicted Risk**: High  
SHAP shows `sysBP`, `cigsPerDay`, and `age` were key contributors.

---

## 📊 Model Interpretability

- SHAP values are shown for every prediction.
- Each feature’s impact is visualized via force plots.
- Threshold tuning was used to improve minority class recall.

---

## 📄 License

MIT License. Feel free to reuse, modify, and extend — but **please cite the repository** if you build upon it.

---

## ✨ Credits

Developed by Mufitha Majeed as part of a research/portfolio project in applied healthcare ML.  
Guided by strong academic principles of interpretability, fairness, and transparency.