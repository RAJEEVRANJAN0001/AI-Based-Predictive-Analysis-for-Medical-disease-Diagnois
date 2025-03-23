import streamlit as st
import numpy as np
import joblib

# ✅ Set page configuration (Must be the first Streamlit command)
st.set_page_config(page_title="Diabetes Prediction", page_icon="💉", layout="wide")

# ✅ Load the trained SVM model and scaler
svm_model = joblib.load("svm_diabetes_best_model.pkl")
scaler = joblib.load("scaler_diabetes.pkl")

# ✅ Apply custom CSS for a stylish UI
st.markdown(
    """
    <style>
        body {
            background-color: #f4f4f4;
        }
        .stButton>button {
            background-color: #008CBA;
            color: white;
            font-size: 18px;
            padding: 10px 24px;
            border-radius: 8px;
            border: none;
            transition: 0.3s;
        }
        .stButton>button:hover {
            background-color: #005f73;
        }
        .prediction-result {
            font-size: 24px;
            font-weight: bold;
            text-align: center;
            padding: 15px;
            border-radius: 10px;
        }
        .positive {
            background-color: #ff4d4d;
            color: white;
        }
        .negative {
            background-color: #28a745;
            color: white;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# ✅ Sidebar for User Input
st.sidebar.title("🔹 Enter Your Health Details")

feature_names = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin",
    "BMI", "DiabetesPedigreeFunction", "Age"
]

user_input = []
for feature in feature_names:
    value = st.sidebar.number_input(f"👉 {feature}", min_value=0.0, format="%.2f")
    user_input.append(value)

# ✅ Predict Button
if st.sidebar.button("🔍 Predict Diabetes"):
    user_data = np.array(user_input).reshape(1, -1)
    user_data_scaled = scaler.transform(user_data)

    # ✅ Add a progress bar for better user experience
    progress_bar = st.progress(0)
    for percent in range(100):
        progress_bar.progress(percent + 1)

    # ✅ Make prediction
    prediction = svm_model.predict(user_data_scaled)[0]
    probability = svm_model.predict_proba(user_data_scaled)[0][1]  # Probability of being diabetic

    # ✅ Display the result
    st.subheader("✅ Diabetes Prediction Result:")

    if prediction == 1:
        st.markdown('<div class="prediction-result positive">🔴 You are at risk of diabetes. Consult a doctor.</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="prediction-result negative">🟢 You are not at risk of diabetes. Maintain a healthy lifestyle.</div>', unsafe_allow_html=True)

    st.write(f"🔹 **Probability of being diabetic:** {probability:.2%}")
