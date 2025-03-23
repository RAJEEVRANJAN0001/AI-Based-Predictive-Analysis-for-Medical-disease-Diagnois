import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

# Load the trained model and scaler
model = joblib.load("/Users/rajeevranjanpratapsingh/PycharmProjects/new report project intern /svm_hypothyroid_model.pkl")
scaler = joblib.load("/Users/rajeevranjanpratapsingh/PycharmProjects/new report project intern /scaler.pkl")

def predict(data):
    scaled_data = scaler.transform([data])
    prediction = model.predict(scaled_data)
    return prediction[0]

# Streamlit UI
st.set_page_config(page_title="Hypothyroid Detection", layout="centered")
st.markdown("""
    <style>
        body {background-color: #f4f4f4;}
        .stButton>button {background-color: #4CAF50; color: white; border-radius: 10px;}
        .stTextInput>div>input {border-radius: 10px;}
    </style>
""", unsafe_allow_html=True)

st.title("🔬 Hypothyroid Detection System")
st.write("Enter the required patient details to predict hypothyroidism.")

# Load dataset to get feature names
data = pd.read_csv("prepocessed_hypothyroid.csv")
feature_names = list(data.columns[:-1])

test_values = []
for feature in feature_names:
    value = st.number_input(f"{feature}", min_value=-10.0, max_value=100.0, value=0.0)
    test_values.append(value)

if st.button("Predict Hypothyroid"):
    result = predict(test_values)
    if result == 1:
        st.error("🚨 The patient is predicted to have hypothyroidism.")
    else:
        st.success("✅ The patient is not predicted to have hypothyroidism.")
