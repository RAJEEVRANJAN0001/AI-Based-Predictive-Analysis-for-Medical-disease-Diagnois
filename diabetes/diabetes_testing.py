
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import pandas as pd
import joblib
import warnings
from sklearn.preprocessing import StandardScaler

# ✅ Suppress UserWarning
warnings.filterwarnings("ignore", category=UserWarning)

# 🔹 Load the trained SVM model and scaler
svm_model = joblib.load("svm_diabetes_best_model.pkl")
scaler = joblib.load("scaler_diabetes.pkl")

# 🔹 Define feature names (Adjust based on your dataset)
feature_names = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin",
    "BMI", "DiabetesPedigreeFunction", "Age"
]

# 🔹 Ask the user for input values
print("\n🔹 Enter the following health metrics to predict diabetes:")
user_input = []
for feature in feature_names:
    value = float(input(f"👉 Enter {feature}: "))
    user_input.append(value)

# 🔹 Convert input to NumPy array and reshape for model
user_data = np.array(user_input).reshape(1, -1)

# ✅ **Fix: Convert to DataFrame before scaling**
user_data_df = pd.DataFrame(user_data, columns=feature_names)
user_data_scaled = scaler.transform(user_data_df)

# 🔹 Make prediction
prediction = svm_model.predict(user_data_scaled)[0]
probability = svm_model.predict_proba(user_data_scaled)[0][1]  # Probability of being diabetic

# 🔹 Display result
print("\n✅ **Diabetes Prediction Result:**")
if prediction == 1:
    print("🔴 **You are at risk of diabetes.** (Consult a doctor for further evaluation.)")
else:
    print("🟢 **You are not at risk of diabetes.** (Maintain a healthy lifestyle.)")

print(f"🔹 Probability of being diabetic: {probability:.2%}")
