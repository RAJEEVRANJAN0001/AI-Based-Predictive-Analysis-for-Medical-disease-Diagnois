# AI-Based-Predictive-Analysis-for-Medical-disease-Diagnois

## 📌 Project Overview
This repository contains an AI-driven diagnostic system for detecting **diabetes and hypothyroidism** using **Support Vector Machine (SVM)** models. The project leverages machine learning techniques for predictive analytics, incorporating hyperparameter tuning, data preprocessing, and model evaluation.

## 🚀 Features
- **Data Preprocessing**: Standardization, feature scaling, and handling class imbalance.
- **Machine Learning Model**: Optimized **SVM classifier** with **GridSearchCV** for hyperparameter tuning.
- **Model Evaluation**: Accuracy, classification reports, confusion matrices.
- **Explainability**: Visualization of feature importance (for linear kernels).
- **Web Interface**: Streamlit-based UI for real-time disease prediction.

## 📂 Project Structure
```
├── diabetes/
│   ├── diabetes_data.csv               # Preprocessed diabetes dataset
│   ├── diabetes_testing.py             # Script for testing diabetes model
│   ├── scaler_diabetes.pkl             # Standard scaler for diabetes model
│   ├── stremlit_dibetes.py             # Streamlit UI for diabetes prediction
│   ├── svm_diabetes_best_model.pkl     # Trained SVM model for diabetes
│
├── hypothyroid/
│   ├── prepocessed_hypothyroid.csv     # Preprocessed hypothyroid dataset
│   ├── prepocessed_hypothyroid.py      # Script for training hypothyroid model
│   ├── scaler.pkl                      # Standard scaler for hypothyroid model
│   ├── streamlittesting.py             # Streamlit UI for hypothyroid prediction
│   ├── svm_hypothyroid_model.pkl       # Trained SVM model for hypothyroid
│
├── README.md                           # Project documentation (this file)
```

## 🛠 Installation & Setup
1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/ai-diagnostic-system.git
   cd ai-diagnostic-system
   ```
2. **Run the training script (if needed)**:
   ```bash
   python diabetes/diabetes_testing.py
   python hypothyroid/prepocessed_hypothyroid.py
   ```
3. **Launch the Streamlit app**:
   ```bash
   streamlit run diabetes/stremlit_dibetes.py
   streamlit run hypothyroid/streamlittesting.py
   ```

## 🔍 Model Training & Evaluation
- **Hyperparameter tuning**: Utilizes **GridSearchCV** to find the best parameters.
- **Model accuracy**:
  - **Diabetes Model**: ~75.32%
  - **Hypothyroid Model**: ~98.94%
- **Confusion matrix & classification reports**: Generated after testing to assess model performance.

## 📊 Results & Visualizations
- **Confusion Matrix**: Evaluates model predictions vs. actual values.
- **Feature Importance**: If an SVM with a linear kernel is used, feature importance is visualized.
- **Streamlit UI**: Enables users to input medical parameters and receive real-time predictions.

## 📎 Pretrained Models
- `diabetes/svm_diabetes_best_model.pkl`
- `hypothyroid/svm_hypothyroid_model.pkl`
- `diabetes/scaler_diabetes.pkl`
- `hypothyroid/scaler.pkl`

## 💡 Future Enhancements
- Improve feature selection and dataset diversity.
- Incorporate deep learning models (CNNs for medical images).
- Enhance interpretability with SHAP/LIME for better clinician trust.

## ⭐ Acknowledgments
Special thanks to **AICTE and TechSaksham** for providing the learning platform for this project.

---

🚀 If you found this useful, don't forget to **⭐ Star** the repository!
