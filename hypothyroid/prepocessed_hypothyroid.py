import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

# Load the dataset
data = pd.read_csv("/Users/rajeevranjanpratapsingh/PycharmProjects/new report project intern /prepocessed_hypothyroid.csv")

# Assuming the last column is the target variable
y = data.iloc[:, -1]
X = data.iloc[:, :-1]

# Standardize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Save the fitted scaler
joblib.dump(scaler, "scaler.pkl")  # ✅ Now saving the scaler

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Hyperparameter tuning using GridSearchCV
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.01, 0.001],
    'kernel': ['rbf', 'poly', 'sigmoid']
}

grid_search = GridSearchCV(SVC(class_weight='balanced'), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Best model after hyperparameter tuning
best_svm_model = grid_search.best_estimator_

# Train the best model
best_svm_model.fit(X_train, y_train)

# Save the trained model
joblib.dump(best_svm_model, "svm_hypothyroid_model.pkl")

# Make predictions
y_pred = best_svm_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
print("Classification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Plot feature importance using SVM coefficients if linear kernel was used
if best_svm_model.kernel == 'linear':
    feature_importance = np.abs(best_svm_model.coef_).mean(axis=0)
    plt.bar(range(len(feature_importance)), feature_importance)
    plt.xlabel('Feature Index')
    plt.ylabel('Importance')
    plt.title('Feature Importance in SVM')
    plt.show()
