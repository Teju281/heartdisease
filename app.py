# Heart Disease Prediction Web App using Streamlit

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import streamlit as st

# ===============================
# 1. Load and preprocess dataset
# ===============================
df = pd.read_csv("heart_disease.csv")

# Handle missing values
for col in df.columns:
    if df[col].dtype in ["float64", "int64"]:
        df[col].fillna(df[col].mean(), inplace=True)
    else:
        df[col].fillna(df[col].mode()[0], inplace=True)

# Encode categorical variables
le_dict = {}
for col in df.columns:
    if df[col].dtype == "object":
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        le_dict[col] = le

# Select only important features
selected_features = ["Age", "Sex", "Chest pain type", "Cholesterol",
                     "FBS over 120", "Exercise angina", "Thallium"]

X = df[selected_features]
y = df["Heart Disease"]

# Scale features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# ===============================
# 2. Streamlit Web App
# ===============================
st.set_page_config(page_title="❤️ Heart Disease Prediction", page_icon="💊", layout="centered")

st.title("❤️ Heart Disease Prediction App")
st.write("Fill in the details below to predict the likelihood of heart disease.")

# Collect inputs
age = st.slider("Age", 20, 100, 50)
sex = st.radio("Sex", ["Male", "Female"])
cp = st.selectbox("Chest Pain Type", ["Typical Angina", "Atypical Angina", "Non-Anginal", "Asymptomatic"])
chol = st.number_input("Cholesterol Level (mg/dl)", 100, 600, 200)
fbs = st.radio("Fasting Blood Sugar > 120 mg/dl", ["Yes", "No"])
exang = st.radio("Exercise Induced Angina", ["Yes", "No"])
thal = st.selectbox("Thallium Test Result", ["Normal", "Fixed Defect", "Reversible Defect"])

# Convert categorical to numeric
sex_val = le_dict["Sex"].transform([sex])[0] if "Sex" in le_dict else (1 if sex == "Male" else 0)
cp_val = le_dict["Chest pain type"].transform([cp])[0] if "Chest pain type" in le_dict else 0
fbs_val = 1 if fbs == "Yes" else 0
exang_val = 1 if exang == "Yes" else 0
thal_val = le_dict["Thallium"].transform([thal])[0] if "Thallium" in le_dict else 0

# Create input array
input_data = np.array([[age, sex_val, cp_val, chol, fbs_val, exang_val, thal_val]])

# Scale input
input_data = scaler.transform(input_data)

# Predict button
if st.button("🔍 Predict"):
    prediction = model.predict(input_data)[0]
    if prediction == 1:
        st.error("⚠️ The patient is likely to have **Heart Disease**")
    else:
        st.success("✅ The patient is **NOT likely** to have Heart Disease")
