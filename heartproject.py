# Heart Disease Prediction with GUI

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import tkinter as tk
from tkinter import messagebox

# ===============================
# 1. Load and preprocess dataset
# ===============================
df = pd.read_csv("heart_disease.csv")  # <-- make sure your file name matches!

# Handle missing values (numeric → mean, categorical → mode)
for col in df.columns:
    if df[col].dtype in ["float64", "int64"]:
        df[col].fillna(df[col].mean(), inplace=True)
    else:
        df[col].fillna(df[col].mode()[0], inplace=True)

# Encode categorical variables
le = LabelEncoder()
for col in df.columns:
    if df[col].dtype == "object":
        df[col] = le.fit_transform(df[col])

# Define features and target
X = df.drop("Heart Disease", axis=1)   # <-- Adjust if your target column differs
y = df["Heart Disease"]

# Scale features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Model Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))


# ===============================
# 2. Build GUI with Tkinter
# ===============================
root = tk.Tk()
root.title("Heart Disease Prediction")
root.geometry("500x600")

entries = {}
feature_names = df.drop("Heart Disease", axis=1).columns.tolist()

# Create form for input
for idx, feature in enumerate(feature_names):
    lbl = tk.Label(root, text=feature, anchor="w")
    lbl.grid(row=idx, column=0, padx=10, pady=5, sticky="w")
    ent = tk.Entry(root, width=25)
    ent.grid(row=idx, column=1, padx=10, pady=5)
    entries[feature] = ent

# Prediction function
def predict():
    try:
        # Collect input values
        input_data = []
        for feature in feature_names:
            val = entries[feature].get()
            if val == "":
                messagebox.showerror("Error", f"Please enter value for {feature}")
                return
            input_data.append(float(val))  # convert to float

        # Scale input
        input_data = scaler.transform([input_data])

        # Predict
        prediction = model.predict(input_data)[0]
        if prediction == 1:
            messagebox.showinfo("Result", "⚠️ The patient is likely to have Heart Disease")
        else:
            messagebox.showinfo("Result", "✅ The patient is NOT likely to have Heart Disease")
    except Exception as e:
        messagebox.showerror("Error", str(e))

# Button
btn = tk.Button(root, text="Predict", command=predict, bg="green", fg="white", font=("Arial", 12, "bold"))
btn.grid(row=len(feature_names), column=0, columnspan=2, pady=20)

root.mainloop()




























































































































































































































































































































































































































































































