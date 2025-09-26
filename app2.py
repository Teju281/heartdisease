# ==============================
# heart_disease_app_beautiful.py
# ==============================

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import pyttsx3
import speech_recognition as sr
import threading
import matplotlib.pyplot as plt
import difflib

# ===============================
# Load and preprocess dataset
# ===============================
@st.cache_data
def load_data():
    df = pd.read_csv("heart_disease.csv")
    
    df["fbs"] = df["fbs"].map({True:1, False:0, "TRUE":1, "FALSE":0})
    df["exang"] = df["exang"].map({True:1, False:0, "TRUE":1, "FALSE":0})
    
    for col in df.columns:
        if df[col].dtype in ["float64", "int64"]:
            df[col].fillna(df[col].mean(), inplace=True)
        else:
            df[col].fillna(df[col].mode()[0], inplace=True)

    cat_cols = ["sex", "cp", "restecg", "slope", "thal"]
    le_dict = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        le_dict[col] = le

    return df, le_dict

df, le_dict = load_data()

# Features and target
selected_features = ["age", "sex", "cp", "trestbps", "chol", "fbs",
                     "restecg", "thalch", "exang", "oldpeak", "slope", "ca", "thal"]
X = df[selected_features]
y = df["num"].astype(int)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# ===============================
# Streamlit Page Config
# ===============================
st.set_page_config(page_title="❤️ Heart Disease App", page_icon="❤️", layout="wide")
st.markdown("""
    <style>
    .stApp { background-color: #fff0f0; font-family: 'Arial'; }
    .card { background-color: #ffffff; padding: 20px; border-radius: 15px; 
            box-shadow: 0 4px 20px rgba(0,0,0,0.1); margin-bottom: 20px; }
    .title { color: #ff4b5c; font-weight: bold; }
    .metric { background-color: #ff6b6b; color: white; padding: 10px; border-radius: 10px; }
    </style>
""", unsafe_allow_html=True)

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["🏠 Home", "🔍 Prediction", "💬 Voice Chatbot"])

# ===============================
# Helper
# ===============================
def safe_transform(le, val):
    try:
        return le.transform([val])[0]
    except:
        return -1

# ===============================
# Home Page
# ===============================
if page == "🏠 Home":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<h1 class="title">❤️ Heart Disease Prediction App</h1>', unsafe_allow_html=True)
    st.markdown("💖 **Heart disease** is a leading cause of death worldwide.")
    st.markdown("This app helps you: 📈 Understand risk factors | 🩺 Predict likelihood | 💬 Chat with AI")
    
    st.subheader("📊 Dataset Insights")
    disease_counts = df["num"].apply(lambda x: 0 if x==0 else 1).value_counts()
    fig1, ax1 = plt.subplots()
    ax1.pie(disease_counts, labels=["No Disease", "Disease"], autopct="%1.1f%%", startangle=90, colors=["#76c7c0","#ff6b6b"])
    ax1.axis("equal")
    st.pyplot(fig1)

    df["AgeGroup"] = pd.cut(df["age"], bins=[20,30,40,50,60,70,80,100],
                            labels=["20-30","31-40","41-50","51-60","61-70","71-80","81+"])
    age_group_counts = df.groupby("AgeGroup")["num"].apply(lambda x: (x>0).sum())
    fig2, ax2 = plt.subplots()
    age_group_counts.plot(kind="bar", ax=ax2, color="#ff6b6b", alpha=0.8)
    ax2.set_ylabel("Number of Heart Disease Cases")
    ax2.set_xlabel("Age Group")
    ax2.set_title("Heart Disease by Age Group")
    st.pyplot(fig2)
    
    st.markdown("""
    ❤️ **Tips for Heart Health:**  
    - Eat a balanced diet 🥗  
    - Exercise regularly 🏃‍♂️  
    - Avoid smoking 🚭  
    - Monitor blood pressure & cholesterol 🩺  
    - Manage stress 🧘‍♀️  
    """)
    st.markdown('</div>', unsafe_allow_html=True)

# ===============================
# Prediction Page
# ===============================
elif page == "🔍 Prediction":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<h1 class="title">🔍 Predict Heart Disease Risk</h1>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.slider("Age", 20, 100, 50)
        sex = st.selectbox("Sex", le_dict["sex"].classes_)
        cp = st.selectbox("Chest Pain Type", le_dict["cp"].classes_)
        trestbps = st.number_input("Resting BP (mmHg)", 80, 200, 120)
        chol = st.number_input("Cholesterol (mg/dl)", 100, 600, 200)
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["No","Yes"])
        restecg = st.selectbox("Resting ECG Result", le_dict["restecg"].classes_)
        
    with col2:
        thalch = st.number_input("Max Heart Rate Achieved", 60, 220, 150)
        exang = st.selectbox("Exercise Induced Angina", ["No","Yes"])
        oldpeak = st.number_input("ST Depression (oldpeak)", 0.0, 10.0, 1.0, 0.1)
        slope = st.selectbox("Slope of ST Segment", le_dict["slope"].classes_)
        ca = st.slider("Number of Major Vessels (0-3)", 0, 3, 0)
        thal = st.selectbox("Thalassemia", le_dict["thal"].classes_)
    
    sex_val = safe_transform(le_dict["sex"], sex)
    cp_val = safe_transform(le_dict["cp"], cp)
    restecg_val = safe_transform(le_dict["restecg"], restecg)
    slope_val = safe_transform(le_dict["slope"], slope)
    thal_val = safe_transform(le_dict["thal"], thal)
    fbs_val = 1 if fbs=="Yes" else 0
    exang_val = 1 if exang=="Yes" else 0
    
    input_data = np.array([[age, sex_val, cp_val, trestbps, chol, fbs_val,
                            restecg_val, thalch, exang_val, oldpeak, slope_val, ca, thal_val]])
    input_scaled = scaler.transform(input_data)
    
    if st.button("Predict"):
        pred = model.predict(input_scaled)[0]
        prob = model.predict_proba(input_scaled)[0][pred]*100
        if pred == 0:
            st.success(f"✅ No Heart Disease ({prob:.2f}% confidence)")
        else:
            st.error(f"⚠️ Heart Disease Likely ({prob:.2f}% confidence)")
    
    st.markdown('</div>', unsafe_allow_html=True)

# ===============================
# Voice Chatbot
# ===============================
elif page == "💬 Voice Chatbot":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<h1 class="title">💬 Heart Disease Chatbot</h1>', unsafe_allow_html=True)
    
    engine = pyttsx3.init()
    faq = {
        "symptoms": "Symptoms include chest pain, shortness of breath, fatigue, irregular heartbeat, and dizziness.",
        "causes": "Causes include high blood pressure, high cholesterol, diabetes, obesity, smoking, stress, and family history.",
        "prevention": "Healthy diet, regular exercise, avoid smoking, manage stress.",
        "treatment": "Treatments depend on severity: medications, angioplasty, or surgery.",
        "diet": "Heart-healthy diet: fruits, vegetables, whole grains, lean proteins, low salt/saturated fat.",
        "risk factors": "Age, gender, high BP, high cholesterol, diabetes, smoking, family history."
    }
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    def speak(text):
        def run():
            engine.say(text)
            engine.runAndWait()
        threading.Thread(target=run).start()
    
    def listen():
        r = sr.Recognizer()
        with sr.Microphone() as source:
            st.info("🎤 Listening...")
            audio = r.listen(source, phrase_time_limit=5)
        try:
            return r.recognize_google(audio)
        except:
            return "Sorry, I could not understand."
    
    col1, col2 = st.columns(2)
    with col1:
        user_input = st.text_input("Type your question:")
    with col2:
        if st.button("🎙️ Speak"):
            voice_input = listen()
            if voice_input and "Sorry" not in voice_input:
                st.session_state.chat_history.append(("You", voice_input))
                user_input = voice_input
    
    if st.button("Ask"):
        if user_input.strip():
            st.session_state.chat_history.append(("You", user_input))
            match = difflib.get_close_matches(user_input.lower(), list(faq.keys()), n=1, cutoff=0.3)
            response = faq[match[0]] if match else "🤔 I'm not sure. Please consult a doctor."
            st.session_state.chat_history.append(("Bot", response))
            speak(response)
    
    for sender, msg in st.session_state.chat_history:
        if sender == "You":
            st.markdown(f"🧑 **You:** {msg}")
        else:
            st.markdown(f"🤖 **Bot:** {msg}")
    
    st.markdown('</div>', unsafe_allow_html=True)
