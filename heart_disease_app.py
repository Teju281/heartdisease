# ============================== 
# heart_disease_app_all_in_one_webspeech_fixed.py
# ==============================

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import difflib

# ===============================
# 1. Load and preprocess dataset
# ===============================
@st.cache_data
def load_data():
    df = pd.read_csv("heart_disease.csv")
    
    # Convert TRUE/FALSE to 1/0 safely
    df["fbs"] = df["fbs"].map({True:1, False:0, "TRUE":1, "FALSE":0})
    df["exang"] = df["exang"].map({True:1, False:0, "TRUE":1, "FALSE":0})
    
    # Fill missing values
    for col in df.columns:
        if df[col].dtype in ["float64", "int64"]:
            df[col].fillna(df[col].mean(), inplace=True)
        else:
            df[col].fillna(df[col].mode()[0], inplace=True)

    # Encode categorical variables
    cat_cols = ["sex", "cp"]
    le_dict = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        le_dict[col] = le

    return df, le_dict

df, le_dict = load_data()

# ===============================
# 2. Features and target
# ===============================
selected_features = ["age", "sex", "cp", "trestbps", "chol", "thalch", "exang"]
X = df[selected_features]
y = df["num"].astype(int)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# ===============================
# Streamlit Page Config
# ===============================
st.set_page_config(page_title="❤️ Heart Disease App", page_icon="❤️", layout="wide")
st.markdown("""
<style>
.stApp { background-color: #fffaf0; }
.main-content { background-color: #ffffff; padding: 25px; border-radius: 15px; 
                box-shadow: 0px 4px 15px rgba(0,0,0,0.1); }
.card { padding: 15px; margin-bottom: 15px; border-radius: 10px; box-shadow:0px 2px 8px rgba(0,0,0,0.1);}
</style>
""", unsafe_allow_html=True)

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["🏠 Home", "🔍 Prediction", "💬 Voice Chatbot"])

# ===============================
# Safe label transform function
# ===============================
def safe_transform(le, val):
    try:
        return le.transform([val])[0]
    except ValueError:
        return -1  # Return -1 if unseen label

# ===============================
# Home Page
# ===============================
if page == "🏠 Home":
    st.markdown('<div class="main-content">', unsafe_allow_html=True)
    st.title("❤️ Heart Disease Prediction App")
    st.subheader("Learn about Heart Disease & Risk Factors")

    st.markdown("""
💖 Heart disease affects the heart and blood vessels.  
Early detection is key! This app helps you:
- 📈 Understand your risk factors  
- 🩺 Predict your likelihood of heart disease  
- 💬 Ask questions via voice-enabled AI
""")

    st.markdown('<div class="card" style="background-color:#ffe6e6;">', unsafe_allow_html=True)
    st.subheader("💡 What is Heart Disease?")
    st.write("Heart disease includes conditions like coronary artery disease, arrhythmia, and heart failure. "
             "It may reduce blood flow and oxygen to your body.")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card" style="background-color:#e6f2ff;">', unsafe_allow_html=True)
    st.subheader("⚠️ Risk Factors")
    st.write("""
- 🩸 High Blood Pressure  
- 🥓 High Cholesterol  
- 🧪 Diabetes  
- 🚭 Smoking  
- ⚖️ Obesity  
- 🛋️ Sedentary Lifestyle  
- 👨‍👩‍👧‍👦 Family History
""")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card" style="background-color:#fff0b3;">', unsafe_allow_html=True)
    st.subheader("🩺 Symptoms to Watch")
    st.write("""
- 💔 Chest Pain  
- 😮 Shortness of Breath  
- 🥱 Fatigue  
- ⚡ Dizziness or Fainting  
- 💓 Irregular Heartbeat
""")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card" style="background-color:#d1ffd6;">', unsafe_allow_html=True)
    st.subheader("🏃 Prevention & Healthy Lifestyle")
    st.write("""
- 🥗 Balanced Diet  
- 🏃‍♀️ Regular Exercise  
- 🚭 Avoid Smoking & Limit Alcohol  
- 🩺 Regular Checkups  
- 🧘‍♂️ Manage Stress
""")
    st.markdown('</div>', unsafe_allow_html=True)

    st.subheader("📊 Dataset Insights")
    disease_counts = df["num"].apply(lambda x: 0 if x==0 else 1).value_counts()
    fig, ax = plt.subplots()
    ax.pie(disease_counts, labels=["No Disease", "Disease"], autopct="%1.1f%%", startangle=90,
           colors=["#76c7c0","#ff6b6b"])
    ax.axis("equal")
    st.pyplot(fig)
    st.markdown('</div>', unsafe_allow_html=True)

# ===============================
# Prediction Page
# ===============================
elif page == "🔍 Prediction":
    st.markdown('<div class="main-content">', unsafe_allow_html=True)
    st.title("🔍 Predict Heart Disease Risk")
    st.write("Fill in the patient details below:")

    age = st.slider("👤 Age", 20, 100, 45)
    sex = st.radio("⚧ Sex", le_dict["sex"].classes_)
    cp = st.selectbox("💔 Chest Pain Type", le_dict["cp"].classes_)
    trestbps = st.slider("🩸 Resting Blood Pressure (mm Hg)", 80, 200, 120)
    chol = st.slider("🥓 Serum Cholesterol (mg/dl)", 100, 600, 200)
    thalch = st.slider("🏃 Maximum Heart Rate Achieved", 60, 220, 150)
    exang = st.selectbox("😮 Exercise Induced Angina?", ["No","Yes"])

    sex_val = safe_transform(le_dict["sex"], sex)
    cp_val = safe_transform(le_dict["cp"], cp)
    exang_val = 1 if exang=="Yes" else 0

    input_data = np.array([[age, sex_val, cp_val, trestbps, chol, thalch, exang_val]])
    input_scaled = scaler.transform(input_data)

    if st.button("Predict"):
        pred = model.predict(input_scaled)[0]
        prob = model.predict_proba(input_scaled)[0][pred]*100
        if pred == 0:
            st.success(f"✅ No Heart Disease ({prob:.2f}% confidence)")
        else:
            st.error(f"⚠️ Likely to have Heart Disease (Class {pred}, {prob:.2f}% confidence)")

    st.markdown('</div>', unsafe_allow_html=True)

# ===============================
# Voice Chatbot Page (Web Speech API)
# ===============================
elif page == "💬 Voice Chatbot":
    st.markdown('<div class="main-content">', unsafe_allow_html=True)
    st.title("💬 Heart Disease Voice Chatbot")
    st.write("Type your question and hear the answer automatically:")

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

    user_input = st.text_input("Type your question:")

    if st.button("Ask"):
        if user_input.strip():
            st.session_state.chat_history.append(("You", user_input))
            text = user_input.lower().strip()
            
            greetings = ["hi","hello","hey","thanks","thank you"]
            if any(word in text for word in greetings):
                response = "😊 You're welcome!"
            else:
                match = difflib.get_close_matches(text, list(faq.keys()), n=1, cutoff=0.3)
                response = faq[match[0]] if match else "🤔 I'm not sure. Please consult a doctor."
            
            st.session_state.chat_history.append(("Bot", response))
            
            js_code = f"""
            <script>
            var msg = new SpeechSynthesisUtterance("{response}");
            window.speechSynthesis.speak(msg);
            </script>
            """
            st.components.v1.html(js_code)

    for sender, msg in st.session_state.chat_history:
        if sender == "You":
            st.markdown(f"🧑 **You:** {msg}")
        else:
            st.markdown(f"🤖 **Bot:** {msg}")

    st.markdown('</div>', unsafe_allow_html=True)
