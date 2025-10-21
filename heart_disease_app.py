# ==============================
# heart_disease_app_all_in_one_v11.py  ✅ Updated Stable Version
# ==============================

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, recall_score
import matplotlib.pyplot as plt
from rapidfuzz import process, fuzz

# ===============================
# 1. Load and preprocess dataset
# ===============================
@st.cache_data
def load_data():
    df = pd.read_csv("heart_disease.csv")

    # Convert TRUE/FALSE to 1/0 safely
    for col in ["fbs", "exang"]:
        if col in df.columns:
            df[col] = df[col].map({True: 1, False: 0, "TRUE": 1, "FALSE": 0}).fillna(0)

    # Fill missing values
    for col in df.columns:
        if df[col].dtype in ["float64", "int64"]:
            df[col].fillna(df[col].mean(), inplace=True)
        else:
            df[col].fillna(df[col].mode()[0], inplace=True)

    # Encode categorical variables safely
    cat_cols = [c for c in ["sex", "cp"] if c in df.columns]
    le_dict = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        le_dict[col] = le

    return df, le_dict

df, le_dict = load_data()

# ===============================
# 2. Features and Target
# ===============================
required_columns = ["age", "sex", "cp", "trestbps", "chol", "thalch", "exang"]
available_cols = [col for col in required_columns if col in df.columns]
X = df[available_cols]

# Target column handling (support for 'num' or 'target')
target_col = "num" if "num" in df.columns else "target"
y = df[target_col].astype(int)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# ===============================
# 3. Model Training & Metrics
# ===============================
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "SVM": SVC(probability=True, random_state=42),
    "KNN": KNeighborsClassifier()
}

accuracies, recalls = {}, {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Accuracy
    accuracies[name] = accuracy_score(y_test, y_pred) * 100

    # Recall (macro average = handles multiclass safely)
    recalls[name] = recall_score(y_test, y_pred, average="macro") * 100

best_model = max(recalls, key=recalls.get)

# ===============================
# Streamlit Page Config & Styling
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
# Helper Function for Safe Label Transform
# ===============================
def safe_transform(le, val):
    try:
        return le.transform([val])[0]
    except ValueError:
        return -1

# ===============================
# 🏠 HOME PAGE
# ===============================
if page == "🏠 Home":
    st.markdown('<div class="main-content">', unsafe_allow_html=True)
    st.title("❤️ Heart Disease Prediction App")

    # About
    st.markdown('<div class="card" style="background-color:#ffe6e6;">', unsafe_allow_html=True)
    st.subheader("💡 What is Heart Disease?")
    st.write("Heart disease refers to several conditions affecting the heart and blood vessels, "
             "such as coronary artery disease, arrhythmia, and heart failure.")
    st.write("It reduces oxygen supply to the body and can be fatal if not managed.")
    st.markdown('</div>', unsafe_allow_html=True)

    # Risk Factors
    st.markdown('<div class="card" style="background-color:#e6f2ff;">', unsafe_allow_html=True)
    st.subheader("⚠️ Risk Factors")
    st.write("🩸 High Blood Pressure, 🧂 High Cholesterol, 🍔 Obesity, 🚭 Smoking, 🧪 Diabetes, 🛋️ Sedentary Lifestyle")
    st.markdown('</div>', unsafe_allow_html=True)

    # Dataset Pie Chart
    st.subheader("📊 Dataset Overview (Heart Disease Distribution)")
    disease_counts = df[target_col].apply(lambda x: 0 if x == 0 else 1).value_counts()
    labels = ["No Heart Disease", "Has Heart Disease"]
    colors = ["#8fd9a8", "#ff6b6b"]
    fig, ax = plt.subplots()
    ax.pie(disease_counts, labels=labels, autopct="%1.1f%%", startangle=90, colors=colors, textprops={'fontsize': 12})
    ax.axis("equal")
    plt.title("Heart Disease Distribution", fontsize=14)
    st.pyplot(fig)

    # Model Accuracy Chart
    st.subheader("🤖 Model Accuracy Comparison")
    fig2, ax2 = plt.subplots(figsize=(9, 5))
    ax2.bar(accuracies.keys(), accuracies.values(), color=["#5dade2", "#58d68d", "#f7b731", "#af7ac5"], width=0.5)
    ax2.set_ylim(0, 100)
    ax2.set_ylabel("Accuracy (%)")
    for bar in ax2.patches:
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height()+1,
                 f"{bar.get_height():.1f}%", ha="center", fontsize=10)
    st.pyplot(fig2)

    # Recall Chart
    st.subheader("🩺 Model Recall (Sensitivity) Comparison")
    fig3, ax3 = plt.subplots(figsize=(9, 5))
    ax3.bar(recalls.keys(), recalls.values(), color=["#3498db", "#2ecc71", "#f1c40f", "#9b59b6"], width=0.5)
    ax3.set_ylim(0, 100)
    ax3.set_ylabel("Recall (%)")
    for bar in ax3.patches:
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height()+1,
                 f"{bar.get_height():.1f}%", ha="center", fontsize=10)
    st.pyplot(fig3)

    st.success(f"🏆 Best model (highest recall): **{best_model}** "
               f"(Recall: {recalls[best_model]:.2f}%, Accuracy: {accuracies[best_model]:.2f}%)")
    st.markdown('</div>', unsafe_allow_html=True)

# ===============================
# 🔍 PREDICTION PAGE
# ===============================
elif page == "🔍 Prediction":
    st.markdown('<div class="main-content">', unsafe_allow_html=True)
    st.title("🔍 Predict Heart Disease Risk")
    st.warning("⚠️ This prediction is for educational purposes only. Not a medical diagnosis.")

    algo = st.selectbox("🛠 Select Algorithm", list(models.keys()))

    age = st.slider("👤 Age", 20, 100, 45)
    sex = st.radio("⚧ Sex", le_dict["sex"].classes_ if "sex" in le_dict else ["Male", "Female"])
    cp = st.selectbox("💔 Chest Pain Type", le_dict["cp"].classes_ if "cp" in le_dict else [0, 1, 2, 3])
    trestbps = st.slider("🩸 Resting Blood Pressure (mm Hg)", 80, 200, 120)
    chol = st.slider("🥓 Cholesterol (mg/dl)", 100, 600, 200)
    thalch = st.slider("🏃 Max Heart Rate Achieved", 60, 220, 150)
    exang = st.selectbox("😮 Exercise Induced Angina?", ["No", "Yes"])

    sex_val = safe_transform(le_dict.get("sex", LabelEncoder().fit(["Male", "Female"])), sex)
    cp_val = safe_transform(le_dict.get("cp", LabelEncoder().fit([0, 1, 2, 3])), cp)
    exang_val = 1 if exang == "Yes" else 0

    input_data = np.array([[age, sex_val, cp_val, trestbps, chol, thalch, exang_val]])
    input_scaled = scaler.transform(input_data)

    if st.button("🔎 Predict"):
        model = models[algo]
        prediction = model.predict(input_scaled)[0]

        # Probability check (only for models with predict_proba)
        if hasattr(model, "predict_proba"):
            pred_prob = model.predict_proba(input_scaled)[0][1] * 100
        else:
            pred_prob = 50.0  # fallback

        pred_prob = np.clip(pred_prob, 0.1, 99.9)

        if prediction == 1:
            st.error(f"⚠️ High risk of Heart Disease ({pred_prob:.1f}%) using {algo}")
        else:
            st.success(f"✅ Low risk of Heart Disease ({100 - pred_prob:.1f}%) using {algo}")

        st.info(f"Model Accuracy: {accuracies[algo]:.2f}% | Macro Recall: {recalls[algo]:.2f}%")

    st.markdown('</div>', unsafe_allow_html=True)

# ===============================
# 💬 VOICE CHATBOT PAGE
# ===============================
elif page == "💬 Voice Chatbot":
    st.markdown('<div class="main-content">', unsafe_allow_html=True)
    st.title("💬 Heart Disease Voice Chatbot")

    faq = {
        "symptoms": "Common symptoms include chest pain, shortness of breath, fatigue, and irregular heartbeat.",
        "causes": "High blood pressure, cholesterol, diabetes, smoking, stress, and genetics are key causes.",
        "prevention": "Eat healthy, exercise regularly, quit smoking, and manage stress.",
        "treatment": "Depends on severity—may include medication, lifestyle change, angioplasty, or surgery.",
        "diet": "Focus on fruits, vegetables, lean protein, and low-salt, low-fat foods.",
        "risk factors": "Include age, blood pressure, diabetes, and family history."
    }

    synonyms = {
        "prevention": ["prevent", "avoid", "reduce risk", "protection"],
        "symptoms": ["signs", "indications", "warning"],
        "causes": ["reason", "why", "factor"],
        "treatment": ["cure", "therapy", "medicine"],
        "diet": ["food", "nutrition", "meal"],
        "risk factors": ["risk", "danger", "likelihood"]
    }

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_input = st.text_input("Ask your question:")

    if st.button("Ask"):
        if user_input.strip():
            st.session_state.chat_history.append(("You", user_input))
            text = user_input.lower().strip()
            response = "🤔 I'm not sure. Please consult a doctor."

            all_keys, key_map = [], {}
            for key, syns in synonyms.items():
                all_keys.append(key)
                key_map[key] = key
                for s in syns:
                    all_keys.append(s)
                    key_map[s] = key

            matched, score, _ = process.extractOne(text, all_keys, scorer=fuzz.token_sort_ratio)
            if score >= 60:
                response = faq[key_map[matched]]

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
