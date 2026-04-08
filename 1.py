import os
import pickle
import streamlit as st
from streamlit_option_menu import option_menu
import google.generativeai as genai
from dotenv import load_dotenv
import numpy as np
import pandas as pd

# -------------------- CONFIG --------------------
st.set_page_config(page_title="AI Health Predictor", layout="wide")

def save_history(data):
    file = "history.csv"

    # Ensure consistent columns
    required_cols = ["Type", "Result"]

    for col in required_cols:
        if col not in data:
            data[col] = None

    new_df = pd.DataFrame([data])

    if os.path.exists(file):
        old_df = pd.read_csv(file)
        new_df = pd.concat([old_df, new_df], ignore_index=True)

    new_df.to_csv(file, index=False)

st.markdown("""
<style>
    .main {
        background-color: #0e1117;
        color: white;
    }

    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 10px;
        height: 3em;
        width: 100%;
    }

    .stSlider label {
        color: #ddd;
    }

    .css-1d391kg {
        background-color: #111;
    }

    h1, h2, h3 {
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# -------------------- SIDEBAR --------------------
with st.sidebar:
    selected = option_menu(
    "Menu",
    ["Home", "Diabetes", "Heart", "Common Disease", "History"],
    icons=["house","activity","heart","stethoscope","clock"]
    )

# =====================================================
# 🏠 HOME
# =====================================================
if selected == "Home":
    st.markdown("""
    <h1>🧑‍⚕️ AI Health Prediction System</h1>
    <p style='text-align:center; font-size:18px;'>
    Predict diseases using Machine Learning & get AI-powered health advice
    </p>
    """, unsafe_allow_html=True)

    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 🩺 Diabetes Prediction")
        st.write("Check diabetes risk using medical parameters")

    with col2:
        st.markdown("### ❤️ Heart Disease")
        st.write("Analyze heart disease risk factors")

    with col3:
        st.markdown("### 🤒 General Disease")
        st.write("Predict disease based on symptoms")

    st.markdown("---")

    st.info("⚠️ This system is for educational purposes only. Consult a doctor for medical advice.")

# -------------------- HEADER --------------------
st.markdown("""
<h1 style='text-align: center; color: #4CAF50;'>🧑‍⚕️ AI Health Prediction System</h1>
<p style='text-align: center;'>Predict diseases & get AI-powered advice</p>
""", unsafe_allow_html=True)

# -------------------- ENV --------------------
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")

if not API_KEY:
    st.error("Missing GOOGLE_API_KEY")
    st.stop()

genai.configure(api_key=API_KEY)
model_gen = genai.GenerativeModel("gemini-flash-latest")

def get_ai_response(prompt):
    try:
        return model_gen.generate_content(prompt).text
    except:
        return "AI Error"

# -------------------- LOAD MODELS --------------------
working_dir = os.path.dirname(os.path.abspath(__file__))

diabetes_model = pickle.load(open(os.path.join(working_dir, 'saved_models/diabetes_model.sav'), 'rb'))
heart_model = pickle.load(open(os.path.join(working_dir, 'saved_models/rf_classifier.pkl'), 'rb'))
scaler = pickle.load(open(os.path.join(working_dir, 'saved_models/scaler.pkl'), 'rb'))

disease_model = pickle.load(open(os.path.join(working_dir, 'saved_models/disease_model.pkl'), 'rb'))
label_encoder = pickle.load(open(os.path.join(working_dir, 'saved_models/label_encoder.pkl'), 'rb'))
symptoms_list = pickle.load(open(os.path.join(working_dir, 'saved_models/symptoms.pkl'), 'rb'))

# =====================================================
# 📜 HISTORY
# =====================================================
if selected == "History":
    st.subheader("📜 Prediction History")

    file = "history.csv"

    if os.path.exists(file):
        df = pd.read_csv(file)
        st.dataframe(df, use_container_width=True)

        col1, col2 = st.columns(2)

        with col1:
            if st.button("🗑️ Clear History"):
                os.remove(file)
                st.success("History cleared!")
                st.experimental_rerun()

        with col2:
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("⬇️ Download History", csv, "history.csv")

    else:
        st.warning("No history found")


# =====================================================
# 🩺 DIABETES
# =====================================================
if selected == "Diabetes":
    st.subheader("🩺 Diabetes Prediction")

    col1, col2 = st.columns(2)

    with col1:
        Glucose = st.slider("Glucose", 70, 300)
        BMI = st.slider("BMI", 15.0, 50.0)
    with col2:
        Age = st.slider("Age", 1, 100)

    if st.button("Predict"):
        with st.spinner("Analyzing..."):
            data = [[0, Glucose, 80, 20, 80, BMI, 0.5, Age]]

            proba = diabetes_model.predict_proba(data)[0]
            pred = proba.argmax()
            confidence = round(max(proba)*100,2)

            st.info(f"Confidence: {confidence}%")

            if pred == 1:
                st.error("⚠️ Risk of Diabetes")
                prompt = f"Glucose {Glucose}, BMI {BMI}, Age {Age}. Give diet & precautions"
            else:
                st.success("✅ Healthy")

                prompt = "Give healthy lifestyle tips"

            save_history({"Type":"Diabetes","Result":pred})

            with st.expander("🤖 AI Advice"):
                st.write(get_ai_response(prompt))

# =====================================================
# ❤️ HEART
# =====================================================
if selected == "Heart":
    st.subheader("❤️ Heart Disease Prediction")

    col1, col2, col3 = st.columns(3)
    
    with col1:
        male = st.selectbox("Gender", ["male", "female"])
        male = 1 if male == "male" else 0

        age = st.slider("Age", 20, 100)

        currentSmoker = st.selectbox("Current Smoker", ["yes", "no"])
        currentSmoker = 1 if currentSmoker == "yes" else 0

        cigsPerDay = st.slider("Cigarettes per Day", 0, 50)

    with col2:
        BPMeds = st.selectbox("BP Medication", ["yes", "no"])
        BPMeds = 1 if BPMeds == "yes" else 0

        prevalentStroke = st.selectbox("Stroke History", ["yes", "no"])
        prevalentStroke = 1 if prevalentStroke == "yes" else 0

        prevalentHyp = st.selectbox("Hypertension", ["yes", "no"])
        prevalentHyp = 1 if prevalentHyp == "yes" else 0

        diabetes = st.selectbox("Diabetes", ["yes", "no"])
        diabetes = 1 if diabetes == "yes" else 0

    with col3:
        chol = st.slider("Cholesterol", 100, 400)
        sysBP = st.slider("Systolic BP", 80, 200)
        diaBP = st.slider("Diastolic BP", 50, 120)
        BMI = st.slider("BMI", 15.0, 40.0)
        heartRate = st.slider("Heart Rate", 40, 150)
        glucose = st.slider("Glucose", 50, 300)

    if st.button("Predict Heart"):
        with st.spinner("Analyzing..."):

            # ✅ Build input dictionary
            input_dict = {
                "male": male,
                "age": age,
                "currentSmoker": currentSmoker,
                "cigsPerDay": cigsPerDay,
                "BPMeds": BPMeds,
                "prevalentStroke": prevalentStroke,
                "prevalentHyp": prevalentHyp,
                "diabetes": diabetes,
                "totChol": chol,
                "sysBP": sysBP,
                "diaBP": diaBP,
                "BMI": BMI,
                "heartRate": heartRate,
                "glucose": glucose
            }

            # ✅ Load correct feature order (VERY IMPORTANT)
            try:
                heart_features = pickle.load(open("saved_models/heart_features.pkl", "rb"))
            except:
                st.error("❌ heart_features.pkl not found. Please retrain model.")
                st.stop()

            # ✅ Arrange data in correct order
            input_data = [input_dict[col] for col in heart_features]
            input_data = np.array(input_data).reshape(1, -1)

            # ✅ Scale
            input_data = scaler.transform(input_data)

            # ✅ Predict
            result = heart_model.predict(input_data)[0]

            # ✅ Confidence (if available)
            try:
                proba = heart_model.predict_proba(input_data)[0]
                confidence = round(max(proba) * 100, 2)
                st.info(f"Confidence: {confidence}%")
            except:
                pass

            # ✅ Output
            if result == 1:
                st.error("⚠️ Risk of Heart Disease")
                prompt = f"Patient risk detected. Age {age}, BP {sysBP}/{diaBP}, Cholesterol {chol}. Give precautions."
            else:
                st.success("✅ Healthy")
                prompt = "Give heart fitness and lifestyle advice."

            # ✅ Save history
            save_history({"Type": "Heart", "Result": result})

            # ✅ AI Advice
            with st.expander("🤖 AI Advice"):
                st.write(get_ai_response(prompt))

# =====================================================
# 🤒 COMMON DISEASE
# =====================================================
if selected == "Common Disease":
    st.subheader("🤒 Disease Prediction")

    selected_symptoms = st.multiselect("Select Symptoms", symptoms_list)

    if st.button("Predict Disease"):
        if not selected_symptoms:
            st.warning("Select symptoms")
        else:
            vector = [1 if s in selected_symptoms else 0 for s in symptoms_list]
            vector = np.array(vector).reshape(1,-1)

            pred = disease_model.predict(vector)
            disease = label_encoder.inverse_transform(pred)[0]

            st.success(f"Predicted: {disease}")

            save_history({"Type":"Common","Result":disease})

            with st.expander("🤖 AI Advice"):
                st.write(get_ai_response(f"Symptoms: {selected_symptoms}, Disease: {disease}"))

