import streamlit as st
import numpy as np
import joblib

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="Diabetes Risk Checker",
    page_icon="🧠",
    layout="centered"
)

# ------------------ LOAD MODEL ------------------
try:
    model = joblib.load("regmodel.pkl")
except:
    st.error("❌ Model file not found. Keep regmodel.pkl in same folder.")
    st.stop()

# ------------------ TITLE ------------------
st.title("🧠 Diabetes Risk Checker")
st.markdown("### Simple health check (not medical advice)")

st.markdown("---")

# ------------------ INPUTS ------------------

st.subheader("🧍 Basic Info")

age = st.number_input("Age", 10, 80, 25)
gender = st.radio("Gender", ["Female", "Male", "Other"])
height = st.number_input("Height (cm)", 100, 220, 165)
weight = st.number_input("Weight (kg)", 30, 150, 60)

st.markdown("---")

st.subheader("🩺 Health Info")

bp = st.selectbox("Blood Pressure Level", ["Normal", "High"])
family_history = st.selectbox("Family history of diabetes?", ["No", "Yes"])
activity = st.selectbox("Physical Activity Level", ["Low", "Medium", "High"])

st.markdown("---")

st.subheader("🧪 Clinical Inputs")

fasting_glucose = st.number_input("Fasting Blood Sugar (mg/dL)", 70, 200, 100)
post_glucose = st.number_input("After-meal Blood Sugar (mg/dL)", 90, 300, 140)

urination = st.selectbox("Frequent Urination?", ["No", "Yes"])
wound = st.selectbox("Slow Wound Healing?", ["No", "Yes"])

st.markdown("---")

# ------------------ BUTTON ------------------

if st.button("🔍 Check Diabetes Risk"):

    # BMI
    bmi = weight / ((height / 100) ** 2)

    # conversions
    gender_val = 1 if gender == "Male" else 0
    bp_val = 120 if bp == "Normal" else 150
    family_val = 1.0 if family_history == "Yes" else 0.3

    # placeholders (model expects these)
    insulin = 80
    if activity == "Low":
        insulin += 20
    elif activity == "High":
        insulin -= 10

    skin = 20
    cholesterol = 180
    pregnancies = 0

    # ------------------ FEATURE VECTOR ------------------
    features = [
        age, gender_val, bmi, bp_val, cholesterol,
        insulin, fasting_glucose, skin, family_val, pregnancies
    ]

    # ------------------ ML PREDICTION ------------------
    prediction = model.predict([features])[0]
    score = max(0, min(int(prediction), 200))

    # ------------------ CLINICAL RULES ------------------
    risk_points = 0

    # WHO-style glucose thresholds
    if fasting_glucose >= 126:
        risk_points += 2
    elif fasting_glucose >= 100:
        risk_points += 1

    if post_glucose >= 200:
        risk_points += 2
    elif post_glucose >= 140:
        risk_points += 1

    # symptoms
    if urination == "Yes":
        risk_points += 1
    if wound == "Yes":
        risk_points += 1

    # combine ML + clinical
    final_score = score + (risk_points * 20)

    # ------------------ OUTPUT ------------------

    st.subheader("📈 Result")

    st.progress(min(final_score, 200))

    if final_score < 80:
        st.success(f"✅ Low Risk ({final_score})")
    elif final_score < 130:
        st.warning(f"⚠️ Moderate Risk ({final_score})")
    else:
        st.error(f"🚨 High Risk ({final_score})")

    # ------------------ DETAILS ------------------

    st.subheader("📊 Health Summary")

    st.write(f"📏 BMI: {bmi:.2f}")
    st.write(f"🧪 Fasting Glucose: {fasting_glucose}")
    st.write(f"🍽️ Post-meal Glucose: {post_glucose}")

    # ------------------ INSIGHTS ------------------

    st.subheader("💡 Quick Health Tips")

    tips = []

    if bmi > 25:
        tips.append("⚖️ Maintain a healthy weight")
    if fasting_glucose > 120:
        tips.append("🍭 Reduce sugar intake")
    if activity == "Low":
        tips.append("🏃 Increase physical activity")
    if bp_val == 150:
        tips.append("🫀 Monitor blood pressure")
    if urination == "Yes":
        tips.append("🚰 Check hydration and glucose levels")
    if wound == "Yes":
        tips.append("🩹 Monitor healing, consult a doctor if slow")

    if tips:
        for tip in tips:
            st.write(tip)
    else:
        st.write("🎉 You're doing well. Keep it up!")

    # ------------------ CHART ------------------

    st.subheader("📊 Health Snapshot")

    st.bar_chart({
        "BMI": bmi,
        "Glucose": fasting_glucose,
        "BP": bp_val,
        "Insulin": insulin
    })

# ------------------ SIDEBAR ------------------

st.sidebar.title("ℹ️ About")

st.sidebar.write("""
This app estimates diabetes risk using:
- Machine Learning model (Linear Regression)
- Clinical rules (WHO-style thresholds)

⚠️ Not a medical diagnosis.
""")

st.sidebar.markdown("---")
st.sidebar.write("Made by Angelina ✨")