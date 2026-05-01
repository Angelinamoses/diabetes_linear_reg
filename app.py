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

# ------------------ USER INPUT ------------------
st.subheader("🧍 Basic Info")

age = st.slider("Age", 10, 80, 25)
gender = st.radio("Gender", ["Female", "Male"])
height = st.number_input("Height (cm)", 100, 220, 165)
weight = st.number_input("Weight (kg)", 30, 150, 60)

# BMI calculation
bmi = weight / ((height / 100) ** 2)
st.write(f"📊 Calculated BMI: **{bmi:.2f}**")

st.markdown("---")

st.subheader("🩺 Health Info")

bp = st.selectbox("Blood Pressure Level", ["Normal", "High"])
glucose = st.number_input("Blood Sugar Level", 70, 200, 100)

family_history = st.selectbox("Family history of diabetes?", ["No", "Yes"])
activity = st.selectbox("Physical Activity Level", ["Low", "Medium", "High"])

st.markdown("---")

# ------------------ CONVERSIONS ------------------

gender = 1 if gender == "Male" else 0
bp = 120 if bp == "Normal" else 150
family_history = 1.0 if family_history == "Yes" else 0.3

# approximate placeholders (model requirement)
insulin = 80
skin = 20
pedigree = family_history
pregnancies = 0
cholesterol = 180

# adjust based on activity (simple logic)
if activity == "Low":
    insulin += 20
elif activity == "High":
    insulin -= 10

# ------------------ FINAL FEATURE VECTOR ------------------
features = [
    age, gender, bmi, bp, cholesterol,
    insulin, glucose, skin, pedigree, pregnancies
]

# ------------------ PREDICT ------------------
if st.button("🔍 Check Diabetes Risk"):

    prediction = model.predict([features])[0]

    st.subheader("📈 Result")

    # normalize weird values (since regression output can be large)
    score = max(0, min(int(prediction), 200))

    st.progress(score)

    if score < 100:
        st.success(f"✅ Low Risk ({score})")
    elif score < 140:
        st.warning(f"⚠️ Moderate Risk ({score})")
    else:
        st.error(f"🚨 High Risk ({score})")

    # ------------------ INSIGHTS ------------------
    st.subheader("💡 Quick Health Tips")

    tips = []

    if bmi > 25:
        tips.append("⚖️ Try maintaining a healthy weight")
    if glucose > 120:
        tips.append("🍭 Reduce sugar intake")
    if activity == "Low":
        tips.append("🏃 Increase physical activity")
    if bp == 150:
        tips.append("🫀 Monitor blood pressure regularly")

    if tips:
        for tip in tips:
            st.write(tip)
    else:
        st.write("🎉 You're doing well. Keep it up!")

    # ------------------ CHART ------------------
    st.subheader("📊 Your Health Snapshot")

    chart_data = {
        "Age": age,
        "BMI": bmi,
        "Glucose": glucose,
        "BP": bp,
        "Insulin": insulin
    }

    st.bar_chart(chart_data)

# ------------------ SIDEBAR ------------------
st.sidebar.title("ℹ️ About")

st.sidebar.write("""
This app estimates diabetes risk using a machine learning model.

⚠️ Not a medical diagnosis.
""")

st.sidebar.markdown("---")
st.sidebar.write("Made by Angelina ✨")