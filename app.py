if st.button("🔍 Check Diabetes Risk"):

    # BMI
    bmi = weight / ((height / 100) ** 2)

    # conversions
    gender_val = 1 if gender == "Male" else 0
    bp_val = 120 if bp == "Normal" else 150
    family_val = 1.0 if family_history == "Yes" else 0.3

    insulin = 80
    if activity == "Low":
        insulin += 20
    elif activity == "High":
        insulin -= 10

    # feature vector (IMPORTANT ORDER)
    features = [
        age, gender_val, bmi, bp_val, 180,
        insulin, fasting_glucose, 20, family_val, 0
    ]

    # ML prediction
    prediction = model.predict([features])[0]
    score = max(0, min(int(prediction), 200))

    # ------------------ CLINICAL LOGIC ------------------

    risk_points = 0

    if fasting_glucose >= 126:
        risk_points += 2
    elif fasting_glucose >= 100:
        risk_points += 1

    if post_glucose >= 200:
        risk_points += 2
    elif post_glucose >= 140:
        risk_points += 1

    if urination == "Yes":
        risk_points += 1

    if wound == "Yes":
        risk_points += 1

    final_score = score + (risk_points * 20)

    # ------------------ OUTPUT ------------------

    st.subheader("📈 Result")

    st.progress(final_score)

    if final_score < 80:
        st.success(f"✅ Low Risk ({final_score})")
    elif final_score < 130:
        st.warning(f"⚠️ Moderate Risk ({final_score})")
    else:
        st.error(f"🚨 High Risk ({final_score})")

    # ------------------ INSIGHTS ------------------

    st.subheader("💡 Quick Health Tips")

    tips = []

    if bmi > 25:
        tips.append("⚖️ Maintain healthy weight")
    if fasting_glucose > 120:
        tips.append("🍭 Reduce sugar intake")
    if activity == "Low":
        tips.append("🏃 Increase physical activity")
    if bp_val == 150:
        tips.append("🫀 Monitor blood pressure")

    for tip in tips:
        st.write(tip)

    # chart
    st.subheader("📊 Health Snapshot")
    st.bar_chart({
        "BMI": bmi,
        "Glucose": fasting_glucose,
        "BP": bp_val,
        "Insulin": insulin
    })