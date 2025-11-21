import streamlit as st
import numpy as np
import joblib

st.set_page_config(page_title="⚡ Early Outage Prediction", page_icon="⚡", layout="wide")

# Load model + scaler
model = joblib.load("my_model.pkl")
scaler = joblib.load("scaler.pkl")

# Page title & description
st.markdown(
    """
    <h1 style='text-align:center; color:#FF4B4B;'>⚡ Early Outage Prediction System</h1>
    <p style='text-align:center; font-size:18px; color:#555;'>Predict potential system outages before they happen. Fill in the parameters below.</p>
    """,
    unsafe_allow_html=True
)

# Use columns for neat layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("System Conditions")
    temperature = st.slider("🌡️ Temperature (°C)", 0, 100, 45)
    load = st.number_input("⚡ Load (kW)", value=78.0)
    humidity = st.slider("💧 Humidity (%)", 0, 100, 88)
    vibration = st.slider("🔧 Vibration Level", 0.0, 5.0, 0.84, 0.01)
    cycles = st.number_input("🔄 Operational Cycles", value=530)

with col2:
    st.subheader("Operational Status")
    runtime = st.number_input("⏱️ Runtime Hours", value=3012.0)
    is_peak_hour = st.radio("🌇 Peak Hour?", ["No", "Yes"], index=0, horizontal=True)
    is_backup_active = st.radio("🔋 Backup System Active?", ["No", "Yes"], index=0, horizontal=True)
    weather_severity = st.slider("⛈️ Weather Severity (0 - 1)", 0.0, 1.0, 0.79, 0.01)
    wear_and_tear = st.slider("⚙️ Wear Level (0 - 1)", 0.0, 1.0, 0.91, 0.01)

# Convert Yes/No to numerical
is_peak_hour = 1 if is_peak_hour == "Yes" else 0
is_backup_active = 1 if is_backup_active == "Yes" else 0

# Predict button
if st.button("🚀 Predict Outage Risk"):
    X = np.array([[temperature, load, humidity, vibration, cycles, runtime,
                   is_peak_hour, is_backup_active, weather_severity, wear_and_tear]])
    X_scaled = scaler.transform(X)
    prediction = model.predict(X_scaled)[0]
    prob = model.predict_proba(X_scaled)[0][1]

    # Result card
    if prediction == 1:
        st.markdown(
            f"""
            <div style="background-color:#FF4B4B; padding:20px; border-radius:15px; color:white; text-align:center">
                <h2>⚠️ HIGH Outage Risk!</h2>
                <p style='font-size:18px;'>Probability: {prob:.2f}</p>
            </div>
            """, unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"""
            <div style="background-color:#4CAF50; padding:20px; border-radius:15px; color:white; text-align:center">
                <h2>🟢 System Stable</h2>
                <p style='font-size:18px;'>Low Outage Risk. Probability: {prob:.2f}</p>
            </div>
            """, unsafe_allow_html=True
        )

    # Bonus: show a risk meter
    st.markdown("### Risk Meter")
    st.progress(int(prob * 100))