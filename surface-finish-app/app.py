import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
import io

st.set_page_config(page_title="Surface Finish AI System", layout="wide")

st.title("🔥 Smart Surface Finish Monitoring System (AI + Vibration Analysis)")

# ---------------- FFT ----------------
def compute_fft(signal, fs):
    n = len(signal)
    freq = np.fft.rfftfreq(n, d=1/fs)
    fft_vals = np.abs(np.fft.rfft(signal))
    return freq, fft_vals

# ---------------- Features ----------------
def extract_features(signal, fs):
    freq, fft_vals = compute_fft(signal, fs)

    peak = np.max(fft_vals)
    dom_freq = freq[np.argmax(fft_vals)]
    rms = np.sqrt(np.mean(signal**2))
    kurt = np.mean((signal - np.mean(signal))**4) / (np.std(signal)**4 + 1e-9)
    crest = peak / (rms + 1e-9)

    return {
        "Peak": peak,
        "Dominant Frequency": dom_freq,
        "RMS": rms,
        "Kurtosis": kurt,
        "Crest Factor": crest
    }

# ---------------- ML Model ----------------
def train_model(X, y):
    model = RandomForestRegressor(n_estimators=100)
    model.fit(X, y)
    return model

# ---------------- PDF ----------------
def create_pdf(text):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer)

    styles = getSampleStyleSheet()
    content = []

    for line in text:
        content.append(Paragraph(line, styles["Normal"]))
        content.append(Spacer(1, 10))

    doc.build(content)
    buffer.seek(0)
    return buffer

# ---------------- UI ----------------
file = st.file_uploader("Upload CSV (first column = vibration signal)", type=["csv"])
fs = st.number_input("Sampling Frequency", value=5000)

if file:
    data = pd.read_csv(file)
    signal = data.iloc[:, 0].values

    st.subheader("📊 Time Signal")
    fig1 = plt.figure()
    plt.plot(signal)
    st.pyplot(fig1)

    freq, fft_vals = compute_fft(signal, fs)

    st.subheader("📈 FFT Spectrum")
    fig2 = plt.figure()
    plt.plot(freq, fft_vals)
    st.pyplot(fig2)

    features = extract_features(signal, fs)

    st.subheader("🧠 Extracted Features")
    st.json(features)

    # ---------------- Rule-based Classification ----------------
    score = 0
    if features["Peak"] < 80:
        score += 30
    if features["Dominant Frequency"] < 55:
        score += 25
    if features["Kurtosis"] < 5:
        score += 20
    if features["Crest Factor"] < 10:
        score += 25

    if score > 80:
        label = "Excellent (Grinding)"
    elif score > 60:
        label = "Good"
    elif score > 40:
        label = "Medium"
    else:
        label = "Poor (Turning)"

    st.subheader("📌 Surface Classification")
    st.success(f"{label} | Score: {score}/100")

    # ---------------- AI Model (if Ra exists) ----------------
    if "Ra" in data.columns:
        st.subheader("🤖 AI Model (Ra Prediction)")

        X = data.drop(columns=["Ra"]).values
        y = data["Ra"].values

        model = train_model(X, y)
        pred = model.predict([features.values()])[0]

        st.info(f"Predicted Surface Roughness (Ra): {pred:.4f} µm")

    # ---------------- Report ----------------
    st.subheader("📄 Generate Report")

    if st.button("Download PDF Report"):
        text = [
            "Surface Finish Analysis Report",
            f"Classification: {label}",
            f"Score: {score}/100",
            f"Peak: {features['Peak']}",
            f"Dominant Frequency: {features['Dominant Frequency']}",
            f"RMS: {features['RMS']}"
        ]

        pdf = create_pdf(text)

        st.download_button(
            label="Download Report",
            data=pdf,
            file_name="surface_report.pdf",
            mime="application/pdf"
        )
