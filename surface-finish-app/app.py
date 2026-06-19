import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.title("Smart Surface Finish Monitoring System (Advanced)")

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
    kurtosis = np.mean((signal - np.mean(signal))**4) / (np.std(signal)**4 + 1e-9)
    crest = peak / (rms + 1e-9)

    return {
        "Peak": peak,
        "Dominant Frequency": dom_freq,
        "RMS": rms,
        "Kurtosis": kurtosis,
        "Crest Factor": crest
    }

# ---------------- Decision Model ----------------
def classify(features):
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
        return "Excellent (Grinding)", score
    elif score > 60:
        return "Good", score
    elif score > 40:
        return "Medium", score
    else:
        return "Poor (Turning-like)", score

# ---------------- UI ----------------
file = st.file_uploader("Upload Vibration CSV", type=["csv"])
fs = st.number_input("Sampling Frequency", value=5000)

if file:
    data = pd.read_csv(file)
    signal = data.iloc[:, 0].values

    features = extract_features(signal, fs)
    result, score = classify(features)

    st.subheader("Features")
    st.json(features)

    st.subheader("Decision")
    st.success(f"{result} | Confidence Score: {score}%")

    # -------- Time Signal --------
    st.subheader("Time Signal")
    plt.figure()
    plt.plot(signal)
    st.pyplot(plt)

    # -------- FFT --------
    freq, fft_vals = compute_fft(signal, fs)
    plt.figure()
    plt.plot(freq, fft_vals)
    plt.title("FFT Spectrum")
    st.pyplot(plt)
