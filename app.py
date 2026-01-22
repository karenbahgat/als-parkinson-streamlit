import streamlit as st
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
import joblib
import os
from scipy.signal import butter, lfilter

# ===============================
# Constants
# ===============================
TARGET_SR = 16000
FIXED_DUR = 2.5
N_MFCC = 20

# ===============================
# Load model
# ===============================
@st.cache_resource
def load_artifacts():
    if not os.path.exists("xgb_model.joblib"):
        st.error("❌ Model file xgb_model.joblib not found!")
        st.stop()
    artifact = joblib.load("xgb_model.joblib")
    return artifact["model"], artifact["label_encoder"], artifact["top_features"]

model, label_encoder, top_features = load_artifacts()

# ===============================
# Audio preprocessing
# ===============================
def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype="high", analog=False)
    return b, a

def highpass_filter(data, cutoff=60, fs=TARGET_SR, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    return lfilter(b, a, data)

def fix_length_center(y, sr, fixed_dur):
    L = int(round(fixed_dur * sr))
    if len(y) < L:
        return np.pad(y, (0, L-len(y)))
    start = (len(y) - L) // 2
    return y[start:start+L]

def preprocess_audio(y, sr):
    if y.ndim > 1:
        y = np.mean(y, axis=1)
    if sr != TARGET_SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=TARGET_SR)
        sr = TARGET_SR
    y, _ = librosa.effects.trim(y, top_db=25)
    y = highpass_filter(y, fs=sr)
    if np.max(np.abs(y)) > 0:
        y = y / np.max(np.abs(y))
    y = fix_length_center(y, sr, FIXED_DUR)
    return y

# ===============================
# Feature extraction
# ===============================
def extract_features(y, sr=TARGET_SR):
    features = {}
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    delta1 = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    for i in range(N_MFCC):
        features[f"mfcc_{i+1}_mean"] = np.mean(mfcc[i])
        features[f"mfcc_{i+1}_std"] = np.std(mfcc[i])
        features[f"delta_mfcc_{i+1}_mean"] = np.mean(delta1[i])
        features[f"delta_mfcc_{i+1}_std"] = np.std(delta1[i])
        features[f"delta2_mfcc_{i+1}_mean"] = np.mean(delta2[i])
        features[f"delta2_mfcc_{i+1}_std"] = np.std(delta2[i])
    f0, _, _ = librosa.pyin(y, fmin=70, fmax=400)
    f0 = f0[~np.isnan(f0)]
    features["pitch_mean"] = np.mean(f0) if len(f0) else 0
    features["pitch_std"] = np.std(f0) if len(f0) else 0
    features["rms_energy"] = np.mean(librosa.feature.rms(y=y))
    return features

# ===============================
# Streamlit UI
# ===============================
st.set_page_config(page_title="Speech Disorder Classifier", layout="centered")
st.title("🗣️ Speech Disorder Classifier")

patient_name = st.text_input("Patient Name")
uploaded_file = st.file_uploader("Upload WAV file", type=["wav"])

if uploaded_file is not None:
    y, sr = sf.read(uploaded_file)
    duration = len(y)/sr
    st.write(f"Audio duration: {duration:.2f} seconds")
    if duration < 1.25:
        st.warning(f"⚠️ Audio too short ({duration:.2f}s). Results may be unreliable.")
    
    with st.spinner("Processing audio..."):
        y = preprocess_audio(y, sr)
        features = extract_features(y)
        X = pd.DataFrame([features])
        X = X[top_features]
        probs = model.predict_proba(X)[0]
        pred_idx = np.argmax(probs)
        pred_label = label_encoder.inverse_transform([pred_idx])[0]

    st.success(f"### 🧠 Prediction for {patient_name if patient_name else 'Patient'}: **{pred_label}**")

    # Probability bars with colors
    prob_df = pd.DataFrame({
        "Class": label_encoder.classes_,
        "Probability": probs
    })
    prob_df = prob_df.set_index("Class")
    st.bar_chart(prob_df)

    # Show exact probabilities
    st.subheader("Probabilities")
    for c, p in zip(label_encoder.classes_, probs):
        st.write(f"{c}: {p*100:.2f}%")
