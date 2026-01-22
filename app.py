import streamlit as st
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
import joblib
from scipy.signal import butter, lfilter

# ===============================
# Constants (MUST match training)
# ===============================
TARGET_SR = 16000
FIXED_DURATION = 2.5  # seconds
N_MFCC = 20

# ===============================
# Load model
# ===============================
@st.cache_resource
def load_artifacts():
    artifact = joblib.load("xgb_model.joblib")
    return (
        artifact["model"],
        artifact["label_encoder"],
        artifact["top_features"]
    )

model, label_encoder, top_features = load_artifacts()

# ===============================
# Audio preprocessing
# ===============================
def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype="high", analog=False)
    return b, a

def highpass_filter(data, cutoff=60, fs=16000, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    return lfilter(b, a, data)

def preprocess_audio(y, sr):
    # Stereo → Mono
    if y.ndim > 1:
        y = np.mean(y, axis=1)

    # Resample
    if sr != TARGET_SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=TARGET_SR)
        sr = TARGET_SR

    # Trim silence
    y, _ = librosa.effects.trim(y, top_db=25)

    # High-pass filter
    y = highpass_filter(y, fs=sr)

    # Normalize [-1, 1]
    if np.max(np.abs(y)) > 0:
        y = y / np.max(np.abs(y))

    # Fix length (pad / truncate)
    target_len = int(FIXED_DURATION * sr)
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    else:
        y = y[:target_len]

    return y

# ===============================
# Feature extraction
# ===============================
def extract_features(y, sr=TARGET_SR):
    features = {}

    # MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    for i in range(N_MFCC):
        features[f"mfcc_{i+1}_mean"] = np.mean(mfcc[i])
        features[f"mfcc_{i+1}_std"] = np.std(mfcc[i])

    # Delta MFCC
    delta = librosa.feature.delta(mfcc)
    for i in range(N_MFCC):
        features[f"delta_mfcc_{i+1}_mean"] = np.mean(delta[i])
        features[f"delta_mfcc_{i+1}_std"] = np.std(delta[i])

    # Pitch (F0)
    f0, _, _ = librosa.pyin(
        y,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C7")
    )
    f0 = f0[~np.isnan(f0)]
    features["pitch_mean"] = np.mean(f0) if len(f0) else 0
    features["pitch_std"] = np.std(f0) if len(f0) else 0

    # Spectral features
    features["spectral_centroid"] = np.mean(
        librosa.feature.spectral_centroid(y=y, sr=sr)
    )
    features["spectral_bandwidth"] = np.mean(
        librosa.feature.spectral_bandwidth(y=y, sr=sr)
    )
    features["spectral_rolloff"] = np.mean(
        librosa.feature.spectral_rolloff(y=y, sr=sr)
    )

    # Energy
    features["rms_energy"] = np.mean(librosa.feature.rms(y=y))

    return features

# ===============================
# Streamlit UI
# ===============================
st.set_page_config(page_title="Speech Disorder Classification", layout="centered")

st.title("🗣️ Speech Disorder Classification")
st.write("Upload a WAV file to classify speech condition")

uploaded_file = st.file_uploader("Upload WAV file", type=["wav"])

if uploaded_file is not None:
    with st.spinner("Processing audio..."):
        y, sr = sf.read(uploaded_file)
        y = preprocess_audio(y, sr)
        features = extract_features(y)

        X = pd.DataFrame([features])
        X = X[top_features]  # IMPORTANT: same feature order

        prediction = model.predict(X)
        label = label_encoder.inverse_transform(prediction)[0]

    st.success(f"### 🧠 Prediction: **{label}**")
