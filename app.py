import streamlit as st
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
import joblib
import altair as alt
from scipy.signal import butter, lfilter
import tempfile
import os

# ===============================
# Constants (MUST match training)
# ===============================
TARGET_SR = 16000
FIXED_DURATION = 2.5
MIN_DURATION = 1.25
N_MFCC = 20

# ===============================
# Load model artifacts
# ===============================
@st.cache_resource
def load_artifacts():
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

def highpass_filter(data, cutoff=60, fs=16000, order=5):
    b, a = butter_highpass(cutoff, fs, order)
    return lfilter(b, a, data)

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

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    for i in range(N_MFCC):
        features[f"mfcc_{i+1}_mean"] = np.mean(mfcc[i])
        features[f"mfcc_{i+1}_std"] = np.std(mfcc[i])

    delta = librosa.feature.delta(mfcc)
    for i in range(N_MFCC):
        features[f"delta_mfcc_{i+1}_mean"] = np.mean(delta[i])
        features[f"delta_mfcc_{i+1}_std"] = np.std(delta[i])

    f0, _, _ = librosa.pyin(
        y,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C7")
    )
    f0 = f0[~np.isnan(f0)]
    features["pitch_mean"] = np.mean(f0) if len(f0) else 0
    features["pitch_std"] = np.std(f0) if len(f0) else 0

    features["spectral_centroid"] = np.mean(
        librosa.feature.spectral_centroid(y=y, sr=sr)
    )
    features["spectral_bandwidth"] = np.mean(
        librosa.feature.spectral_bandwidth(y=y, sr=sr)
    )
    features["spectral_rolloff"] = np.mean(
        librosa.feature.spectral_rolloff(y=y, sr=sr)
    )

    features["rms_energy"] = np.mean(librosa.feature.rms(y=y))

    return features

# ===============================
# Streamlit UI
# ===============================
st.set_page_config(
    page_title="Speech Disorder Screening Tool",
    layout="centered"
)

st.title("🗣️ Speech Disorder Screening Tool")
st.write("Upload speech recordings to estimate the probability of neurological speech disorders.")

patient_name = st.text_input("Patient Name")

uploaded_files = st.file_uploader(
    "Upload WAV files (minimum duration: 1.25 seconds)",
    type=["wav"],
    accept_multiple_files=True
)

if uploaded_files and patient_name:
    for file in uploaded_files:
        st.divider()
        st.subheader(f"🎧 File: {file.name}")

        with st.spinner("Processing audio..."):
            y, sr = sf.read(file)
            duration = len(y) / sr

            if duration < MIN_DURATION:
                st.error(
                    f"Audio duration is {duration:.2f}s. "
                    f"Minimum required duration is {MIN_DURATION}s."
                )
                continue

            y = preprocess_audio(y, sr)
            features = extract_features(y)

            X = pd.DataFrame([features])
            X = X.reindex(columns=top_features, fill_value=0.0)

            probs = model.predict_proba(X)[0]
            classes = label_encoder.classes_

            prob_df = pd.DataFrame({
                "Condition": classes,
                "Probability (%)": np.round(probs * 100, 2)
            })

            prediction = classes[np.argmax(probs)]

        st.success(f"🧠 **Prediction:** {prediction}")
        st.write(f"**Patient:** {patient_name}")

        st.dataframe(prob_df, use_container_width=True)

        chart = (
            alt.Chart(prob_df)
            .mark_bar()
            .encode(
                x=alt.X("Condition", sort=None),
                y=alt.Y("Probability (%)", scale=alt.Scale(domain=[0, 100])),
                tooltip=["Condition", "Probability (%)"]
            )
            .properties(height=300)
        )

        st.altair_chart(chart, use_container_width=True)

st.caption(
    "⚠️ This application is intended for research and screening purposes only. "
    "It does not provide a medical diagnosis."
)
