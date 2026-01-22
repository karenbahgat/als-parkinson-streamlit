import streamlit as st
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
import joblib
from pathlib import Path
from scipy.stats import skew, kurtosis

# ===============================
# Constants
# ===============================
TARGET_SR = 16000
FIXED_DUR = 2.5
N_MELS    = 80
N_FFT     = 1024
HOP       = 256
FMIN      = 50
FMAX      = TARGET_SR // 2
N_MFCC    = 20

WAV_DIR = Path("processed") / "wav_clean"
MEL_DIR = Path("processed") / "logmel_clean"
WAV_DIR.mkdir(parents=True, exist_ok=True)
MEL_DIR.mkdir(parents=True, exist_ok=True)

# ===============================
# Load model artifacts
# ===============================
@st.cache_resource
def load_artifacts():
    artifact = joblib.load("xgb_model.joblib")
    return artifact["model"], artifact["label_encoder"], artifact["top_features"]

model, label_encoder, top_features = load_artifacts()

# ===============================
# Audio preprocessing functions
# ===============================
def fix_length_center(y: np.ndarray, sr: int, fixed_dur: float) -> np.ndarray:
    L = int(round(fixed_dur * sr))
    if len(y) < L:
        return np.pad(y, (0, L-len(y)))
    start = (len(y) - L) // 2
    return y[start:start+L]

def compute_logmel(y: np.ndarray, sr: int) -> np.ndarray:
    S = librosa.feature.melspectrogram(
        y=y, sr=sr,
        n_fft=N_FFT, hop_length=HOP,
        n_mels=N_MELS, fmin=FMIN, fmax=FMAX,
        power=2.0
    )
    return librosa.power_to_db(S, ref=np.max).astype(np.float32)

def preprocess_audio(fp: str) -> tuple[np.ndarray, int, np.ndarray]:
    y, sr0 = librosa.load(fp, sr=None, mono=True)
    y = y.astype(np.float32)

    if sr0 != TARGET_SR:
        y = librosa.resample(y, orig_sr=sr0, target_sr=TARGET_SR)

    y, _ = librosa.effects.trim(y, top_db=25)
    y = y / (np.max(np.abs(y)) + 1e-8)
    y = fix_length_center(y, TARGET_SR, FIXED_DUR)
    logmel = compute_logmel(y, TARGET_SR)

    return y, TARGET_SR, logmel

# ===============================
# Feature extraction functions
# ===============================
def pre_emphasis(y: np.ndarray, coeff: float = 0.97) -> np.ndarray:
    return np.append(y[0], y[1:] - coeff * y[:-1])

def robust_stats(x: np.ndarray, prefix: str) -> dict:
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {f"{prefix}_{k}": 0.0 for k in ["mean","std","skew","kurt","p10","p25","p50","p75","p90"]}
    return {
        f"{prefix}_mean": float(np.mean(x)),
        f"{prefix}_std":  float(np.std(x)),
        f"{prefix}_skew": float(skew(x)),
        f"{prefix}_kurt": float(kurtosis(x)),
        f"{prefix}_p10":  float(np.percentile(x, 10)),
        f"{prefix}_p25":  float(np.percentile(x, 25)),
        f"{prefix}_p50":  float(np.percentile(x, 50)),
        f"{prefix}_p75":  float(np.percentile(x, 75)),
        f"{prefix}_p90":  float(np.percentile(x, 90)),
    }

def extract_features_from_audio(y: np.ndarray) -> dict:
    y = y.astype(np.float32)
    y = y / (np.max(np.abs(y)) + 1e-8)
    y = pre_emphasis(y)
    feats = {}

    # Time-domain
    rms = librosa.feature.rms(y=y)[0]
    zcr = librosa.feature.zero_crossing_rate(y=y)[0]
    feats.update(robust_stats(rms, "rms"))
    feats.update(robust_stats(zcr, "zcr"))

    # Spectral
    centroid  = librosa.feature.spectral_centroid(y=y, sr=TARGET_SR)[0]
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=TARGET_SR)[0]
    rolloff   = librosa.feature.spectral_rolloff(y=y, sr=TARGET_SR, roll_percent=0.85)[0]
    flatness  = librosa.feature.spectral_flatness(y=y)[0]
    feats.update(robust_stats(centroid,  "centroid"))
    feats.update(robust_stats(bandwidth, "bandwidth"))
    feats.update(robust_stats(rolloff,   "rolloff"))
    feats.update(robust_stats(flatness,  "flatness"))

    # MFCC + deltas + delta-deltas
    mfcc = librosa.feature.mfcc(y=y, sr=TARGET_SR, n_mfcc=N_MFCC)
    d1   = librosa.feature.delta(mfcc)
    d2   = librosa.feature.delta(mfcc, order=2)
    for i in range(N_MFCC):
        feats.update(robust_stats(mfcc[i], f"mfcc{i:02d}"))
        feats.update(robust_stats(d1[i],   f"mfcc{i:02d}_d1"))
        feats.update(robust_stats(d2[i],   f"mfcc{i:02d}_d2"))

    # Pitch via YIN
    f0 = librosa.yin(y, fmin=70, fmax=400, sr=TARGET_SR)
    f0 = f0[np.isfinite(f0)]
    feats.update(robust_stats(f0, "f0"))

    # Jitter proxy
    if f0.size > 2:
        f0_diff = np.abs(np.diff(f0))
        feats["jitter_rel_mean"] = float(np.mean(f0_diff) / (np.mean(f0) + 1e-8))
        feats["jitter_rel_std"]  = float(np.std(f0_diff)  / (np.mean(f0) + 1e-8))
    else:
        feats["jitter_rel_mean"] = 0.0
        feats["jitter_rel_std"]  = 0.0

    # HNR proxy
    y_h, y_p = librosa.effects.hpss(y)
    harm_energy = float(np.mean(y_h**2))
    perc_energy = float(np.mean(y_p**2))
    feats["harm_energy"] = harm_energy
    feats["perc_energy"] = perc_energy
    feats["harm_perc_ratio"] = float((harm_energy + 1e-8) / (perc_energy + 1e-8))

    return feats

# ===============================
# Streamlit UI
# ===============================
st.set_page_config(page_title="Speech Disorder Classification", layout="centered")
st.title("🗣️ Speech Disorder Classification")

uploaded_file = st.file_uploader("Upload a WAV or audio file", type=["wav","mp3","flac"])
patient_name = st.text_input("Enter patient name:")

if uploaded_file and patient_name:
    fp = Path("uploads") / uploaded_file.name
    fp.parent.mkdir(parents=True, exist_ok=True)
    with open(fp, "wb") as f:
        f.write(uploaded_file.getbuffer())

    y, sr, logmel = preprocess_audio(str(fp))

    # Save cleaned WAV and log-mel
    out_base = f"{patient_name}_{Path(uploaded_file.name).stem}"
    out_wav = WAV_DIR / f"{out_base}.wav"
    out_mel = MEL_DIR / f"{out_base}.npy"
    sf.write(out_wav, y, sr)
    np.save(out_mel, logmel)

    features = extract_features_from_audio(y)
    X = pd.DataFrame([features])
    for col in top_features:
        if col not in X.columns:
            X[col] = 0.0
    X = X[top_features]

    probas = model.predict_proba(X)[0]
    pred_idx = np.argmax(probas)
    pred_label = label_encoder.inverse_transform([pred_idx])[0]

    # Display result only (no plots)
    st.markdown(f"## Prediction for **{patient_name}**")
    st.write(f"**File:** {uploaded_file.name}")
    st.write(f"**Predicted label:** {pred_label}")
    st.write("**Probabilities:**")
    for label, prob in zip(label_encoder.classes_, probas):
        st.write(f"{label}: {prob*100:.1f}%")
