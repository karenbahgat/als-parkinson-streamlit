import streamlit as st
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
import joblib
import tempfile
from scipy.stats import skew, kurtosis

# ===============================
# Constants (MUST match training)
# ===============================
TARGET_SR = 16000
FIXED_DUR = 2.5
N_MFCC = 20

# ===============================
# Load model artifacts
# ===============================
@st.cache_resource
def load_artifacts():
    artifact = joblib.load("model_artifact.pkl")  # <-- نفس الاسم اللي حفظتيه
    return (
        artifact["model"],
        artifact["label_encoder"],
        artifact["top_features"],
        artifact["target_sr"],
        artifact["fixed_dur"],
    )

model, label_encoder, top_features, TARGET_SR, FIXED_DUR = load_artifacts()

# ===============================
# Audio preprocessing (same as training)
# ===============================
def fix_length_center(y: np.ndarray, sr: int, fixed_dur: float) -> np.ndarray:
    L = int(round(fixed_dur * sr))
    if len(y) < L:
        return np.pad(y, (0, L - len(y)))
    start = (len(y) - L) // 2
    return y[start:start + L]

def preprocess_audio(y, sr):
    if y.ndim > 1:
        y = np.mean(y, axis=1)

    if sr != TARGET_SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=TARGET_SR)

    y, _ = librosa.effects.trim(y, top_db=25)

    mx = np.max(np.abs(y)) + 1e-8
    y = (y / mx).astype(np.float32)

    y = fix_length_center(y, TARGET_SR, FIXED_DUR)
    return y

# ===============================
# Feature extraction (IDENTICAL)
# ===============================
def pre_emphasis(y: np.ndarray, coeff: float = 0.97) -> np.ndarray:
    return np.append(y[0], y[1:] - coeff * y[:-1])

def robust_stats(x: np.ndarray, prefix: str) -> dict:
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {f"{prefix}_{k}": 0.0 for k in
                ["mean","std","skew","kurt","p10","p25","p50","p75","p90"]}
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

def extract_voice_features(y: np.ndarray) -> dict:
    y = pre_emphasis(y)

    feats = {}

    rms = librosa.feature.rms(y=y)[0]
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    feats.update(robust_stats(rms, "rms"))
    feats.update(robust_stats(zcr, "zcr"))

    centroid  = librosa.feature.spectral_centroid(y=y, sr=TARGET_SR)[0]
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=TARGET_SR)[0]
    rolloff   = librosa.feature.spectral_rolloff(y=y, sr=TARGET_SR, roll_percent=0.85)[0]
    flatness  = librosa.feature.spectral_flatness(y=y)[0]
    feats.update(robust_stats(centroid,  "centroid"))
    feats.update(robust_stats(bandwidth, "bandwidth"))
    feats.update(robust_stats(rolloff,   "rolloff"))
    feats.update(robust_stats(flatness,  "flatness"))

    mfcc = librosa.feature.mfcc(y=y, sr=TARGET_SR, n_mfcc=N_MFCC)
    d1   = librosa.feature.delta(mfcc)
    d2   = librosa.feature.delta(mfcc, order=2)

    for i in range(N_MFCC):
        feats.update(robust_stats(mfcc[i], f"mfcc{i:02d}"))
        feats.update(robust_stats(d1[i],   f"mfcc{i:02d}_d1"))
        feats.update(robust_stats(d2[i],   f"mfcc{i:02d}_d2"))

    f0 = librosa.yin(y, fmin=70, fmax=400, sr=TARGET_SR)
    f0 = f0[np.isfinite(f0)]
    feats.update(robust_stats(f0, "f0"))

    if f0.size > 2:
        f0_diff = np.abs(np.diff(f0))
        feats["jitter_rel_mean"] = float(np.mean(f0_diff) / (np.mean(f0) + 1e-8))
        feats["jitter_rel_std"]  = float(np.std(f0_diff)  / (np.mean(f0) + 1e-8))
    else:
        feats["jitter_rel_mean"] = 0.0
        feats["jitter_rel_std"]  = 0.0

    y_h, y_p = librosa.effects.hpss(y)
    harm_energy = float(np.mean(y_h ** 2))
    perc_energy = float(np.mean(y_p ** 2))
    feats["harm_energy"] = harm_energy
    feats["perc_energy"] = perc_energy
    feats["harm_perc_ratio"] = float((harm_energy + 1e-8) / (perc_energy + 1e-8))

    return feats

# ===============================
# Streamlit UI
# ===============================
st.set_page_config(page_title="Speech Disorder Classifier", layout="centered")
st.title("🗣️ Speech Disorder Classification")
st.write("Upload a WAV file and get prediction probabilities")

uploaded_file = st.file_uploader("Upload WAV file", type=["wav"])

if uploaded_file is not None:
    with st.spinner("Processing audio..."):
        y, sr = sf.read(uploaded_file)
        y = preprocess_audio(y, sr)

        feats = extract_voice_features(y)
        X = pd.DataFrame([feats])

        # ⭐ THE MAGIC LINE (no KeyError ever)
        X = X.reindex(columns=top_features, fill_value=0.0)

        probs = model.predict_proba(X)[0]
        pred_idx = int(np.argmax(probs))
        label = label_encoder.inverse_transform([pred_idx])[0]

    st.success(f"### 🧠 Prediction: **{label}**")

    st.subheader("Prediction probabilities")
    prob_df = pd.DataFrame({
        "Class": label_encoder.classes_,
        "Probability": probs
    }).sort_values("Probability", ascending=False)

    st.dataframe(prob_df, use_container_width=True)


