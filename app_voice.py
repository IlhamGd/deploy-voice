import streamlit as st
import numpy as np
import librosa
import soundfile as sf
import os
from sklearn.metrics.pairwise import cosine_similarity
import tempfile

# ------------------------------
# Ekstraksi fitur suara (MFCC)
# ------------------------------
def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=None)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        return np.mean(mfcc.T, axis=0)
    except Exception as e:
        st.error(f"Gagal ekstrak fitur dari {file_path}: {e}")
        return None

# ------------------------------
# Verifikasi identitas pengguna
# ------------------------------
def verify_user(audio_path, enroll_dir="enroll"):
    test_feat = extract_features(audio_path)
    if test_feat is None:
        return None, 0.0

    similarities = {}
    for user in os.listdir(enroll_dir):
        user_dir = os.path.join(enroll_dir, user)
        if not os.path.isdir(user_dir):
            continue

        scores = []
        for file in os.listdir(user_dir):
            if file.endswith(".wav"):
                feat = extract_features(os.path.join(user_dir, file))
                if feat is not None:
                    sim = cosine_similarity([test_feat], [feat])[0][0]
                    scores.append(sim)
        if scores:
            similarities[user] = np.mean(scores)

    if not similarities:
        return None, 0.0

    best_user = max(similarities, key=similarities.get)
    best_score = similarities[best_user]
    return best_user, best_score

# ------------------------------
# Deteksi kata kunci (buka/tutup)
# ------------------------------
def detect_command(audio_path):
    try:
        y, sr = librosa.load(audio_path, sr=None)
        text = ""  # Dummy command recognizer

        # Kita akan pakai fitur energi + durasi
        energy = np.mean(np.abs(y))
        duration = librosa.get_duration(y=y, sr=sr)

        # Sederhana: misal deteksi dengan panjang & energi
        # (kalau mau real KWS, bisa pakai model kecil nanti)
        if duration < 1:
            text = "buka"
        elif duration > 1:
            text = "tutup"

        return text
    except Exception as e:
        st.error(f"Gagal deteksi perintah: {e}")
        return None

# ------------------------------
# UI Streamlit
# ------------------------------
st.title("🔐 Sistem Verifikasi Suara - Perintah Buka/Tutup")
st.caption("Hanya pengguna terdaftar yang dapat memberikan perintah suara 'buka' atau 'tutup'.")

uploaded_file = st.file_uploader("🎙️ Unggah suara (.wav)", type=["wav"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_file.read())
        temp_audio_path = tmp.name

    st.audio(uploaded_file, format="audio/wav")

    if st.button("Mulai Verifikasi"):
        with st.spinner("Menganalisis suara..."):
            user, score = verify_user(temp_audio_path)

            if user and score > 0.85:
                st.success(f"✅ Pengguna terdeteksi: **{user}** (skor {score:.2f})")

                cmd = detect_command(temp_audio_path)
                if cmd == "buka":
                    st.success("🟢 Perintah terdeteksi: **BUKA** — Sistem terbuka.")
                elif cmd == "tutup":
                    st.warning("🔴 Perintah terdeteksi: **TUTUP** — Sistem tertutup.")
                else:
                    st.info("⚠️ Tidak dapat mengenali perintah.")
            else:
                st.error("🚫 Akses ditolak! Suara tidak dikenali.")

        os.remove(temp_audio_path)
