import streamlit as st
import soundfile as sf
import numpy as np
import librosa
import os
from sklearn.metrics.pairwise import cosine_similarity

# =========================
# Konfigurasi Awal
# =========================
st.title("🔐 Sistem Verifikasi Suara (Offline - Tanpa Hugging Face)")

BASE_DIR = "voice_database"
os.makedirs(BASE_DIR, exist_ok=True)
THRESHOLD = 0.85  # Semakin tinggi semakin ketat

# =========================
# Fungsi Ekstraksi Fitur
# =========================
def extract_mfcc(file_path, n_mfcc=13):
    y, sr = librosa.load(file_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfcc, axis=1).reshape(1, -1)

# =========================
# Fungsi Pendaftaran Suara
# =========================
def register_voice(name, audio_file):
    save_path = os.path.join(BASE_DIR, f"{name}.wav")
    with open(save_path, "wb") as f:
        f.write(audio_file.read())
    st.success(f"✅ Suara {name} berhasil disimpan.")
    return save_path

# =========================
# Fungsi Verifikasi Suara
# =========================
def verify_voice(audio_file, registered_name):
    test_path = "temp_voice.wav"
    with open(test_path, "wb") as f:
        f.write(audio_file.read())

    registered_path = os.path.join(BASE_DIR, f"{registered_name}.wav")
    if not os.path.exists(registered_path):
        st.error("❌ Tidak ada data suara untuk nama ini.")
        return

    mfcc_reg = extract_mfcc(registered_path)
    mfcc_test = extract_mfcc(test_path)
    sim = cosine_similarity(mfcc_reg, mfcc_test)[0][0]

    if sim >= THRESHOLD:
        st.success(f"✅ Verifikasi Berhasil! (Similarity: {sim:.2f})")
    else:
        st.error(f"❌ Verifikasi Gagal (Similarity: {sim:.2f})")

    os.remove(test_path)

# =========================
# Tampilan Utama
# =========================
mode = st.sidebar.radio("Pilih Mode:", ["🗣️ Daftar Suara Baru", "🔍 Verifikasi Suara"])

if mode == "🗣️ Daftar Suara Baru":
    name = st.text_input("Masukkan nama pengguna:")
    audio = st.file_uploader("Upload rekaman suara (.wav):", type=["wav"])

    if st.button("Daftar") and name and audio:
        register_voice(name, audio)
    elif st.button("Daftar"):
        st.warning("⚠️ Lengkapi nama dan file suara terlebih dahulu!")

elif mode == "🔍 Verifikasi Suara":
    registered_name = st.text_input("Masukkan nama yang ingin diverifikasi:")
    audio = st.file_uploader("Upload rekaman baru untuk verifikasi (.wav):", type=["wav"])

    if st.button("Verifikasi") and registered_name and audio:
        verify_voice(audio, registered_name)
    elif st.button("Verifikasi"):
        st.warning("⚠️ Lengkapi nama dan file suara terlebih dahulu!")

st.caption("💡 Sistem ini bekerja lokal tanpa internet atau model eksternal (MFCC + Cosine Similarity).")
