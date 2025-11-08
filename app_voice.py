import streamlit as st
import numpy as np
import librosa
import os
from sklearn.metrics.pairwise import cosine_similarity
# Tidak perlu import soundfile, librosa akan menggunakannya secara otomatis

# --- PATH KONFIGURASI ---
# Menggunakan path absolut untuk menemukan folder 'enroll'
APP_DIR = os.path.dirname(os.path.abspath(__file__))
ENROLL_DIR = os.path.join(APP_DIR, "enroll")

# ------------------------------
# Ekstraksi fitur suara (MFCC)
# ------------------------------
def extract_features(audio_source):
    """
    Kombinasi: Fungsi ini dari skrip Anda,
    tapi dimodifikasi untuk menerima 'audio_source' (file-like object)
    agar kita tidak perlu file temporer.
    """
    try:
        # librosa.load bisa membaca file-like object dari st.uploader
        y, sr = librosa.load(audio_source, sr=None)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        return np.mean(mfcc.T, axis=0)
    except Exception as e:
        st.error(f"Gagal ekstrak fitur: {e}")
        return None

# ------------------------------
# Verifikasi identitas pengguna
# ------------------------------
def verify_user(audio_source, enroll_dir):
    """
    Logika dari skrip Anda, dikombinasikan dengan path yang lebih
    kuat (ENROLL_DIR) dan fungsi 'extract_features' yang
    sudah dimodifikasi.
    """
    test_feat = extract_features(audio_source)
    if test_feat is None:
        return None, 0.0

    similarities = {}
    
    if not os.path.isdir(enroll_dir):
        st.error(f"Folder 'enroll' tidak ditemukan di: {enroll_dir}")
        return None, 0.0

    for user in os.listdir(enroll_dir):
        user_dir = os.path.join(enroll_dir, user)
        if not os.path.isdir(user_dir):
            continue

        scores = []
        for file in os.listdir(user_dir):
            if file.endswith(".wav"):
                # Untuk file pendaftaran, kita gunakan file path (str)
                file_path = os.path.join(user_dir, file)
                feat = extract_features(file_path)
                
                if feat is not None:
                    # Logika 'cosine_similarity' Anda
                    sim = cosine_similarity([test_feat], [feat])[0][0]
                    scores.append(sim)
        if scores:
            similarities[user] = np.mean(scores)

    if not similarities:
        st.error("Tidak ada data pendaftaran (enroll) yang ditemukan/diproses.")
        return None, 0.0

    best_user = max(similarities, key=similarities.get)
    best_score = similarities[best_user]
    return best_user, best_score

# ------------------------------
# Deteksi kata kunci (buka/tutup) - Logika Anda
# ------------------------------
def detect_command(audio_source):
    """
    Ini adalah fungsi 'dummy' dari skrip Anda.
    Logikanya (berdasarkan durasi) dipertahankan.
    """
    try:
        y, sr = librosa.load(audio_source, sr=None)
        duration = librosa.get_duration(y=y, sr=sr)

        # Logika placeholder dari kode Anda:
        if duration < 1.0:
            text = "buka"
        else:
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

st.warning(
    """PERHATIAN: Deteksi perintah ('buka'/'tutup') saat ini 
    hanyalah **placeholder** berdasarkan durasi audio (audio pendek = 'buka', 
    audio panjang = 'tutup') dan **tidak akurat**.""", 
    icon="⚠️"
)

uploaded_file = st.file_uploader("🎙️ Unggah suara (.wav)", type=["wav"])

if uploaded_file is not None:
    # Kombinasi: Kita HAPUS 'tempfile' dan 'os.remove'.
    # Kita gunakan 'uploaded_file' secara langsung.
    st.audio(uploaded_file, format="audio/wav")

    if st.button("Mulai Verifikasi"):
        with st.spinner("Menganalisis suara..."):
            
            # Mengirim 'uploaded_file' (objek memori) langsung ke fungsi
            user, score = verify_user(uploaded_file, ENROLL_DIR)

            # Logika UI Anda
            if user and score > 0.85:
                st.success(f"✅ Pengguna terdeteksi: **{user}** (skor {score:.2f})")

                cmd = detect_command(uploaded_file)
                
                if cmd == "buka":
                    st.success("🟢 Perintah terdeteksi: **BUKA** — Sistem terbuka.")
                elif cmd == "tutup":
                    st.warning("🔴 Perintah terdeteksi: **TUTUP** — Sistem tertutup.")
                else:
                    st.info("⚠️ Tidak dapat mengenali perintah.")
            else:
                st.error("🚫 Akses ditolak! Suara tidak dikenali.")

# Tidak ada 'os.remove()' lagi karena tidak ada file temporer
