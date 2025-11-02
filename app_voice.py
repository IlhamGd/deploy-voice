import streamlit as st
import pandas as pd
import numpy as np
import joblib
import librosa
import os
from scipy.stats import skew, kurtosis

# --- FUNGSI EKSTRAKSI FITUR (WAJIB ADA) ---
# Ini adalah fungsi yang sama dari Colab Anda
def ekstrak_fitur(file_audio):
    """
    Memuat file audio dan mengekstrak fitur statistik time series.
    """
    try:
        y, sr = librosa.load(file_audio, sr=None) 
        y, _ = librosa.effects.trim(y, top_db=20) 
        
        # Fitur Statistik
        zcr_mean = np.mean(librosa.feature.zero_crossing_rate(y))
        zcr_std = np.std(librosa.feature.zero_crossing_rate(y))
        rms_mean = np.mean(librosa.feature.rms(y=y))
        rms_std = np.std(librosa.feature.rms(y=y))
        amp_mean = np.mean(y)
        amp_std = np.std(y)
        amp_var = np.var(y)
        amp_skew = skew(y)
        amp_kurtosis = kurtosis(y)
        amp_min = np.min(y)
        amp_max = np.max(y)
        
        return [
            zcr_mean, zcr_std, rms_mean, rms_std,
            amp_mean, amp_std, amp_var, amp_skew,
            amp_kurtosis, amp_min, amp_max
        ]
        
    except Exception as e:
        print(f"Error memproses {file_audio}: {e}")
        return None

# --- KONFIGURASI APLIKASI STREAMLIT ---
st.set_page_config(page_title="Deteksi Suara 'Buka' & 'Tutup'", layout="centered")
st.title("🎤 Aplikasi Deteksi Suara")
st.write("Unggah file audio (.wav) untuk memprediksi apakah isinya 'buka' atau 'tutup'.")

# --- 1. Muat Model, Scaler, dan Encoder ---
@st.cache_resource
def load_assets():
    try:
        model = joblib.load('audio_model.joblib')
        scaler = joblib.load('audio_scaler.joblib')
        encoder = joblib.load('audio_encoder.joblib')
        return model, scaler, encoder
    except FileNotFoundError:
        st.error("ERROR: File model (.joblib) tidak ditemukan.")
        st.info("Pastikan file 'audio_model.joblib', 'audio_scaler.joblib', dan 'audio_encoder.joblib' ada di repository GitHub.")
        return None, None, None

model, scaler, encoder = load_assets()

if model:
    # --- 2. Widget Upload File ---
    uploaded_file = st.file_uploader("Pilih file .wav", type=["wav"])

    if uploaded_file is not None:
        # Tampilkan audio player
        st.audio(uploaded_file, format='audio/wav')
        
        # Untuk memproses, librosa perlu file path, bukan buffer memori.
        # Kita simpan sementara.
        temp_file_path = "temp_audio.wav"
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # --- 3. Proses Prediksi Saat Tombol Ditekan ---
        if st.button("Deteksi Suara"):
            with st.spinner('Menganalisis audio...'):
                # Ekstrak fitur dari file audio yang di-upload
                fitur = ekstrak_fitur(temp_file_path)
                
                if fitur:
                    # Ubah fitur menjadi 2D array (sesuai tuntutan scaler)
                    fitur_2d = np.array(fitur).reshape(1, -1)
                    
                    # Scaling fitur
                    fitur_scaled = scaler.transform(fitur_2d)
                    
                    # Prediksi
                    pred_raw = model.predict(fitur_scaled) # Hasilnya [0] atau [1]
                    
                    # Ubah hasil (0/1) kembali ke label ('buka'/'tutup')
                    pred_label = encoder.inverse_transform(pred_raw) 
                    
                    # Tampilkan hasil
                    if pred_label[0] == 'buka':
                        st.success(f"### Hasil Prediksi: **BUKA**")
                    else:
                        st.error(f"### Hasil Prediksi: **TUTUP**")
                else:
                    st.error("Gagal mengekstrak fitur dari file audio.")
        
        # Hapus file sementara
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

else:
    st.warning("Aplikasi tidak dapat dimuat karena aset model tidak ditemukan.")
