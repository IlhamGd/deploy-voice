import streamlit as st
import torch
import os
import sys
import librosa
import numpy as np
import pandas as pd
import joblib  # Atau pickle, sesuaikan dengan cara Anda menyimpan model
from speechbrain.inference.speaker import SpeakerRecognition

# --- KONFIGURASI APLIKASI ---
st.set_page_config(page_title="Verifikasi Suara", layout="centered")
st.title("🔐 Sistem Verifikasi Perintah Suara")
st.write("Aplikasi ini hanya akan merespon perintah 'Buka' atau 'Tutup' jika diucapkan oleh pengguna yang terdaftar.")

# --- PATH & PENGATURAN MODEL ---
# Sesuaikan path ini
APP_DIR = os.path.dirname(os.path.abspath(__file__))

# Buat semua path lain berdasarkan APP_DIR
PATH_MODEL_KWS = os.path.join(APP_DIR, "model_kws.pkl")
PATH_LABEL_ENCODER = os.path.join(APP_DIR, "label_encoder.pkl")
PATH_ANDA = os.path.join(APP_DIR, "enroll", "v_ilham")
PATH_TEMAN = os.path.join(APP_DIR, "enroll", "v_danendra")

# Pengaturan threshold (tetap sama)
THRESHOLD = 0.85  # Sesuaikan ini!

# --- FUNGSI BANTUAN MODEL 2 (SpeechBrain) ---
# Fungsi-fungsi ini dari skrip kita sebelumnya

def get_embedding(file_path, model_sv):
    if not os.path.exists(file_path):
        st.error(f"File not found: {file_path}")
        return None
    try:
        embedding = model_sv.encode_file(file_path)
        return embedding.squeeze()
    except Exception as e:
        st.error(f"Error processing {file_path}: {e}")
        return None

def get_similarity(emb1, emb2, model_sv):
    emb1_batch = emb1.unsqueeze(0)
    emb2_batch = emb2.unsqueeze(0)
    score = model_sv.similarity(emb1_batch, emb2_batch)
    return score.item()

def create_master_voiceprint(directory_path, model_sv):
    embeddings = []
    if not os.path.isdir(directory_path):
        st.warning(f"Direktori pendaftaran tidak ditemukan: {directory_path}")
        return None

    for file_name in os.listdir(directory_path):
        if file_name.endswith(".wav"):
            file_path = os.path.join(directory_path, file_name)
            emb = get_embedding(file_path, model_sv)
            if emb is not None:
                embeddings.append(emb)
    
    if not embeddings:
        st.error(f"Tidak ada file .wav ditemukan di {directory_path}")
        return None
        
    master_voiceprint = torch.mean(torch.stack(embeddings), dim=0)
    return master_voiceprint

# --- LOADING MODEL (DENGAN CACHE) ---
# Ini agar model tidak di-load ulang setiap kali ada interaksi

@st.cache_resource
def load_model_sv():
    st.info("Memuat Model Verifikasi Suara (SpeechBrain)...")
    try:
        model = SpeakerRecognition.from_hparams(
            source="speechbrain/spkrec-ecapa-tdnn",
            savedir="pretrained_models/spkrec-ecapa-tdnn",
            use_auth_token=False # Tambahkan ini untuk menghindari error 401
        )
        st.success("Model Verifikasi Suara siap.")
        return model
    except Exception as e:
        st.exception(e)
        st.error("Gagal memuat model SpeechBrain. Cek koneksi internet.")
        return None

@st.cache_resource
def load_model_kws(path):
    st.info("Memuat Model Pengenal Kata Kunci (KWS)...")
    if not os.path.exists(path):
        st.error(f"File model KWS tidak ditemukan di: {path}")
        return None
    try:
        model = joblib.load(path)
        st.success("Model KWS siap.")
        return model
    except Exception as e:
        st.exception(e)
        return None

@st.cache_resource
def load_voiceprints(_model_sv):
    st.info("Membuat master voiceprint...")
    voiceprints = {}
    
    vp_a = create_master_voiceprint(PATH_ANDA, _model_sv)
    if vp_a is not None:
        voiceprints["anda"] = vp_a
        st.success("Voiceprint 'anda' dibuat.")
        
    vp_b = create_master_voiceprint(PATH_TEMAN, _model_sv)
    if vp_b is not None:
        voiceprints["teman"] = vp_b
        st.success("Voiceprint 'teman' dibuat.")
        
    if not voiceprints:
        st.error("Gagal membuat voiceprint. Pastikan folder 'enroll' ada.")
        return None
    return voiceprints

# --- FUNGSI PIPELINE UTAMA ---

def ekstrak_fitur_kws(audio_file):
    """
    PENTING: Fungsi ini HARUS mengekstrak fitur yang SAMA PERSIS
    dengan yang Anda gunakan untuk melatih Model 1 di Ppreprocessing.pdf.
    Ini hanya contoh!
    """
    y, sr = librosa.load(audio_file, sr=16000)
    
    # Preprocessing dari PDF Anda: trim & normalisasi
    y_trimmed, _ = librosa.effects.trim(y, top_db=20)
    y_norm = y_trimmed / np.max(np.abs(y_trimmed))
    
    # Ekstrak fitur (HARUS SAMA DENGAN PDF ANDA)
    # [cite_start]Ini hanya contoh dari PDF Anda [cite: 194-212]
    fitur = {
        'spectral_centroid': np.mean(librosa.feature.spectral_centroid(y=y_norm, sr=sr)),
        'spectral_bandwidth': np.mean(librosa.feature.spectral_bandwidth(y=y_norm, sr=sr)),
        'spectral_rolloff': np.mean(librosa.feature.spectral_rolloff(y=y_norm, sr=sr)),
        'spectral_contrast': np.mean(librosa.feature.spectral_contrast(y=y_norm, sr=sr)),
        'spectral_flatness': np.mean(librosa.feature.spectral_flatness(y=y_norm)),
        'mfcc_delta2_mean': np.mean(librosa.feature.delta(librosa.feature.mfcc(y=y_norm, sr=sr, n_mfcc=13), order=2)),
        'f0_mean': np.nanmean(librosa.pyin(y_norm, fmin=50, fmax=400, sr=sr)[0]),
        'rms': np.mean(librosa.feature.rms(y=y_norm)),
        'duration': librosa.get_duration(y=y_norm, sr=sr),
        'std_amplitude': np.std(y_norm)
    }
    
    # Mengisi NaN jika ada (misal dari f0)
    df = pd.DataFrame([fitur]).fillna(0)
    
    # Pastikan urutan kolom sama dengan saat training!
    # Ini hanya contoh berdasarkan 10 fitur akhir di PDF Anda
    nama_fitur_akhir = [
        'spectral_centroid', 'spectral_bandwidth', 'spectral_rolloff',
        'spectral_contrast', 'spectral_flatness', 'mfcc_delta2_mean',
        'f0_mean', 'rms', 'duration', 'std_amplitude'
    ]
    
    # Filter dan urutkan
    df = df[nama_fitur_akhir]
    return df


def cek_keyword(audio_file, model_kws):
    """
    Menjalankan Model 1 (KWS) untuk mendeteksi kata kunci.
    """
    try:
        fitur = ekstrak_fitur_kws(audio_file)
        prediksi = model_kws.predict(fitur)
        return prediksi[0]  # Mengambil hasil prediksi (misal: "buka" atau "tutup")
    except Exception as e:
        st.exception(e)
        st.error("Gagal mengekstrak fitur KWS.")
        return "error"

def verifikasi_suara(audio_file, model_sv, voiceprints, threshold):
    """
    Menjalankan Model 2 (SV) untuk verifikasi pembicara.
    """
    try:
        test_embedding = get_embedding(audio_file, model_sv)
        if test_embedding is None:
            return False, 0.0, "Gagal buat embedding"

        best_score = -1.0
        best_match = "None"
        
        for name, master_vp in voiceprints.items():
            score = get_similarity(test_embedding, master_vp, model_sv)
            if score > best_score:
                best_score = score
                best_match = name
        
        if best_score >= threshold:
            return True, best_score, best_match
        else:
            return False, best_score, "None"
            
    except Exception as e:
        st.exception(e)
        st.error("Gagal saat verifikasi suara.")
        return False, 0.0, "Error"

# --- MAIN APP ---
# Memuat semua model saat aplikasi dimulai
model_sv = load_model_sv()
model_kws = load_model_kws(PATH_MODEL_KWS)
voiceprints = load_voiceprints(model_sv)

# Cek jika model gagal di-load
if not all([model_sv, model_kws, voiceprints]):
    st.error("Gagal memuat semua model atau voiceprint. Aplikasi tidak bisa berjalan.")
else:
    st.header("Upload Audio Perintah (.wav)")
    uploaded_file = st.file_uploader("Pilih file audio...", type=["wav"])
    
    # Buat file audio sementara
    temp_audio_path = None
    if uploaded_file is not None:
        # Simpan file yang di-upload ke disk sementara
        # karena librosa & speechbrain butuh path file
        with open("temp_audio.wav", "wb") as f:
            f.write(uploaded_file.getbuffer())
        temp_audio_path = "temp_audio.wav"
        
        st.audio(temp_audio_path)

    if st.button("Proses Perintah", disabled=(temp_audio_path is None)):
        if temp_audio_path:
            with st.spinner("Menganalisis audio..."):
                
                # --- LANGKAH 1: Cek Kata Kunci ---
                st.subheader("Hasil Model 1: Pengenalan Kata Kunci")
                kata_kunci = cek_keyword(temp_audio_path, model_kws)
                
                if kata_kunci in ["buka", "tutup"]:
                    st.info(f"Kata kunci terdeteksi: **{kata_kunci.upper()}**")
                    
                    # --- LANGKAH 2: Verifikasi Suara ---
                    st.subheader("Hasil Model 2: Verifikasi Suara")
                    terverifikasi, skor, nama = verifikasi_suara(
                        temp_audio_path, model_sv, voiceprints, THRESHOLD
                    )
                    
                    st.info(f"Skor kemiripan tertinggi: **{skor:.2%}** (dengan '{nama}')")
                    
                    # --- KEPUTUSAN AKHIR ---
                    st.header("Keputusan Akhir")
                    if terverifikasi:
                        st.success(f"✅ DITERIMA. Suara terverifikasi sebagai '{nama}'. Perintah **{kata_kunci.upper()}** dijalankan.")
                    else:
                        st.error(f"❌ DITOLAK. Suara tidak dikenal. Perintah **{kata_kunci.upper()}** dibatalkan.")
                        
                elif kata_kunci == "error":
                    st.error("Terjadi error saat memproses kata kunci.")
                else:
                    st.header("Keputusan Akhir")
                    st.warning(f"❌ DITOLAK. Perintah tidak dikenal (terdeteksi sebagai: '{kata_kunci}').")

            # Hapus file sementara
            os.remove(temp_audio_path)
