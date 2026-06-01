---
title: BurnAway AI
emoji: 🔥
colorFrom: red
colorTo: yellow
sdk: docker
pinned: false
app_port: 8000
---


# 🔥 BurnAway AI Engineer Repository

[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org)
[![Gemini AI](https://img.shields.io/badge/Gemini%20AI-4285F4?style=for-the-badge&logo=google&logoColor=white)](https://ai.google.dev/)

**BurnAway** adalah solusi AI berbasis Deep Learning untuk mendeteksi tingkat *burnout* pada software developer secara dini. Proyek ini mencakup pipeline Machine Learning lengkap, dari pemrosesan data, pelatihan model *multi-task*, hingga deployment API.

---

## 📂 Struktur Direktori

```text
.
├── api/
│   └── main.py                        # Backend FastAPI (Rate-limited)
│   ├── class_labels.json              # Mapping label kelas
│   ├── feature_cols.json              # Daftar nama fitur
│   └── pipeline_config.json           # Konfigurasi pipeline
├── notebook/
│   └── BurnAway_AI_Engineer.ipynb     # Main ML Pipeline (End-to-End)
├── models/                            # Artifacts hasil training
│   ├── burnaway_model.keras           # Model terbaik (checkpoint)
│   ├── scaler.joblib                  # Scaler untuk preprocessing
├── data/                              # Dataset
├── logs/                              # TensorBoard training logs
├── _archives_and_notes/               # Dokumentasi lama & file eksperimen
├── inference.py                       # Standalone CLI inference script
├── requirements.txt                   # Dependensi Python
├── .gitignore                         # File & folder yang diabaikan Git
├── Dockerfile                         # Konfigurasi containerization
└── README.md                          # Dokumentasi & panduan penggunaan
```

---

## 🚀 Fitur Utama

1. **Custom Deep Learning Architecture**: Menggunakan `BurnoutAttentionLayer` dengan *attention mechanism* dan *Residual Connections* untuk menangani fitur tabular secara efektif.
2. **K-Fold Cross-Validation**: Validasi model dengan 5-fold stratified cross-validation untuk memastikan generalisasi yang robust.
3. **BorderlineSMOTE**: Menangani ketidakseimbangan kelas dengan BorderlineSMOTE untuk menghasilkan data sintetis yang lebih representatif di batas keputusan antar kelas.
4. **Custom Loss Function**: `WeightedCategoricalCrossentropy` dengan label smoothing untuk training yang lebih stabil.
5. **Cosine Decay LR Scheduler**: Learning rate decay otomatis dengan warmup untuk konvergensi yang mulus.
6. **GenAI Advisory Layer**: Terintegrasi dengan **Google Gemini API** dan **Groq (LLaMA 3.1)** sebagai fallback untuk memberikan saran kesehatan mental yang personal berdasarkan hasil prediksi.
7. **Production-Ready API**: Dilengkapi dengan *rate limiting* (SlowAPI) dan validasi skema Pydantic.

---

## 🛠️ Persiapan & Instalasi
 
### 1. Instalasi Dependensi
 
Pastikan menggunakan Python 3.9 atau lebih tinggi.
 
```bash
pip install -r requirements.txt
```
 
### 2. Konfigurasi Environment
 
Buat file `.env` di root directory atau set variabel lingkungan:
 
```bash
export GEMINI_API_KEY="your_gemini_api_key_here"
export GROQ_API_KEY="your_groq_api_key_here"
```
 
---

## 📈 Alur Kerja

### A. Training & Export
Jalankan notebook `notebook/BurnAway_AI_Engineer.ipynb`. Notebook ini akan mengekspor:
- `models/burnaway_model.keras`
- `models/scaler.joblib`
- `api/class_labels.json`
- `api/feature_cols.json`
- `api/pipeline_config.json`

### B. Uji Coba CLI
Gunakan skrip `inference.py` untuk prediksi cepat via terminal:
```bash
python inference.py --age 28 --exp 3 --work 11.5 --sleep 5.2 --caffeine 5 --bugs 14 --commits 7 --meetings 6 --screen 13.5 --exercise 0.3 --stress 5 --name Budi --role "Backend Engineer" --primary groq
```

### C. Deployment API
Jalankan server FastAPI:
```bash
python api/main.py
```
Akses dokumentasi interaktif di: `http://localhost:8000/docs`

---

## 🔌 API Endpoint: `/predict_burnout`

**Request Body Example:**
```json
{
  "age": 30,
  "experience_years": 5,
  "daily_work_hours": 10,
  "sleep_hours": 6,
  "caffeine_intake": 4,
  "bugs_per_day": 10,
  "commits_per_day": 5,
  "meetings_per_day": 4,
  "screen_time": 12,
  "exercise_hours": 1,
  "stress_level": 75
}
```

**Response Example:**
```json
{
  "prediction": {
    "burnout_level": "High",
    "confidence": 0.92,
    "stress_estimate": 78.5,
    "probabilities": { "Low": 0.01, "Medium": 0.07, "High": 0.92 }
  },
  "engineered_features": {
    "work_sleep_ratio": 1.67,
    "screen_time_intensity": 1.2,
    "commit_bug_ratio": 0.5,
    "work_category": 1.0
  },
  "advice": "..."
}
```

---
