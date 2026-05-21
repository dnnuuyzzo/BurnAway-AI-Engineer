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
├── notebooks/
│   └── BurnAway_AI_Engineer.ipynb     # Main ML Pipeline (End-to-End)
├── models/                            # Artifacts hasil training
│   ├── burnaway_model_best.keras      # Model terbaik (checkpoint)
│   ├── scaler.joblib                  # Scaler untuk preprocessing
│   ├── class_labels.json              # Mapping label kelas
│   ├── feature_cols.json              # Daftar nama fitur
│   └── pipeline_config.json           # Konfigurasi pipeline
├── logs/                              # TensorBoard training logs
├── _archives_and_notes/               # Dokumentasi lama & file eksperimen
├── inference.py                       # Standalone CLI inference script
├── requirements.txt                   # Dependensi Python
└── Dockerfile                         # Konfigurasi containerization
```

---

## 🚀 Fitur Utama

1.  **Multi-Task Deep Learning**: Memprediksi kategori *burnout* (Classification) sekaligus memberikan estimasi skor stres (Regression).
2.  **Custom Architecture**: Menggunakan `FTTransformerBlock` untuk menangani fitur tabular dengan *attention mechanism*.
3.  **GenAI Advisory Layer**: Terintegrasi dengan **Google Gemini API** untuk memberikan saran kesehatan mental yang personal berdasarkan hasil prediksi.
4.  **Production-Ready API**: Dilengkapi dengan *rate limiting* (SlowAPI) dan validasi skema Pydantic.

---

## 🛠️ Persiapan & Instalasi

### 1. Instalasi Dependensi
Pastikan Anda menggunakan Python 3.9 atau lebih tinggi.
```bash
pip install -r requirements.txt
```

### 2. Konfigurasi Environment
Buat file `.env` di root directory atau set variabel lingkungan untuk fitur saran AI:
```bash
export GEMINI_API_KEY="your_api_key_here"
```

---

## 📈 Alur Kerja

### A. Training & Export
Jalankan notebook `notebooks/BurnAway_AI_Engineer.ipynb`. Notebook ini akan mengekspor:
- `models/burnaway_model_best.keras`
- `models/scaler.joblib`
- `models/class_labels.json`
- `models/feature_cols.json`
- `models/pipeline_config.json`

### B. Uji Coba CLI
Gunakan skrip `inference.py` untuk prediksi cepat via terminal:
```bash
python inference.py --age 28 --exp 5 --work 11.5 --sleep 6 --screen 12 --stress 80
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
  "advice": "..."
}
```

---
