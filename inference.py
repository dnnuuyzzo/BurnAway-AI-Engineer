import os
import argparse
import numpy as np
import joblib
import requests
import tensorflow as tf

try:
    import google.generativeai as genai
    GENAI_SDK_AVAILABLE = True
except ImportError:
    GENAI_SDK_AVAILABLE = False

# ── Constants ────────────────────────────────────────────────────────────────
NUM_CLASSES     = 3
CLASS_LABELS    = {0: "Low", 1: "Medium", 2: "High"}
ATTENTION_UNITS = 64

GEMINI_API_KEY  = os.environ.get("GEMINI_API_KEY", "")
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"
GEMINI_MODEL    = os.environ.get("GEMINI_MODEL", "gemini-3.1-flash-lite")

GROQ_API_KEY    = os.environ.get("GROQ_API_KEY", "")
GROQ_BASE_URL   = "https://api.groq.com/openai/v1/"
GROQ_MODEL      = "llama-3.1-8b-instant"

FEATURE_COLS = [
    "age", "experience_years", "daily_work_hours", "sleep_hours",
    "caffeine_intake", "bugs_per_day", "commits_per_day", "meetings_per_day",
    "screen_time", "exercise_hours", "work_sleep_ratio",
    "screen_time_intensity", "commit_bug_ratio", "work_category", "stress_level"
]

MODEL_PATH  = os.path.join("models", "burnaway_model.keras")
SCALER_PATH = os.path.join("models", "scaler.joblib")


# Model Components
@tf.keras.utils.register_keras_serializable(package="Custom")
class BurnoutAttentionLayer(tf.keras.layers.Layer):
    def __init__(self, units: int = 32, **kwargs):
        super().__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer="glorot_uniform", trainable=True, name="attention_kernel"
        )
        self.b = self.add_weight(
            shape=(self.units,),
            initializer="zeros", trainable=True, name="attention_bias"
        )
        self.attention_v = self.add_weight(
            shape=(self.units, 1),
            initializer="glorot_uniform", trainable=True, name="attention_v"
        )

    def call(self, inputs):
        score             = tf.nn.tanh(tf.matmul(inputs, self.w) + self.b)
        attention_weights = tf.nn.softmax(tf.matmul(score, self.attention_v), axis=-1)
        return inputs * attention_weights

    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units})
        return config


@tf.keras.utils.register_keras_serializable(package="Custom")
class WeightedCategoricalCrossentropy(tf.keras.losses.Loss):
    """Custom loss dengan bobot per kelas dan label smoothing."""

    def __init__(self, class_weights: list = None, label_smoothing: float = 0.0, **kwargs):
        super().__init__(**kwargs)
        weights = class_weights if class_weights is not None else [1.0, 1.0, 1.0]
        self.class_weights   = tf.constant(weights, dtype=tf.float32)
        self.label_smoothing = tf.constant(label_smoothing, dtype=tf.float32)

    def call(self, y_true, y_pred):
        y_true     = tf.cast(y_true, tf.int32)
        y_pred     = tf.clip_by_value(y_pred, 1e-7, 1.0)
        y_onehot   = tf.one_hot(y_true, depth=NUM_CLASSES)
        y_smoothed = y_onehot * (1.0 - self.label_smoothing) + (self.label_smoothing / NUM_CLASSES)
        ce         = -tf.reduce_sum(y_smoothed * tf.math.log(y_pred), axis=-1)
        sample_w   = tf.gather(self.class_weights, y_true)
        return tf.reduce_mean(ce * sample_w)

    def get_config(self):
        config = super().get_config()
        config.update({
            "class_weights":   self.class_weights.numpy().tolist(),
            "label_smoothing": self.label_smoothing.numpy().item()
        })
        return config


# Inference
def predict_burnout(raw_input: dict, inference_model, inference_scaler) -> dict:
    missing = [c for c in FEATURE_COLS if c not in raw_input]
    if missing:
        raise ValueError(f"[ERROR] Fitur berikut tidak ditemukan dalam input: {missing}")

    x_raw   = np.array([[raw_input[c] for c in FEATURE_COLS]], dtype=np.float32)

    cbr_idx = FEATURE_COLS.index("commit_bug_ratio") if "commit_bug_ratio" in FEATURE_COLS else -1
    if cbr_idx >= 0:
        x_raw[0, cbr_idx] = min(x_raw[0, cbr_idx], 220.0)

    x_scaled  = inference_scaler.transform(x_raw).astype(np.float32)
    clf_probs  = inference_model.predict(x_scaled, verbose=0)
    probs      = clf_probs[0]
    pred_cls   = int(np.argmax(probs))

    return {
        "label":       CLASS_LABELS[pred_cls],
        "confidence":  float(probs[pred_cls]),
        "class_probs": {CLASS_LABELS[i]: float(p) for i, p in enumerate(probs)},
    }


# GenAI Advisory
def build_advisory_prompt(prediction_result: dict, user_context: dict = None) -> str:
    label       = prediction_result["label"]
    confidence  = prediction_result["confidence"]
    class_probs = prediction_result["class_probs"]

    context_str = ""
    if user_context:
        name       = user_context.get("name", "Developer")
        work_hours = user_context.get("work_hours", None)
        role       = user_context.get("role", None)

        context_str = f"Developer yang dianalisis bernama {name}."
        if work_hours:
            context_str += f" Ia bekerja rata-rata {work_hours} jam per hari."
        if role:
            context_str += f" Role-nya adalah {role}."

    return (
        f"Kamu adalah asisten kesehatan mental profesional untuk developer perangkat lunak.\n"
        f"{context_str}\n"
        f"Berdasarkan analisis AI, developer ini terdeteksi berada di level burnout: **{label}** "
        f"(confidence: {confidence:.1%}).\n"
        f"Distribusi probabilitas kelas: {class_probs}.\n\n"
        f"Berikan 5 rekomendasi konkret dan personal dalam bahasa Indonesia yang:\n"
        f"1. Spesifik untuk level burnout '{label}'\n"
        f"2. Dapat langsung diterapkan oleh developer\n"
        f"3. Mencakup aspek: manajemen waktu, kesehatan fisik, dan produktivitas\n"
        f"Tutup dengan kalimat motivasi singkat."
    )

def get_burnout_advice_sdk(prompt: str) -> str:
    client   = genai.Client(api_key=GEMINI_API_KEY)
    response = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
    return response.text

def get_burnout_advice_rest(prompt: str) -> str:
    if not GEMINI_API_KEY:
        return "[INFO] GEMINI_API_KEY tidak ditemukan."
    url     = f"{GEMINI_BASE_URL}/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.7, "maxOutputTokens": 1024},
    }
    resp = requests.post(url, json=payload, timeout=30)
    resp.raise_for_status()
    return resp.json()["candidates"][0]["content"]["parts"][0]["text"]

def get_burnout_advice_groq(prompt: str) -> str:
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY tidak ditemukan.")
    url     = f"{GROQ_BASE_URL}chat/completions"
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": GROQ_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "max_tokens": 1024,
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=30)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]

def _offline_fallback(label: str) -> str:
    label = label.lower()
    if label == "high":
        return (
            "Tingkat burnout Anda TINGGI.\n"
            "1. Segera jadwalkan waktu istirahat atau cuti panjang.\n"
            "2. Diskusikan pendelegasian tugas dengan manajer.\n"
            "3. Cari bantuan profesional (psikolog/konselor) jika kewalahan.\n"
            "4. Batasi jam kerja secara ketat, hindari membuka pekerjaan di luar jam kantor.\n"
            "5. Lakukan aktivitas fisik ringan setiap hari, minimal jalan kaki santai.\n\n"
            "Kesehatan mentalmu jauh lebih berharga daripada tenggat waktu mana pun. Istirahatlah!"
        )
    elif label == "medium":
        return (
            "Tingkat burnout Anda SEDANG.\n"
            "1. Rapikan manajemen waktu dengan metode Pomodoro atau Timeblocking.\n"
            "2. Hindari lembur yang tidak perlu, kenali batas energimu.\n"
            "3. Gunakan akhir pekan untuk hobi dan kegiatan yang merilekskan pikiran.\n"
            "4. Sempatkan peregangan kecil dan istirahat mata (aturan 20-20-20).\n"
            "5. Katakan 'tidak' pada tugas tambahan jika kapasitasmu sudah penuh.\n\n"
            "Cegah burnout sebelum semakin parah dengan mulai mengatur keseimbangan hidup."
        )
    else:
        return (
            "Tingkat burnout Anda RENDAH.\n"
            "1. Pertahankan rutinitas kerja dan work-life balance saat ini.\n"
            "2. Jaga pola makan, tidur, dan olahraga yang teratur.\n"
            "3. Tetap waspada jika beban kerja tiba-tiba meningkat tajam.\n"
            "4. Lanjutkan praktik manajemen waktu yang sudah berjalan baik.\n"
            "5. Berikan apresiasi kepada diri sendiri atas pekerjaan yang diselesaikan.\n\n"
            "Bagus sekali! Pertahankan energimu dan teruslah berkarya secara sehat."
        )


def get_burnout_advice(prediction_result: dict, primary: str = "gemini", user_context: dict = None) -> tuple:
    prompt = build_advisory_prompt(prediction_result, user_context)

    def call_gemini():
        if GENAI_SDK_AVAILABLE and GEMINI_API_KEY:
            return get_burnout_advice_sdk(prompt), "Gemini (SDK)"
        elif GEMINI_API_KEY:
            return get_burnout_advice_rest(prompt), "Gemini (REST)"
        raise ValueError("GEMINI_API_KEY tidak tersedia.")

    def call_groq():
        return get_burnout_advice_groq(prompt), "Groq (REST)"

    try:
        return call_gemini() if primary == "gemini" else call_groq()
    except Exception as e:
        print(f"[GENAI WARNING] {primary.upper()} gagal ({e}). Mencoba fallback...")
        try:
            return call_groq() if primary == "gemini" else call_gemini()
        except Exception as fe:
            fallback_api = "GROQ" if primary == "gemini" else "GEMINI"
            print(f"[GENAI WARNING] {fallback_api} juga gagal ({fe}). Beralih ke Offline Mode...")
            return _offline_fallback(prediction_result.get("label", "low")), "Offline (Rule-Based)"


# Main
def main():
    parser = argparse.ArgumentParser(description="BurnAway Standalone CLI Inference")
    parser.add_argument("--age",        type=float, default=28.0,  help="Usia developer")
    parser.add_argument("--exp",        type=float, default=3.0,   help="Tahun pengalaman kerja")
    parser.add_argument("--work",       type=float, default=11.5,  help="Jam kerja per hari")
    parser.add_argument("--sleep",      type=float, default=5.2,   help="Jam tidur per hari")
    parser.add_argument("--caffeine",   type=float, default=5.0,   help="Konsumsi kafein per hari")
    parser.add_argument("--bugs",       type=float, default=14.0,  help="Jumlah bug per hari")
    parser.add_argument("--commits",    type=float, default=7.0,   help="Jumlah commit per hari")
    parser.add_argument("--meetings",   type=float, default=6.0,   help="Jumlah meeting per hari")
    parser.add_argument("--screen",     type=float, default=13.5,  help="Jam screen time per hari")
    parser.add_argument("--exercise",   type=float, default=0.3,   help="Jam olahraga per hari")
    parser.add_argument("--stress",     type=float, default=5.0,   help="Tingkat stres (skala)")
    parser.add_argument("--name",       type=str,   default=None,  help="Nama developer (opsional)")
    parser.add_argument("--role",       type=str,   default=None,  help="Role developer (opsional)")
    parser.add_argument("--primary",    type=str,   default="gemini", choices=["gemini", "groq"],
                        help="Provider GenAI utama (default: gemini)")
    args = parser.parse_args()

    # Load model & scaler
    print("[INFO] Loading model dan scaler...")
    try:
        model = tf.keras.models.load_model(
            MODEL_PATH,
            custom_objects={
                "BurnoutAttentionLayer":          BurnoutAttentionLayer,
                "WeightedCategoricalCrossentropy": WeightedCategoricalCrossentropy,
            }
        )
        scaler = joblib.load(SCALER_PATH)
        print("[OK] Model dan scaler berhasil dimuat.\n")
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        print("Pastikan notebook sudah dijalankan hingga selesai untuk menghasilkan model dan scaler.")
        return

    # Hitung engineered features
    work_sleep_ratio      = args.work / (args.sleep + 0.1)
    screen_time_intensity = args.screen / (args.work + 0.1)
    commit_bug_ratio      = args.commits / (args.bugs + 0.1)

    if work_sleep_ratio <= 1:
        work_category = 0.0
    elif work_sleep_ratio <= 2:
        work_category = 1.0
    elif work_sleep_ratio <= 3:
        work_category = 2.0
    else:
        work_category = 3.0

    raw_input = {
        "age":                  args.age,
        "experience_years":     args.exp,
        "daily_work_hours":     args.work,
        "sleep_hours":          args.sleep,
        "caffeine_intake":      args.caffeine,
        "bugs_per_day":         args.bugs,
        "commits_per_day":      args.commits,
        "meetings_per_day":     args.meetings,
        "screen_time":          args.screen,
        "exercise_hours":       args.exercise,
        "work_sleep_ratio":     work_sleep_ratio,
        "screen_time_intensity": screen_time_intensity,
        "commit_bug_ratio":     commit_bug_ratio,
        "work_category":        work_category,
        "stress_level":         args.stress,
    }

    # Tampilkan input
    print("[Input Features]")
    for k, v in raw_input.items():
        print(f"  {k:<25} : {v:.2f}")

    # Prediksi
    prediction = predict_burnout(raw_input, model, scaler)

    print("\n" + "=" * 55)
    print("              BURNOUT PREDICTION")
    print("=" * 55)
    print(f"  Predicted Level : {prediction['label']}")
    print(f"  Confidence      : {prediction['confidence']:.2%}")
    print("\n  Probability Distribution:")
    for label, prob in prediction["class_probs"].items():
        bar = "#" * int(prob * 40)
        print(f"    {label:<8} : {prob:.4f} |{bar}")
    print("=" * 55)

    # User context untuk GenAI
    user_context = None
    if args.name or args.role:
        user_context = {
            "name":       args.name or "Developer",
            "work_hours": args.work,
            "role":       args.role,
        }

    # GenAI Advisory
    print(f'\n[GENAI] Generating advice via {args.primary.upper()}...')
    advice, route = get_burnout_advice(prediction, primary=args.primary, user_context=user_context)
    print(f"[GENAI] Routing: {route} | Model: {GROQ_MODEL if 'Groq' in route else GEMINI_MODEL}")
    print(f"\n[REKOMENDASI UNTUK {prediction['label'].upper()} BURNOUT]")
    print("-" * 55)
    print(advice)
    print("=" * 55)


if __name__ == "__main__":
    main()
