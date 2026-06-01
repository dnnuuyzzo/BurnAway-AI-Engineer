import json
import os

import joblib
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from google import genai
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address


app = FastAPI(
    title="BurnAway Inference API",
    description="API for predicting developer burnout and generating AI advice.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

model = None
scaler = None
FEATURE_COLS = []
LABEL_MAP = {}

API_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(API_DIR, "..", "models", "burnaway_model.keras")
SCALER_PATH = os.path.join(API_DIR, "..", "models", "scaler.joblib")
FEATURE_COLS_PATH = os.path.join(API_DIR, "feature_cols.json")
CLASS_LABELS_PATH = os.path.join(API_DIR, "class_labels.json")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.1-flash-lite")


@tf.keras.utils.register_keras_serializable(package="Custom")
class BurnoutAttentionLayer(tf.keras.layers.Layer):
    def __init__(self, units: int = 32, **kwargs):
        super().__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer="glorot_uniform",
            trainable=True,
            name="attention_kernel",
        )
        self.b = self.add_weight(
            shape=(self.units,),
            initializer="zeros",
            trainable=True,
            name="attention_bias",
        )
        self.attention_v = self.add_weight(
            shape=(self.units, 1),
            initializer="glorot_uniform",
            trainable=True,
            name="attention_v",
        )

    def call(self, inputs):
        score = tf.nn.tanh(tf.matmul(inputs, self.w) + self.b)
        attention_weights = tf.nn.softmax(tf.matmul(score, self.attention_v), axis=-1)
        return inputs * attention_weights

    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units})
        return config


@tf.keras.utils.register_keras_serializable(package="Custom")
class WeightedCategoricalCrossentropy(tf.keras.losses.Loss):
    def __init__(self, class_weights: list = None, label_smoothing: float = 0.0, **kwargs):
        super().__init__(**kwargs)
        weights = class_weights if class_weights is not None else [1.0, 1.0, 1.0]
        self.class_weights = tf.constant(weights, dtype=tf.float32)
        self.label_smoothing = tf.constant(label_smoothing, dtype=tf.float32)

    def call(self, y_true, y_pred):
        num_classes = int(self.class_weights.shape[0])
        y_true = tf.cast(y_true, tf.int32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0)
        y_onehot = tf.one_hot(y_true, depth=num_classes)
        y_smoothed = y_onehot * (1.0 - self.label_smoothing) + (
            self.label_smoothing / float(num_classes)
        )
        ce = -tf.reduce_sum(y_smoothed * tf.math.log(y_pred), axis=-1)
        sample_w = tf.gather(self.class_weights, y_true)
        return tf.reduce_mean(ce * sample_w)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "class_weights": self.class_weights.numpy().tolist(),
                "label_smoothing": self.label_smoothing.numpy().item(),
            }
        )
        return config


class DeveloperData(BaseModel):
    age: float
    experience_years: float
    daily_work_hours: float
    sleep_hours: float
    caffeine_intake: float
    bugs_per_day: float
    commits_per_day: float
    meetings_per_day: float
    screen_time: float
    exercise_hours: float
    stress_level: float


def load_feature_cols():
    with open(FEATURE_COLS_PATH, "r", encoding="utf-8") as file:
        feature_cols = json.load(file)
    if not isinstance(feature_cols, list) or not feature_cols:
        raise ValueError("feature_cols.json must contain a non-empty list.")
    return feature_cols


def load_label_map():
    with open(CLASS_LABELS_PATH, "r", encoding="utf-8") as file:
        raw_labels = json.load(file)
    if not isinstance(raw_labels, dict) or not raw_labels:
        raise ValueError("class_labels.json must contain a non-empty object.")
    return {int(index): str(label) for index, label in raw_labels.items()}


def artifacts_ready():
    return model is not None and scaler is not None and bool(FEATURE_COLS) and bool(LABEL_MAP)


@app.on_event("startup")
async def startup_event():
    global model, scaler, FEATURE_COLS, LABEL_MAP

    try:
        FEATURE_COLS = load_feature_cols()
        LABEL_MAP = load_label_map()
        model = tf.keras.models.load_model(
            MODEL_PATH,
            custom_objects={
                "BurnoutAttentionLayer": BurnoutAttentionLayer,
                "WeightedCategoricalCrossentropy": WeightedCategoricalCrossentropy,
            },
            compile=False,
        )
        scaler = joblib.load(SCALER_PATH)
        print(f"Model loaded from {MODEL_PATH}")
        print(f"Scaler loaded from {SCALER_PATH}")
        print(f"Loaded {len(FEATURE_COLS)} feature columns and {len(LABEL_MAP)} class labels.")
    except Exception as exc:
        model = None
        scaler = None
        FEATURE_COLS = []
        LABEL_MAP = {}
        raise RuntimeError(f"Failed to load BurnAway serving artifacts: {exc}") from exc


@app.get("/health")
def health_check():
    status = {
        "status": "healthy" if artifacts_ready() else "unhealthy",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None,
        "feature_cols_loaded": bool(FEATURE_COLS),
        "class_labels_loaded": bool(LABEL_MAP),
        "gemini_model": GEMINI_MODEL,
    }
    if not artifacts_ready():
        raise HTTPException(status_code=503, detail=status)
    return status


@app.post("/predict_burnout")
@limiter.limit("15/minute")
async def predict_burnout(request: Request, data: DeveloperData):
    if not artifacts_ready():
        raise HTTPException(status_code=503, detail="Model, scaler, or config not loaded.")

    work_sleep_ratio = data.daily_work_hours / (data.sleep_hours + 0.1)
    screen_time_intensity = data.screen_time / (data.daily_work_hours + 0.1)
    commit_bug_ratio = data.commits_per_day / (data.bugs_per_day + 0.1)

    if work_sleep_ratio <= 1:
        work_category = 0
    elif work_sleep_ratio <= 2:
        work_category = 1
    elif work_sleep_ratio <= 3:
        work_category = 2
    else:
        work_category = 3

    features_dict = data.model_dump()
    features_dict.update(
        {
            "work_sleep_ratio": work_sleep_ratio,
            "screen_time_intensity": screen_time_intensity,
            "commit_bug_ratio": commit_bug_ratio,
            "work_category": float(work_category),
        }
    )

    try:
        missing_features = [feature for feature in FEATURE_COLS if feature not in features_dict]
        if missing_features:
            raise ValueError(f"Missing features: {missing_features}")
        feature_array = np.array(
            [features_dict[feature] for feature in FEATURE_COLS], dtype=np.float32
        ).reshape(1, -1)
        feature_scaled = scaler.transform(feature_array).astype(np.float32)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Data processing error: {exc}") from exc

    try:
        probabilities = np.asarray(model.predict(feature_scaled, verbose=0)[0], dtype=np.float32)
        predicted_class = int(np.argmax(probabilities))
        confidence = float(np.max(probabilities))
        burnout_level = LABEL_MAP[predicted_class]
        prob_dict = {
            LABEL_MAP[i]: float(probabilities[i])
            for i in range(len(LABEL_MAP))
        }
        stress_estimate = data.stress_level
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Model prediction error: {exc}") from exc

    advice = ""
    api_key = os.environ.get("GEMINI_API_KEY")
    if api_key:
        try:
            client = genai.Client(api_key=api_key)
            prompt = (
                "Kamu adalah seorang psikolog industri berpengalaman yang membantu "
                "developer mengelola kesehatan kerja mereka. Berikan saran yang empatik "
                "namun langsung. Seorang software developer diprediksi memiliki tingkat burnout: "
                f"{burnout_level}. Developer memiliki jam kerja {data.daily_work_hours} jam, "
                f"waktu tidur {data.sleep_hours} jam, dan screen time {data.screen_time} jam sehari. "
                "Berikan 3 saran konkret dan personal untuk mengelola stres dan mencegah burnout. "
                "Format jawaban wajib menggunakan Markdown yang rapi: mulai dengan heading pendek, "
                "pakai numbered list, tebalkan kata kunci penting dengan **bold**, dan jangan gunakan HTML. "
                "Gunakan bahasa Indonesia yang natural dan tidak menggurui."
            )
            response = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
            advice = response.text
        except Exception as exc:
            advice = f"GenAI advice generation failed: {exc}"
    else:
        advice = "GEMINI_API_KEY environment variable not set. Advice generation skipped."

    return {
        "prediction": {
            "burnout_level": burnout_level,
            "confidence": confidence,
            "stress_estimate": round(float(stress_estimate), 2),
            "probabilities": prob_dict,
        },
        "engineered_features": {
            "work_sleep_ratio": work_sleep_ratio,
            "screen_time_intensity": screen_time_intensity,
            "commit_bug_ratio": commit_bug_ratio,
            "work_category": work_category,
        },
        "advice": advice,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
