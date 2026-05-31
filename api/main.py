from fastapi import FastAPI, HTTPException, Request
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import tensorflow as tf
import numpy as np
import joblib
import os
import google.generativeai as genai

# Setup FastAPI App
app = FastAPI(
    title="BurnAway Inference API",
    description="API for predicting developer burnout and generating AI advice.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup Rate Limiting untuk proteksi API dan meminimalisir spam (Gemini API protection)
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Global variables for model and scaler
model = None
scaler = None

# Constants
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "burnaway_model.keras")
SCALER_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "scaler.joblib")
LABEL_MAP = {0: "Low", 1: "Medium", 2: "High"}

# Feature definitions
RAW_FEATURES = [
    "age", "experience_years", "daily_work_hours", "sleep_hours",
    "caffeine_intake", "bugs_per_day", "commits_per_day", "meetings_per_day",
    "screen_time", "exercise_hours", "stress_level"
]

ALL_FEATURES = RAW_FEATURES + [
    "work_sleep_ratio", "screen_time_intensity", "commit_bug_ratio", "work_category"
]

# Custom Components from Training
@tf.keras.utils.register_keras_serializable(package="Custom")
class FTTransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim=16, num_heads=2, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads

    def build(self, input_shape):
        self.n_features = input_shape[-1]
        self.feature_weights = self.add_weight(
            shape=(self.n_features, self.embed_dim),
            initializer="glorot_uniform", trainable=True, name="tokenizer_weights"
        )
        self.feature_biases = self.add_weight(
            shape=(self.n_features, self.embed_dim),
            initializer="zeros", trainable=True, name="tokenizer_biases"
        )
        self.mha = tf.keras.layers.MultiHeadAttention(num_heads=self.num_heads, key_dim=self.embed_dim)
        self.layernorm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.flatten = tf.keras.layers.Flatten()
        super().build(input_shape)

    def call(self, inputs):
        x = tf.expand_dims(inputs, -1) * tf.expand_dims(self.feature_weights, 0) + tf.expand_dims(self.feature_biases, 0)
        attn_out = self.mha(x, x)
        x = self.layernorm(x + attn_out)
        return self.flatten(x)

    def get_config(self):
        config = super().get_config()
        config.update({"embed_dim": self.embed_dim, "num_heads": self.num_heads})
        return config

@tf.keras.utils.register_keras_serializable(package="Custom")
class BurnoutFocalLoss(tf.keras.losses.Loss):
    def __init__(self, gamma=2.0, alpha=0.25, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.alpha = alpha

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1.0 - tf.keras.backend.epsilon())
        cross_entropy = -y_true * tf.math.log(y_pred)
        weight = self.alpha * tf.math.pow((1.0 - y_pred), self.gamma)
        return tf.reduce_mean(tf.reduce_sum(weight * cross_entropy, axis=-1))

    def get_config(self):
        config = super().get_config()
        config.update({"gamma": self.gamma, "alpha": self.alpha})
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

@app.on_event("startup")
async def startup_event():
    global model, scaler
    try:
        model = tf.keras.models.load_model(
            MODEL_PATH,
            custom_objects={
                "FTTransformerBlock": FTTransformerBlock,
                "BurnoutFocalLoss": BurnoutFocalLoss
            }
        )
        print(f"Model loaded from {MODEL_PATH}")
        
        scaler = joblib.load(SCALER_PATH)
        print(f"Scaler loaded from {SCALER_PATH}")
    except Exception as e:
        print(f"Warning: Failed to load model or scaler. Exception: {e}")

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None
    }

@app.post("/predict_burnout")
@limiter.limit("5/minute")
async def predict_burnout(request: Request, data: DeveloperData):
    if model is None or scaler is None:
        raise HTTPException(status_code=500, detail="Model or Scaler not loaded.")
    
    # 1. Feature Engineering
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
        
    features_dict = data.dict()
    features_dict["work_sleep_ratio"] = work_sleep_ratio
    features_dict["screen_time_intensity"] = screen_time_intensity
    features_dict["commit_bug_ratio"] = commit_bug_ratio
    features_dict["work_category"] = float(work_category)
    
    # 2. Convert to array and scale
    try:
        feature_array = np.array([features_dict[f] for f in ALL_FEATURES], dtype=np.float32).reshape(1, -1)
        feature_scaled = scaler.transform(feature_array).astype(np.float32)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Data processing error: {e}")

    # 3. Predict
    try:
        # Multi-task inference: [classification_output, regression_output]
        outputs = model(feature_scaled, training=False)
        clf_logits = outputs[0].numpy()[0]
        reg_out = outputs[1].numpy()[0][0]

        # Apply softmax to logits for classification
        probabilities = tf.nn.softmax(clf_logits).numpy()
        predicted_class = int(np.argmax(probabilities))
        confidence = float(np.max(probabilities))
        burnout_level = LABEL_MAP[predicted_class]
        prob_dict = {LABEL_MAP[i]: float(probabilities[i]) for i in range(len(LABEL_MAP))}
        stress_estimate = float(reg_out) * 100 # Scaling back to 0-100 if needed
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model prediction error: {e}")

    # 4. GenAI Advice
    advice = ""
    api_key = os.environ.get("GEMINI_API_KEY")
    gemini_model_name = os.environ.get("GEMINI_MODEL", "gemini-3.1-flash-lite")
    if api_key:
        try:
            genai.configure(api_key=api_key)
            gemini_model = genai.GenerativeModel(gemini_model_name)
            prompt = (
                "Kamu adalah seorang psikolog industri berpengalaman yang membantu "
                "developer mengelola kesehatan kerja mereka. Berikan saran yang empatik "
                "namun langsung. Seorang software developer diprediksi memiliki tingkat burnout: "
                f"{burnout_level}. Developer memiliki jam kerja {data.daily_work_hours} jam, "
                f"waktu tidur {data.sleep_hours} jam, dan screen time {data.screen_time} jam sehari. "
                "Berikan 3 saran konkret dan personal (masing-masing 1-2 kalimat) untuk mengelola stres "
                "dan mencegah burnout. Gunakan bahasa Indonesia yang natural dan tidak menggurui."
            )
            response = gemini_model.generate_content(prompt)
            advice = response.text
        except Exception as e:
            advice = f"GenAI advice generation failed: {e}"
    else:
        advice = "GEMINI_API_KEY environment variable not set. Advice generation skipped."

    return {
        "prediction": {
            "burnout_level": burnout_level,
            "confidence": confidence,
            "stress_estimate": round(stress_estimate, 2),
            "probabilities": prob_dict
        },
        "engineered_features": {
            "work_sleep_ratio": work_sleep_ratio,
            "screen_time_intensity": screen_time_intensity,
            "commit_bug_ratio": commit_bug_ratio,
            "work_category": work_category
        },
        "advice": advice
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
