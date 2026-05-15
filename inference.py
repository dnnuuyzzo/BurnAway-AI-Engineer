import os
import argparse
import numpy as np
import joblib
import tensorflow as tf
import google.generativeai as genai

# Model components
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


def main():
    parser = argparse.ArgumentParser(description="BurnAway Standalone Inference Script")
    parser.add_argument("--age", type=float, default=32.0, help="Age of the developer")
    parser.add_argument("--exp", type=float, default=8.0, help="Experience in years")
    parser.add_argument("--work", type=float, default=12.5, help="Daily work hours")
    parser.add_argument("--sleep", type=float, default=5.0, help="Sleep hours")
    parser.add_argument("--screen", type=float, default=16.0, help="Screen time hours")
    parser.add_argument("--stress", type=float, default=85.0, help="Stress level (0-100)")
    args = parser.parse_args()

    model_path = os.path.join("models", "burnaway_model.keras")
    scaler_path = os.path.join("models", "scaler.joblib")

    print("Loading Model and Scaler...")
    try:
        model = tf.keras.models.load_model(
            model_path,
            custom_objects={
                "FTTransformerBlock": FTTransformerBlock,
                "BurnoutFocalLoss": BurnoutFocalLoss,
            },
        )
        scaler = joblib.load(scaler_path)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure the notebook has been executed completely to generate the model and scaler in the 'models' directory.")
        return

    # Calculate engineered features
    # Defaults for other parameters not passed as args
    caffeine_intake = 6.0
    bugs_per_day = 14.0
    commits_per_day = 3.0
    meetings_per_day = 7.0
    exercise_hours = 0.1

    work_sleep_ratio = args.work / (args.sleep + 0.1)
    screen_time_intensity = args.screen / (args.work + 0.1)
    commit_bug_ratio = commits_per_day / (bugs_per_day + 0.1)

    if work_sleep_ratio <= 1:
        work_category = 0
    elif work_sleep_ratio <= 2:
        work_category = 1
    elif work_sleep_ratio <= 3:
        work_category = 2
    else:
        work_category = 3

    features = [
        args.age, args.exp, args.work, args.sleep, caffeine_intake,
        bugs_per_day, commits_per_day, meetings_per_day, args.screen,
        exercise_hours, args.stress,
        work_sleep_ratio, screen_time_intensity, commit_bug_ratio, float(work_category)
    ]
    
    feature_names = [
        "age", "experience_years", "daily_work_hours", "sleep_hours",
        "caffeine_intake", "bugs_per_day", "commits_per_day", "meetings_per_day",
        "screen_time", "exercise_hours", "stress_level",
        "work_sleep_ratio", "screen_time_intensity", "commit_bug_ratio", "work_category"
    ]

    print("\n[Input Features]")
    for n, v in zip(feature_names, features):
        print(f"  {n:<22} : {v:.2f}")

    feature_array = np.array(features, dtype=np.float32).reshape(1, -1)
    feature_scaled = scaler.transform(feature_array).astype(np.float32)

    # Multi-task inference
    outputs = model(feature_scaled, training=False)
    clf_logits = outputs[0].numpy()[0]
    reg_out = outputs[1].numpy()[0][0]

    probabilities = tf.nn.softmax(clf_logits).numpy()
    predicted_class = int(np.argmax(probabilities))
    confidence = float(np.max(probabilities))
    stress_est = float(reg_out) * 100
    
    label_map = {0: "Low", 1: "Medium", 2: "High"}
    burnout_level = label_map[predicted_class]

    print("\n=======================================================")
    print("                BURNOUT PREDICTION")
    print("=======================================================")
    print(f"  Predicted Level : {burnout_level}")
    print(f"  Confidence      : {confidence:.1%}")
    print(f"  Stress Estimate : {stress_est:.1f} / 100")
    print("\n  Probability Distribution:")
    for i, label in label_map.items():
        bar = "#" * int(probabilities[i] * 40)
        print(f"    {label:<8} : {probabilities[i]:.4f} |{bar}")
    print("=======================================================\n")

    api_key = os.environ.get("GEMINI_API_KEY")
    if api_key:
        print("Generating AI Advice using Gemini...")
        genai.configure(api_key=api_key)
        gemini_model = genai.GenerativeModel("gemini-1.5-flash")
        prompt = (
            f"Developer memprediksi burnout: {burnout_level} (kerja {args.work} jam, tidur {args.sleep} jam). "
            "Beri 3 saran singkat bahasa Indonesia."
        )
        try:
            res = gemini_model.generate_content(prompt)
            print("\n[AI Advice]\n" + res.text)
        except Exception as e:
            print(f"Failed to get AI advice: {e}")
    else:
        print("Tip: Set GEMINI_API_KEY environment variable to see AI advice.")

if __name__ == "__main__":
    main()
