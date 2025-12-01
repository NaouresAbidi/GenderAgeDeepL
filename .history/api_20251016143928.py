import os
import io
import json
import threading
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

PORT = int(os.environ.get("PORT", 5000))
HOST = os.environ.get("HOST", "0.0.0.0")
MODEL_PATH = os.environ.get("MODEL_PATH", "best_age_gender_model_children_tuned.h5")

IMAGE_SIZE = (360, 360)
NUM_CHANNELS = 1  

gender_mapping = {0: "Male", 1: "Female"}

try:
    gpus = tf.config.experimental.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
except Exception:
    pass

@tf.function(input_signature=[tf.TensorSpec(shape=[], dtype=tf.string)])
def preprocess_from_path(path):
    img_raw = tf.io.read_file(path)
    img = tf.image.decode_image(img_raw, channels=NUM_CHANNELS, expand_animations=False)
    img = tf.image.resize(img, IMAGE_SIZE, method=tf.image.ResizeMethod.BILINEAR)
    img = tf.cast(img, tf.float32) / 255.0
    img = (img * 2.0) - 1.0         # [-1, 1]
    img = tf.expand_dims(img, axis=0)  # add batch
    return img

@tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=tf.uint8)])
def preprocess_from_bytes(content):
    img_raw = tf.io.decode_image(content, channels=NUM_CHANNELS, expand_animations=False)
    img = tf.image.resize(img_raw, IMAGE_SIZE, method=tf.image.ResizeMethod.BILINEAR)
    img = tf.cast(img, tf.float32) / 255.0
    img = (img * 2.0) - 1.0         # [-1, 1]
    img = tf.expand_dims(img, axis=0)
    return img

_model_lock = threading.Lock()
_model = None

def load_model():
    global _model
    with _model_lock:
        if _model is None:
            _model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    return _model

def predict_tensor(batch_tensor):

    model = load_model()
    preds = model(batch_tensor, training=False)

    age_pred = float(preds[0][0][0].numpy())
    gender_prob = float(preds[1][0][0].numpy())
    gender_label = 1 if gender_prob > 0.5 else 0

    out = {
        "age": int(np.round(age_pred)),
        "gender": gender_mapping[gender_label],
        "gender_probability": gender_prob,
        "is_child_under_18": bool(int(np.round(age_pred)) < 18),
    }
    return out


app = Flask(__name__)

@app.route("/health", methods=["GET"])
def health():
    try:
        load_model()
        return jsonify({"status": "ok", "model": os.path.basename(MODEL_PATH)})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/predict", methods=["POST"])
def predict():

    try:
        if "image" in request.files:
            file_storage = request.files["image"]
            filename = secure_filename(file_storage.filename or "upload")
            content = file_storage.read()
            if not content:
                return jsonify({"error": "Empty file"}), 400

            img_tensor = preprocess_from_bytes(np.frombuffer(content, dtype=np.uint8))
            result = predict_tensor(img_tensor)
            result["source"] = {"type": "upload", "filename": filename}
            return jsonify(result)

        if request.is_json:
            payload = request.get_json(silent=True) or {}
            image_path = payload.get("image_path")
            if not image_path:
                return jsonify({"error": "Missing 'image_path' in JSON body"}), 400
            if not os.path.exists(image_path):
                return jsonify({"error": f"File not found: {image_path}"}), 404

            img_tensor = preprocess_from_path(tf.constant(image_path, dtype=tf.string))
            result = predict_tensor(img_tensor)
            result["source"] = {"type": "path", "path": image_path}
            return jsonify(result)

        return jsonify({"error": "Provide an 'image' file (multipart) or JSON with 'image_path'."}), 400

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "endpoints": {
            "GET /health": "Model status",
            "POST /predict": "Send 'image' file (multipart/form-data) or JSON {'image_path': '...'}"
        },
        "model_path": MODEL_PATH,
        "image_requirements": {
            "channels": "grayscale (auto-converted)",
            "size": "360x360 (auto-resized)",
            "normalization": "[-1, 1]"
        }
    })

if __name__ == "__main__":
    load_model()
    app.run(host=HOST, port=PORT, debug=False)
