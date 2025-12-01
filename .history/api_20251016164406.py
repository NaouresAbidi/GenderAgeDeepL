import os
import json
import threading
import numpy as np
import tensorflow as tf
import cv2
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

# ---------- Config ----------
PORT = int(os.environ.get("PORT", 5000))
HOST = os.environ.get("HOST", "0.0.0.0")
MODEL_PATH = os.environ.get("MODEL_PATH", "best_age_gender_model_children_tuned.h5")

IMAGE_SIZE = (360, 360)
NUM_CHANNELS = 1  # model was trained on grayscale
GENDER_MAPPING = {0: "Male", 1: "Female"}

# Quality thresholds
MIN_FACE_SIDE = 120           # px on the shortest side AFTER crop-before-resize check
BLUR_VAR_THRESHOLD = 80.0     # variance of Laplacian heuristic

# ---------- Optional GPU mem growth ----------
try:
    gpus = tf.config.experimental.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
except Exception:
    pass

# ---------- Face detector (Haar) ----------
_HAAR = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def _detect_and_crop_face_bgr(bgr):
    """Detect largest face; return cropped BGR face and bbox (x,y,w,h). Fallback to centered square."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    faces = _HAAR.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    if len(faces) == 0:
        # fallback: center square crop
        h, w = bgr.shape[:2]
        s = min(h, w)
        y0 = (h - s) // 2
        x0 = (w - s) // 2
        crop = bgr[y0:y0 + s, x0:x0 + s]
        return crop, (x0, y0, s, s)
    # choose largest face
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    # padding margin around face
    m = int(0.2 * max(w, h))
    x0 = max(0, x - m)
    y0 = max(0, y - m)
    x1 = min(bgr.shape[1], x + w + m)
    y1 = min(bgr.shape[0], y + h + m)
    crop = bgr[y0:y1, x0:x1]
    return crop, (x, y, w, h)

def _blur_variance(gray_img):
    # variance of Laplacian—simple blur heuristic
    return float(cv2.Laplacian(gray_img, cv2.CV_64F).var())

def _bgr_to_model_tensor(face_bgr, to_gray=True):
    """Resize, optional grayscale, scale to [-1,1], add batch -> (1,H,W,C) float32"""
    face_resized = cv2.resize(face_bgr, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
    if to_gray:
        face_resized = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)[..., None]  # (H,W,1)
    else:
        face_resized = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)               # (H,W,3)
    face_resized = face_resized.astype(np.float32) / 255.0
    face_resized = (face_resized * 2.0) - 1.0
    return tf.convert_to_tensor(face_resized[None, ...], dtype=tf.float32)

def _load_bgr_from_bytes(content: bytes):
    arr = np.frombuffer(content, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)  # keep alpha if present
    if img is None:
        raise ValueError("Unable to decode image bytes")
    return _to_bgr_3ch(img)

def _load_bgr_from_path(path: str):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)   # keep alpha if present
    if img is None:
        raise ValueError(f"Unable to read image from path: {path}")
    return _to_bgr_3ch(img)


def _pipeline_from_bgr(bgr):
    """Returns (tensor, quality_info dict)."""
    crop_bgr, bbox = _detect_and_crop_face_bgr(bgr)
    # quality (before resize)
    h, w = crop_bgr.shape[:2]
    short_side = min(h, w)
    gray_for_blur = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    blur_var = _blur_variance(gray_for_blur)
    quality = {
        "bbox": {"x": int(bbox[0]), "y": int(bbox[1]), "w": int(bbox[2]), "h": int(bbox[3])},
        "face_short_side_px": int(short_side),
        "blur_variance": blur_var,
        "passed": (short_side >= MIN_FACE_SIDE) and (blur_var >= BLUR_VAR_THRESHOLD),
        "reasons": []
    }
    if short_side < MIN_FACE_SIDE:
        quality["reasons"].append("small_face")
    if blur_var < BLUR_VAR_THRESHOLD:
        quality["reasons"].append("blurry")

    tensor = _bgr_to_model_tensor(crop_bgr, to_gray=(NUM_CHANNELS == 1))
    return tensor, quality
# ---------- Image with alpha channel PNG to BGR ----------

def _to_bgr_3ch(img):
    """
    Ensure 3-channel BGR image.
    - If BGRA: alpha-composite over white then drop alpha.
    - If GRAY: convert to BGR.
    - If already BGR: return as-is.
    """
    if img is None:
        raise ValueError("Decoded image is None")

    if len(img.shape) == 2:  # GRAY
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    if img.shape[2] == 4:  # BGRA
        b, g, r, a = cv2.split(img)
        alpha = a.astype(np.float32) / 255.0
        # composite over white background
        bg = np.ones_like(b, dtype=np.float32) * 255.0
        b = (b.astype(np.float32) * alpha + bg * (1.0 - alpha)).astype(np.uint8)
        g = (g.astype(np.float32) * alpha + bg * (1.0 - alpha)).astype(np.uint8)
        r = (r.astype(np.float32) * alpha + bg * (1.0 - alpha)).astype(np.uint8)
        return cv2.merge([b, g, r])

    if img.shape[2] == 3:  # BGR
        return img

    # Fallback for unusual channel counts
    return cv2.cvtColor(img, cv2.COLOR_BGR2BGR)

# ---------- Model loading / prediction ----------
_model_lock = threading.Lock()
_model = None

def load_model():
    global _model
    with _model_lock:
        if _model is None:
            _model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    return _model

def _raw_predict(batch_tensor):
    model = load_model()
    preds = model(batch_tensor, training=False)
    # preds[0]: age (B,1), preds[1]: gender prob (B,1)
    age_pred = float(preds[0][0][0].numpy())
    gender_prob = float(preds[1][0][0].numpy())
    return age_pred, gender_prob

def _format_output(age_pred, gender_prob):
    age_int = int(np.round(age_pred))
    if 0.4 < gender_prob < 0.6:
        gender_label = "Unknown"
    else:
        gender_label = GENDER_MAPPING[1 if gender_prob > 0.5 else 0]
    return {
        "age": age_int,
        "gender": gender_label,
        "gender_probability": gender_prob,
        "is_child_under_18": bool(age_int < 18),
    }

# ---------- Flask app ----------
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
        # --- Multipart upload ---
        if "image" in request.files:
            fs = request.files["image"]
            filename = secure_filename(fs.filename or "upload")
            content = fs.read()
            if not content:
                return jsonify({"error": "Empty file"}), 400

            bgr = _load_bgr_from_bytes(content)
            tensor, quality = _pipeline_from_bgr(bgr)

            if not quality["passed"]:
                return jsonify({
                    "status": "low_quality",
                    "reason": quality["reasons"],
                    "quality": quality
                }), 422

            age_pred, gender_prob = _raw_predict(tensor)
            result = _format_output(age_pred, gender_prob)
            result["source"] = {"type": "upload", "filename": filename}
            result["quality"] = quality
            return jsonify(result)

        # --- JSON with local path ---
        if request.is_json:
            payload = request.get_json(silent=True) or {}
            image_path = payload.get("image_path")
            if not image_path:
                return jsonify({"error": "Missing 'image_path' in JSON body"}), 400
            if not os.path.exists(image_path):
                return jsonify({"error": f"File not found: {image_path}"}), 404

            bgr = _load_bgr_from_path(image_path)
            tensor, quality = _pipeline_from_bgr(bgr)

            if not quality["passed"]:
                return jsonify({
                    "status": "low_quality",
                    "reason": quality["reasons"],
                    "quality": quality,
                    "source": {"type": "path", "path": image_path}
                }), 422

            age_pred, gender_prob = _raw_predict(tensor)
            result = _format_output(age_pred, gender_prob)
            result["source"] = {"type": "path", "path": image_path}
            result["quality"] = quality
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
            "channels_expected_by_model": "grayscale (auto-converted)",
            "size": f"{IMAGE_SIZE[0]}x{IMAGE_SIZE[1]} (auto-resized)",
            "normalization": "[-1, 1]"
        }
    })

if __name__ == "__main__":
    load_model()
    app.run(host=HOST, port=PORT, debug=False)
