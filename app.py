from pathlib import Path
import pickle

import numpy as np
import tensorflow as tf
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from flask import Flask, render_template, request
from PIL import Image, UnidentifiedImageError
from werkzeug.utils import secure_filename


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "gender_cnn.keras"
CLASS_INDICES_PATH = BASE_DIR / "class_indices.pkl"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

model = tf.keras.models.load_model(MODEL_PATH)
with CLASS_INDICES_PATH.open("rb") as file:
    class_indices = pickle.load(file)
labels_by_index = {index: label for label, index in class_indices.items()}


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def predict_image(image):
    image = image.convert("RGB").resize((128, 128))
    pixels = np.asarray(image, dtype=np.float32) / 255.0
    score = float(model.predict(np.expand_dims(pixels, axis=0), verbose=0)[0][0])
    index = 1 if score >= 0.5 else 0
    return labels_by_index[index], score if index == 1 else 1 - score


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    error = None

    if request.method == "POST":
        uploaded_file = request.files.get("image")
        if uploaded_file is None or not uploaded_file.filename:
            error = "Choose an image before running the prediction."
        elif not allowed_file(uploaded_file.filename):
            error = "Use a PNG, JPG, JPEG, or WEBP image."
        else:
            try:
                image = Image.open(uploaded_file.stream)
                label, confidence = predict_image(image)
                result = {
                    "label": label,
                    "display_label": label.capitalize(),
                    "confidence": round(confidence * 100, 1),
                    "filename": secure_filename(uploaded_file.filename),
                }
            except (UnidentifiedImageError, OSError):
                error = "That file could not be read as an image."

    return render_template("index.html", result=result, error=error)


if __name__ == "__main__":
    #app.run(debug=True)
    app.run( host="0.0.0.0", port=5000, debug=False )
