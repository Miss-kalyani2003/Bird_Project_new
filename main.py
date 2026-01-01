from flask import Flask, request, jsonify,render_template

import os
from src.utils import load_trained_model, load_class_names, predict_image

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


app = Flask(__name__)

# Config
UPLOAD_FOLDER = "uploads"
MODEL_PATH = "artifacts/parrot_mobilenet_model.keras"
CLASS_NAMES_PATH = "artifacts/class_names.json"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load once (IMPORTANT)
model = load_trained_model(MODEL_PATH)
class_names = load_class_names(CLASS_NAMES_PATH)

@app.route("/")
def home():
    return "Parrot Classification API is running"

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]

    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    image_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(image_path)

    label, confidence = predict_image(model, class_names, image_path)

    return jsonify({
        "prediction": label,
        "confidence": f"{confidence:.2f}%"
    })

@app.route("/", methods=["GET", "POST"])
def ui():
    prediction = None
    confidence = None

    if request.method == "POST":
        file = request.files.get("image")
        if file and file.filename != "":
            image_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(image_path)

            prediction, confidence = predict_image(
                model, class_names, image_path
            )

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence
    )


if __name__ == "__main__":
    app.run(debug=True)
