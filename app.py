import io
import os
from flask import Flask, request, jsonify, send_from_directory
from PIL import Image
from model import predict

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@app.route("/")
def home():
    return send_from_directory(BASE_DIR, "index.html")

@app.route("/predict", methods=["POST"])
def predict_route():
    file = request.files["file"]
    file_bytes = file.read()
    image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    result, confidence = predict(image)
    return jsonify({
        "prediction": result.capitalize(),
        "confidence": confidence
    })

if __name__ == "__main__":
    app.run(debug=True)
