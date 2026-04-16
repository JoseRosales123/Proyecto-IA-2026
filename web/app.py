from pathlib import Path
import sys

from flask import Flask, jsonify, request, render_template

BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from model_persistence import load_model
from preprocessing import preprocess_text
from config import MODEL_PATH

app = Flask(__name__)

model = None
model_load_error = None


def initialize_model():
    global model, model_load_error
    try:
        model = load_model(MODEL_PATH)
        model_load_error = None
        print(f"Modelo cargado correctamente desde: {MODEL_PATH}")
    except Exception as e:
        model = None
        model_load_error = str(e)
        print(f"Error al cargar el modelo: {e}")


initialize_model()


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok" if model is not None else "error",
        "model_loaded": model is not None,
        "model_path": str(MODEL_PATH),
        "error": model_load_error
    }), 200 if model is not None else 500


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({
            "error": "El modelo no está cargado.",
            "details": model_load_error,
            "suggestion": "Ejecuta primero el entrenamiento para generar el .pkl"
        }), 500

    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()

    if not text:
        return jsonify({"error": "Debes enviar un texto en el campo 'text'."}), 400

    tokens = preprocess_text(text)
    prediction = model.predict(tokens)
    log_scores = model.predict_log_proba(tokens)

    return jsonify({
        "input_text": text,
        "tokens": tokens,
        "predicted_category": prediction,
        "log_scores": log_scores
    })


if __name__ == "__main__":
    app.run(debug=True)