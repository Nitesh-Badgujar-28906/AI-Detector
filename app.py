from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import joblib
import re
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
import os

# ---------------------------
# NLTK Setup
# ---------------------------
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# ---------------------------
# Flask Setup
# ---------------------------
app = Flask(__name__, static_folder="build", static_url_path="/")
CORS(app)

# Load trained model
try:
    model = joblib.load("model.pkl")
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"⚠️ Error loading model: {e}")
    model = None

# Stemmer & Stopwords
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))


# ---------------------------
# Helper: Text Preprocessing
# ---------------------------
def text_preprocessor(text):
    """
    Preprocess text:
    - Remove non-alphabetic characters
    - Tokenize
    - Lowercase, remove stopwords
    - Apply stemming
    """
    if not text or text.strip() == "":
        return ""

    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = word_tokenize(text)
    filtered_words = [word.lower() for word in words if word.lower() not in stop_words]
    filtered_stemmed_words = [stemmer.stem(word) for word in filtered_words]

    return ' '.join(filtered_stemmed_words)


# ---------------------------
# API Routes
# ---------------------------
@app.route("/api/", methods=["GET"])
def home():
    return jsonify({
        "message": "AI Detector API is running!",
        "status": "healthy",
        "model_loaded": model is not None
    })


@app.route("/api/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded."}), 500

    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "Please provide 'text' field in JSON body"}), 400

    text = data["text"]
    if not text.strip() or len(text.strip()) < 10:
        return jsonify({"error": "Provide at least 10 characters of meaningful text"}), 400

    processed_text = text_preprocessor(text)
    if not processed_text.strip():
        return jsonify({"error": "Preprocessing removed all content. Provide more meaningful input."}), 400

    # Prediction
    prediction = model.predict([processed_text])

    try:
        probabilities = model.predict_proba([processed_text])[0]
        human_prob, ai_prob = float(probabilities[0]), float(probabilities[1])
    except Exception:
        human_prob, ai_prob = (0.8, 0.2) if prediction[0] == 0 else (0.2, 0.8)

    result = {
        "original_text": text,
        "processed_text": processed_text,
        "prediction": "AI Generated" if prediction[0] == 1 else "Human Written",
        "is_ai": bool(prediction[0] == 1),
        "confidence": {
            "human": round(human_prob * 100, 2),
            "ai": round(ai_prob * 100, 2)
        },
        "status": "success"
    }

    return jsonify(result)


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "nltk_ready": True,
        "version": "1.0.0"
    })


# ---------------------------
# React Frontend Routes
# ---------------------------
@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve_react(path):
    if path != "" and os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    return send_from_directory(app.static_folder, "index.html")


# ---------------------------
# Run the app
# ---------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_ENV") == "development"

    print(f"🚀 Starting AI Detector Flask API on port {port}")
    print(f"🔧 Debug mode: {debug}")
    print(f"📦 Model loaded: {model is not None}")

    app.run(host="0.0.0.0", port=port, debug=debug)
