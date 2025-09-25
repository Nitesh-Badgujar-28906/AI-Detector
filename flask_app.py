from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import re
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
import os

# Download NLTK data if not already present
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

app = Flask(__name__)
CORS(app)  # Enable CORS for all domains

# Load the trained model
try:
    model = joblib.load("model.pkl")
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Initialize stemmer and stop words
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def text_preprocessor(text):
    """
    Preprocesses the input text by:
    1. Removing non-alphabetic characters
    2. Tokenizing the text
    3. Converting to lowercase and removing stop words
    4. Stemming the words
    """
    if not text or text.strip() == "":
        return ""
    
    # Remove non-alphabetic characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Tokenize
    words = word_tokenize(text)
    
    # Filter and stem
    filtered_words = [word.lower() for word in words if word.lower() not in stop_words]
    filtered_stemmed_words = [stemmer.stem(word) for word in filtered_words]
    
    return ' '.join(filtered_stemmed_words)

@app.route('/', methods=['GET'])
def home():
    """Health check endpoint"""
    return jsonify({
        "message": "AI Detector API is running!",
        "status": "healthy",
        "model_loaded": model is not None
    })

@app.route('/predict', methods=['POST'])
def predict():
    """
    Predicts whether the given text is AI-generated or human-written
    """
    try:
        # Check if model is loaded
        if model is None:
            return jsonify({
                "error": "Model not loaded. Please check the model file."
            }), 500
        
        # Get JSON data from request
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({
                "error": "Please provide 'text' field in JSON body"
            }), 400
        
        text = data['text']
        
        # Validate input
        if not text or text.strip() == "":
            return jsonify({
                "error": "Text cannot be empty"
            }), 400
        
        if len(text.strip()) < 10:
            return jsonify({
                "error": "Please provide at least 10 characters for accurate prediction"
            }), 400
        
        # Preprocess the text
        processed_text = text_preprocessor(text)
        
        if not processed_text or processed_text.strip() == "":
            return jsonify({
                "error": "Text preprocessing resulted in empty text. Please provide more meaningful content."
            }), 400
        
        # Make prediction
        prediction = model.predict([processed_text])
        
        # Get prediction probabilities if available
        try:
            probabilities = model.predict_proba([processed_text])[0]
            human_prob = float(probabilities[0])
            ai_prob = float(probabilities[1])
        except:
            # If predict_proba is not available, use basic confidence
            human_prob = 0.8 if prediction[0] == 0 else 0.2
            ai_prob = 0.8 if prediction[0] == 1 else 0.2
        
        # Prepare response
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
        
    except Exception as e:
        return jsonify({
            "error": f"An error occurred during prediction: {str(e)}",
            "status": "error"
        }), 500

@app.route('/health', methods=['GET'])
def health():
    """Detailed health check"""
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "nltk_ready": True,
        "version": "1.0.0"
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV') == 'development'
    
    print(f"Starting AI Detector Flask API on port {port}")
    print(f"Debug mode: {debug}")
    print(f"Model loaded: {model is not None}")
    
    app.run(host='0.0.0.0', port=port, debug=debug)