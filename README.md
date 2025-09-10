# AI-Detector (AI Detection Hub)

A simple Streamlit app that predicts whether input text is AI‑generated or human‑written using a scikit‑learn model. The app preprocesses text (cleanup, stopword removal, stemming) and runs inference with a joblib‑serialized model.

## Features

- Streamlit UI with a text area and one‑click prediction
- Text preprocessing: regex cleanup, tokenization, stopword removal, stemming (NLTK)
- scikit‑learn model loaded from `models/model.pkl`
- Lightweight and easy to run locally or in Codespaces

## Project structure

```
app.py                 # Streamlit app
requirements.txt       # Python dependencies
models/
	model.pkl            # Trained classification model (joblib)
dataset/
	archive.zip          # Dataset artifact (optional, for training)
notebooks/
	AI_Detector.ipynb    # Exploration/training notebook (optional)
```

## Requirements

- Python 3.9+
- Internet access on first run (to download NLTK resources like `punkt` and `stopwords`)

## Setup

Create a virtual environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

If you’re using VS Code, the workspace is configured to pick `.venv` automatically.

Optional: pre-download NLTK data to avoid runtime prompts:

```bash
python - << 'PY'
import nltk
nltk.download('punkt')
nltk.download('stopwords')
PY
```

## Run

```bash
source .venv/bin/activate
streamlit run app.py
```

If you’re on a remote environment (e.g., Codespaces) and need relaxed CORS/XSRF for previews:

```bash
streamlit run app.py --server.enableCORS false --server.enableXsrfProtection false
```

## Use

1. Paste or type text into the text area.
2. Click “Run Detection”.
3. See the predicted label and a simple confidence indicator.

Notes:
- The model is probabilistic; treat results as guidance, not ground truth.
- The app performs basic preprocessing before inference.

## Model

- Path: `models/model.pkl`
- Format: joblib (scikit‑learn pipeline or estimator)
- Training: See `notebooks/AI_Detector.ipynb` (if provided) for exploration/training; the app itself does not train.

## Troubleshooting

- FileNotFoundError for model
	- Ensure `models/model.pkl` exists and the path in `app.py` matches.
	- If your model has a different filename, update the load path in `app.py` accordingly.

- NLTK resource not found (e.g., `punkt` or `stopwords`)
	- Ensure internet access on first run so `nltk.download(...)` can fetch data, or pre-download using the snippet above.

- Version mismatch loading the model
	- Use the dependencies from `requirements.txt`. If the pickle was created with a very different scikit‑learn version, re‑export the model with this environment or retrain.

- Streamlit doesn’t open a browser automatically
	- Open the URL shown in the terminal (default http://localhost:8501). In remote environments, use the forwarded port.

## License

No license file is included. Add one if you plan to distribute.
