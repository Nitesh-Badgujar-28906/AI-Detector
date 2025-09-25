# AI Text Detector - React + Flask

A modern web application that detects whether text is AI-generated or human-written using machine learning.

## Architecture

- **Backend**: Flask API with ML model
- **Frontend**: React.js with modern UI
- **ML Model**: Pre-trained scikit-learn model

## Features

- 🤖 **AI Detection**: Analyzes text to determine if it's AI-generated or human-written
- 📊 **Confidence Scores**: Shows probability percentages for both classifications
- 🎨 **Modern UI**: Beautiful, responsive React interface
- ⚡ **Real-time Analysis**: Fast predictions with loading states
- 🔧 **Error Handling**: Comprehensive error messages and validation
- 📱 **Responsive Design**: Works on desktop, tablet, and mobile

## Setup Instructions

### Backend Setup

1. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Flask API**:
   ```bash
   python flask_app.py
   ```
   The API will start on `http://localhost:5000`

### Frontend Setup

1. **Navigate to frontend directory**:
   ```bash
   cd frontend
   ```

2. **Install Node.js dependencies**:
   ```bash
   npm install
   ```

3. **Start the React development server**:
   ```bash
   npm start
   ```
   The frontend will start on `http://localhost:3000`

## API Endpoints

### `GET /`
Health check endpoint
- **Response**: Status information and model health

### `POST /predict`
Analyze text for AI detection
- **Request Body**: 
  ```json
  {
    "text": "Your text to analyze here..."
  }
  ```
- **Response**:
  ```json
  {
    "original_text": "Your text...",
    "processed_text": "preprocessed version...",
    "prediction": "AI Generated" | "Human Written",
    "is_ai": true | false,
    "confidence": {
      "human": 25.6,
      "ai": 74.4
    },
    "status": "success"
  }
  ```

### `GET /health`
Detailed health check with system status

## Usage

1. **Start the Backend**: Run `python flask_app.py`
2. **Start the Frontend**: In `/frontend` directory, run `npm start`
3. **Open Browser**: Navigate to `http://localhost:3000`
4. **Analyze Text**: 
   - Enter or paste text (minimum 10 characters)
   - Click "🚀 Run Detection"
   - View results with confidence scores

## Development

### Project Structure
```
AI-Detector/
├── flask_app.py          # Flask API server
├── app.py               # Original Streamlit app
├── model.pkl            # Trained ML model
├── requirements.txt     # Python dependencies
├── frontend/            # React application
│   ├── package.json     # Node.js dependencies
│   ├── public/          # Static files
│   └── src/             # React source code
│       ├── components/  # React components
│       ├── styles/      # CSS files
│       ├── App.js       # Main App component
│       └── index.js     # Entry point
└── notebooks/           # Jupyter notebooks
```

### Environment Variables

**Flask Backend**:
- `PORT`: Server port (default: 5000)
- `FLASK_ENV`: Set to 'development' for debug mode

**React Frontend**:
- `NODE_ENV`: Automatically set by React scripts
- API calls automatically proxy to Flask backend in development

## Production Deployment

### Backend (Flask)
```bash
# Set environment variables
export FLASK_ENV=production
export PORT=5000

# Install dependencies
pip install -r requirements.txt

# Run the server
python flask_app.py
```

### Frontend (React)
```bash
cd frontend

# Build for production
npm run build

# Serve the build folder using any static server
# Example with serve:
npm install -g serve
serve -s build -l 3000
```

## Features in Detail

### Text Preprocessing
- Removes special characters and punctuation
- Converts to lowercase
- Removes stop words
- Applies Porter Stemming
- Tokenization using NLTK

### UI Components
- **Text Area**: Large input field with character counter
- **Loading States**: Spinner animation during analysis
- **Result Display**: Color-coded results with confidence bars
- **Error Handling**: User-friendly error messages
- **Responsive Design**: Mobile-first approach

### API Features
- **CORS Support**: Cross-origin requests enabled
- **Input Validation**: Comprehensive text validation
- **Error Handling**: Detailed error responses
- **Timeout Handling**: 30-second request timeout
- **Health Checks**: Multiple health check endpoints

## Troubleshooting

### Common Issues

1. **NLTK Download Errors**:
   ```bash
   python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
   ```

2. **CORS Issues**: Ensure Flask-CORS is installed and configured

3. **Port Conflicts**: Change ports in configuration if 3000 or 5000 are in use

4. **Model Loading**: Ensure `model.pkl` exists in the root directory

### Development Tips

- Backend and frontend run on different ports during development
- React proxy configuration automatically forwards API calls to Flask
- Use browser developer tools to debug API calls
- Check Flask console for backend errors

## Made by Nitesh Badgujar

This project combines machine learning with modern web technologies to create an intuitive AI detection tool.