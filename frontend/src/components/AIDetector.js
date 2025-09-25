import React, { useState } from 'react';
import axios from 'axios';
import { toast } from 'react-toastify';
import '../styles/AIDetector.css';

const AIDetector = () => {
  const [text, setText] = useState('');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  // Use proxy for API calls in development, direct URL for production
  const API_URL = process.env.NODE_ENV === 'production' ? '/api' : '';

  const handleTextChange = (e) => {
    setText(e.target.value);
    if (error) setError('');
    if (result) setResult(null);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (!text.trim()) {
      setError('Please enter some text to analyze');
      toast.error('Please enter some text to analyze');
      return;
    }

    if (text.trim().length < 10) {
      setError('Please provide at least 10 characters for accurate prediction');
      toast.error('Please provide at least 10 characters for accurate prediction');
      return;
    }

    setLoading(true);
    setError('');
    setResult(null);

    console.log('API URL:', API_URL); // Debug log

    try {
      const response = await axios.post(`${API_URL}/predict`, {
        text: text
      }, {
        headers: {
          'Content-Type': 'application/json',
        },
        timeout: 30000, // 30 seconds timeout
      });

      setResult(response.data);
      toast.success('Analysis completed successfully!');
    } catch (err) {
      console.error('API Error:', err);
      console.error('API URL was:', API_URL);
      
      let errorMessage = 'Failed to analyze text. Please try again.';
      
      if (err.code === 'ERR_NETWORK' || err.message.includes('Network Error')) {
        errorMessage = `Network Error: Cannot connect to API at ${API_URL}. Please ensure the Flask backend is running.`;
      } else if (err.response?.data?.error) {
        errorMessage = err.response.data.error;
      } else if (err.message) {
        errorMessage = err.message;
      }
      
      setError(errorMessage);
      toast.error(errorMessage);
    } finally {
      setLoading(false);
    }
  };

  const handleClear = () => {
    setText('');
    setResult(null);
    setError('');
  };

  const getConfidenceColor = (confidence) => {
    if (confidence >= 80) return '#10b981'; // green
    if (confidence >= 60) return '#f59e0b'; // yellow
    return '#ef4444'; // red
  };

  return (
    <div className="ai-detector-container">
      <div className="header">
        <h1 className="title">
          🤖 AI Detection Hub 🔍
        </h1>
        <p className="subtitle">
          Where humans & AI meet — and we figure out who's who! 😉
        </p>
      </div>

      <div className="main-content">
        <div className="input-section">
          <form onSubmit={handleSubmit}>
            <div className="textarea-container">
              <label htmlFor="text-input" className="label">
                📝 Enter text to analyze:
              </label>
              <textarea
                id="text-input"
                value={text}
                onChange={handleTextChange}
                placeholder="Type or paste your text here..."
                className={`textarea ${error ? 'error' : ''}`}
                rows={8}
                disabled={loading}
              />
              <div className="character-count">
                {text.length} characters
              </div>
            </div>

            {error && (
              <div className="error-message">
                ⚠️ {error}
              </div>
            )}

            <div className="button-group">
              <button
                type="submit"
                className="analyze-button"
                disabled={loading || !text.trim()}
              >
                {loading ? (
                  <>
                    <span className="spinner"></span>
                    🔍 Analyzing...
                  </>
                ) : (
                  '🚀 Run Detection'
                )}
              </button>

              <button
                type="button"
                onClick={handleClear}
                className="clear-button"
                disabled={loading}
              >
                🗑️ Clear
              </button>
            </div>
          </form>
        </div>

        {result && (
          <div className="result-section">
            <h2 className="result-title">📊 Analysis Result</h2>
            
            <div className={`result-card ${result.is_ai ? 'ai-result' : 'human-result'}`}>
              <div className="result-header">
                <div className="result-icon">
                  {result.is_ai ? '🤖' : '🧑‍💻'}
                </div>
                <div className="result-text">
                  <h3>{result.prediction}</h3>
                  <p className="result-description">
                    {result.is_ai 
                      ? 'This text appears to be generated by AI'
                      : 'This text appears to be written by a human'
                    }
                  </p>
                </div>
              </div>

              <div className="confidence-section">
                <h4>Confidence Levels:</h4>
                <div className="confidence-bars">
                  <div className="confidence-item">
                    <div className="confidence-label">
                      <span>🤖 AI Generated</span>
                      <span className="confidence-value">{result.confidence.ai}%</span>
                    </div>
                    <div className="confidence-bar">
                      <div 
                        className="confidence-fill"
                        style={{ 
                          width: `${result.confidence.ai}%`,
                          backgroundColor: getConfidenceColor(result.confidence.ai)
                        }}
                      ></div>
                    </div>
                  </div>

                  <div className="confidence-item">
                    <div className="confidence-label">
                      <span>🧑‍💻 Human Written</span>
                      <span className="confidence-value">{result.confidence.human}%</span>
                    </div>
                    <div className="confidence-bar">
                      <div 
                        className="confidence-fill"
                        style={{ 
                          width: `${result.confidence.human}%`,
                          backgroundColor: getConfidenceColor(result.confidence.human)
                        }}
                      ></div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        <div className="info-section">
          <div className="info-card">
            <h3>ℹ️ How it works</h3>
            <ol>
              <li>Preprocess the input text (clean, remove stopwords, stem)</li>
              <li>Run the text through our trained ML model</li>
              <li>Predict whether it's AI-generated or human-written</li>
            </ol>
            <p className="disclaimer">
              <strong>Note:</strong> This is a probabilistic model and may not be 100% accurate.
            </p>
          </div>

          <div className="about-section">
            <h3>⚡ About This App</h3>
            <p>
              This tool analyzes text and predicts whether it's <strong>AI-generated</strong> or <strong>human-written</strong>.
            </p>
            <p className="powered-by">
              ✅ Powered by Machine Learning
            </p>
            <div style={{ marginTop: '10px', fontSize: '0.8rem', color: '#6b7280' }}>
              API Endpoint: {API_URL}
            </div>
          </div>
        </div>
      </div>

      <footer className="footer">
        <div className="made-by">
          👨‍💻 Made by <strong>Nitesh Badgujar</strong>
        </div>
      </footer>
    </div>
  );
};

export default AIDetector;