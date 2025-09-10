import streamlit as st
import joblib
import re
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

# Load Model
model = joblib.load("models/model.pkl")

# Title Section
st.markdown("""
# 🤖 AI Detection Hub 🔍
Where humans & AI meet — and we figure out who's who! 😉
---
""")

# Sidebar for Extra Info
st.sidebar.title("⚡ About This App")
st.sidebar.info(
    "This tool analyzes text and predicts whether it's **AI-generated** or **human-written**."
)
st.sidebar.success("✅ Powered by Machine Learning")

# Input Text Area
txt = st.text_area(
    "📝 Enter text to analyze:",
    placeholder="Type or paste your text here..."
)

# Preprocessing
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def text_preprocessor(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = word_tokenize(text)
    filtered_words = [word.lower() for word in words if word.lower() not in stop_words]
    filtered_steamed_words = [stemmer.stem(word) for word in filtered_words]
    return ' '.join(filtered_steamed_words)

processed_txt = text_preprocessor(txt)

# Prediction
if st.button("🚀 Run Detection"):
    with st.spinner("🔍 Analyzing... Please wait!"):
        prediction = model.predict([processed_txt])

    st.subheader("📊 Result:")
    if prediction[0] == 1:
        st.success("🤖 This looks **AI Generated**!")
        st.metric("Confidence", "High", "↑")
    else:
        st.error("🧑‍💻 This looks **Human Written**!")
        st.metric("Confidence", "Medium", "↓")

# Expander for Extra Info
with st.expander("ℹ️ How it works"):
    st.write("""
    1. Preprocess the input text (clean, remove stopwords, stem).  
    2. Run the text through our trained ML model.  
    3. Predict whether it's AI-generated or human-written.  
    """)
    st.write("**Note:** This is a probabilistic model and may not be 100% accurate.")

    # Signature in bottom-left corner
st.markdown("""
    <style>
    .made-by {
        position: fixed;
        left: 10px;
        bottom: 10px;
        font-size: 14px;
        color: gray;
        font-style: italic;
    }
    </style>
    <div class="made-by">👨‍💻 Made by <b>Nitesh Badgujar</b></div>
""", unsafe_allow_html=True)
