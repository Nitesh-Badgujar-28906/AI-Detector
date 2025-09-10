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
model = joblib.load("model.pkl")
txt = st.text_area(
    "Text to analyze",
    placeholder="Enter text here..."
)
import re
stemmer=PorterStemmer()
stop_words = set(stopwords.words('english'))
def text_preprocessor(text):
  text = re.sub(r'[^a-zA-Z\s]', '', text)
  words=word_tokenize(text)
  filtered_words=[word.lower() for word in words if word.lower() not in stop_words]
  filtered_steamed_words=[stemmer.stem(word) for word in filtered_words]

  return ' '.join(filtered_steamed_words)
processed_txt = text_preprocessor(txt)
if st.button("test"):

    prediction = model.predict([processed_txt])
    if prediction[0] == 1:
        st.write("AI generated")
    else:
        st.write("Human")
