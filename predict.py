import pickle
import os
import re
import pandas as pd
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import utils

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "vectorizer.pkl")

# Load compiled model pipeline
model_pipeline = None

try:
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            model_pipeline = pickle.load(f)
        print("Model pipeline loaded successfully.")
    else:
        print("Warning: Model pipeline NOT found. Please run train_model.py first.")
except Exception as e:
    print(f"Error loading model files: {e}")

# Initialize NLTK tools (lazily loaded to avoid IDE startup warnings)
_lemmatizer = None
_stop_words = None


def _get_nltk_tools():
    """Helper to get NLTK tools without global overhead."""
    global _lemmatizer, _stop_words
    if _lemmatizer is None:
        _lemmatizer = WordNetLemmatizer()
    if _stop_words is None:
        _stop_words = set(stopwords.words('english'))
    return _lemmatizer, _stop_words


def clean_text(text):
    """Clean the text for prediction."""
    if not isinstance(text, str):
        return ""
    
    lemmatizer, stop_words = _get_nltk_tools()
    
    # Lowercase but keep numbers and common punctuation indicators
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\!\?\s]', '', text)
    
    # Tokenize, remove stopwords, and lemmatize
    words = text.split()
    cleaned_words = [
        lemmatizer.lemmatize(word) for word in words 
        if word not in stop_words
    ]
    
    return " ".join(cleaned_words)


def predict_news(news_text):
    """
    Predicts if a given news string is FAKE or REAL using the Phase 3 pipeline.
    """
    if not model_pipeline:
        return "ERROR: Model not trained or not found"
    
    # Create input DataFrame to match Phase 3 pipeline requirements
    input_df = pd.DataFrame([{
        'raw_text': news_text,
        'cleaned_text': clean_text(news_text)
    }])
        
    # Predict using the full pipeline (includes preprocessing)
    try:
        prediction = model_pipeline.predict(input_df)
        return prediction[0]
    except Exception as e:
        print(f"Prediction error: {e}")
        return f"ERROR: {str(e)}"
