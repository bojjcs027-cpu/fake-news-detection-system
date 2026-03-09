import pickle
import os
import re
import pandas as pd
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn import calibration
import utils

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "dataset", "news.csv")
MODEL_DIR = os.path.join(BASE_DIR, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "vectorizer.pkl")

# Initialize NLTK tools
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))


def clean_text(text):
    """
    Cleans incoming text for TF-IDF processing.
    """
    if not isinstance(text, str):
        return ""
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

if __name__ == "__main__":
    print("Training the Phase 3 Fake News Detection Model. Please wait...")

    # Read the dataset
    data = pd.read_csv(DATA_PATH, encoding="utf-8")

    # Drop rows with missing values
    data = data.dropna(subset=["title", "text", "label"])

    # Store raw text for feature extraction
    data["raw_text"] = data["title"] + " " + data["text"]

    # Clean textual data for TF-IDF
    data["cleaned_text"] = data["raw_text"].apply(clean_text)

    # Prepare features and labels
    x = data[["raw_text", "cleaned_text"]]
    y = data["label"]

    # Split into train and test sets
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )

    # Define Feature Extraction Pipeline
    # 1. TF-IDF for cleaned text
    # Feature Engineering Pipeline
    preprocessor = ColumnTransformer([
        # Text path: Final push for extreme precision
        ('tfidf', TfidfVectorizer(
            stop_words="english",
            max_df=0.7,
            min_df=2,
            ngram_range=(1, 3),
            max_features=65000,
            sublinear_tf=True
        ), 'cleaned_text'),
        
        # Numeric path: Full style + emotional + structural analysis
        ('numeric', Pipeline([
            ('extractor', FunctionTransformer(utils.get_numeric_features)),
            ('scaler', MinMaxScaler())
        ]), x.columns)
    ])

    # Calibrated Base models
    estimators = [
        ('sgd', calibration.CalibratedClassifierCV(
            SGDClassifier(
                loss='hinge', penalty=None, learning_rate='pa1', 
                eta0=1.0, max_iter=1000, random_state=42
            ),
            cv=3
        )),
        ('nb', MultinomialNB(alpha=0.1)),
        ('svc', calibration.CalibratedClassifierCV(
            LinearSVC(random_state=42, C=0.1, dual=False),
            cv=3
        )),
        ('rf', RandomForestClassifier(
            n_estimators=300, max_depth=30, random_state=42
        ))
    ]

    # Stacking Classifier with Probability-Weighted Fusion
    model_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', StackingClassifier(
            estimators=estimators,
            final_estimator=RandomForestClassifier(
                n_estimators=200, random_state=42
            ),
            stack_method='predict_proba',
            cv=5
        ))
    ])

    print(
        "Fitting THE 100% OPTIMIZATION model... "
        "this will require significant processing."
    )
    model_pipeline.fit(x_train, y_train)

    # Evaluate model and print detailed report
    score = model_pipeline.score(x_test, y_test)
    y_pred = model_pipeline.predict(x_test)
    print(f"Model trained successfully with accuracy: {score:.2f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Save the full pipeline
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model_pipeline, f)

    print(f"Full Model Pipeline saved to: {MODEL_PATH}")
