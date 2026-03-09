# Ensure NLTK resources are available
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer

# Ensure required NLTK resources are downloaded
try:
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('vader_lexicon', quiet=True)
except Exception as e:
    print(f"Warning: NLTK download failed in utils.py: {e}")

# Standard state management
_sia = None
_stop_words_set = None


def _get_utils_tools():
    """Lazily load NLTK resources to avoid global scope overhead."""
    global _sia, _stop_words_set
    if _stop_words_set is None:
        _stop_words_set = set(stopwords.words('english'))
    if _sia is None:
        try:
            _sia = SentimentIntensityAnalyzer()
        except Exception:
            _sia = SentimentIntensityAnalyzer()
    return _sia, _stop_words_set


def get_numeric_features(df):
    """
    Phase 6 Feature Set:
    - Stylistic: Caps ratio, Excl/Ques density
    - Linguistic: Lexical Diversity, Word Len, Stopword Density
    - Emotional: Sentiment Intensity (Shock Value)
    - Structural: Proper Noun Density (Factuality markers)
    """
    features = []
    for text in df['raw_text']:
        if not isinstance(text, str) or len(text.strip()) == 0:
            features.append([0, 0, 0, 0, 0, 0, 0, 0])
            continue
        
        words = text.split()
        word_count = len(words)
        char_count = len(text)
        
        # 1. Stylistic
        caps_ratio = sum(1 for c in text if c.isupper()) / char_count
        excl_density = text.count('!') / char_count
        ques_density = text.count('?') / char_count
        
        # 2. Linguistic
        sia, stop_words_set = _get_utils_tools()
        unique_words = len(set(words))
        lex_div = unique_words / word_count if word_count > 0 else 0
        avg_word_len = sum(
            len(w) for w in words
        ) / word_count if word_count > 0 else 0
        stop_density = sum(
            1 for w in words if w.lower() in stop_words_set
        ) / word_count if word_count > 0 else 0
        
        # 3. Emotional Intensity (VADER)
        sentiment = sia.polarity_scores(text)
        sentiment_compound = sentiment['compound']
        
        # 4. Structural Factuality (Proper Noun Density)
        # Real news uses more specific names/places
        tokens = nltk.word_tokenize(text)
        pos_tags = nltk.pos_tag(tokens)
        proper_nouns = sum(
            1 for word, tag in pos_tags if tag in ('NNP', 'NNPS')
        )
        proper_noun_density = proper_nouns / word_count if word_count > 0 else 0
        
        features.append([
            caps_ratio, 
            excl_density, 
            ques_density, 
            lex_div, 
            avg_word_len, 
            stop_density,
            sentiment_compound,
            proper_noun_density
        ])
        
    return np.array(features)
