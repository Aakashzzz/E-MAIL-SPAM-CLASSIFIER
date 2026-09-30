import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

def create_tfidf_features(train_messages, max_features=3000):
    """
    Creates and fits a TF-IDF vectorizer on the training data.
    Uses both unigrams and bigrams.
    """
    print(f"Creating TF-IDF features with max_features={max_features} (unigrams + bigrams)...")
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=max_features,
        sublinear_tf=True # Use sublinear scaling (1 + log(tf)) to reduce influence of very frequent words
    )
    
    # Fit vectorizer and transform text
    X_train_vectorized = vectorizer.fit_transform(train_messages)
    return X_train_vectorized, vectorizer

def transform_messages(vectorizer, messages):
    """
    Transforms text messages using a fitted TF-IDF vectorizer.
    """
    return vectorizer.transform(messages)

def save_vectorizer(vectorizer, save_path):
    """
    Saves the fitted vectorizer to a joblib file.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump(vectorizer, save_path)
    print(f"Vectorizer successfully saved to {save_path}")

def load_vectorizer(load_path):
    """
    Loads a vectorizer from a joblib file.
    """
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"No vectorizer found at {load_path}")
    return joblib.load(load_path)
