import re
import string
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

# Automatically download NLTK dependencies if not already present
def download_nltk_resources():
    for resource in ['stopwords', 'punkt']:
        try:
            nltk.data.find(f'corpora/{resource}' if resource == 'stopwords' else f'tokenizers/{resource}')
        except LookupError:
            print(f"NLTK resource '{resource}' not found. Downloading...")
            nltk.download(resource, quiet=True)

# Run the download when importing/running the module
download_nltk_resources()

def clean_text(text):
    """
    Cleans raw text message:
    - Converts to lowercase
    - Removes URLs
    - Removes punctuation
    - Removes numbers
    """
    if not isinstance(text, str):
        return ""
    
    # 1. Lowercase
    text = text.lower()
    
    # 2. Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # 3. Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # 4. Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # 5. Remove numbers
    text = re.sub(r'\d+', '', text)
    
    # 6. Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def preprocess_message(text):
    """
    Complete NLP preprocessing pipeline:
    - Cleaning
    - Tokenization
    - Stopword removal
    - Stemming (Porter Stemmer)
    """
    # Clean text first
    cleaned = clean_text(text)
    
    # Tokenize
    try:
        tokens = word_tokenize(cleaned)
    except Exception:
        # Fallback split if word_tokenize fails
        tokens = cleaned.split()
        
    # Stopwords removal and Stemming
    stemmer = PorterStemmer()
    stop_words = set(stopwords.words('english'))
    
    processed_tokens = [
        stemmer.stem(word) for word in tokens 
        if word not in stop_words
    ]
    
    return " ".join(processed_tokens)

def load_and_clean_data(csv_path):
    """
    Loads dataset, handles encoding, drops unused columns, renames to label/message,
    drops nulls and duplicates.
    """
    print(f"Loading data from {csv_path}...")
    
    # Try different encodings as the spam collection dataset has special characters
    encodings = ['latin-1', 'cp1252', 'utf-8']
    df = None
    for enc in encodings:
        try:
            df = pd.read_csv(csv_path, encoding=enc)
            print(f"Successfully loaded dataset with {enc} encoding.")
            break
        except UnicodeDecodeError:
            continue
            
    if df is None:
        raise ValueError(f"Could not load file at {csv_path} with any of the standard encodings.")
        
    # Inspect columns
    print(f"Original columns: {df.columns.tolist()}")
    
    # Keep only target and message columns
    # In UCI/Kaggle, these are usually 'v1' and 'v2'
    if 'v1' in df.columns and 'v2' in df.columns:
        df = df[['v1', 'v2']].copy()
        df.rename(columns={'v1': 'label', 'v2': 'message'}, inplace=True)
    elif 'label' in df.columns and 'message' in df.columns:
        df = df[['label', 'message']].copy()
    else:
        # If columns are named differently, pick first two columns
        df = df.iloc[:, [0, 1]].copy()
        df.columns = ['label', 'message']
        
    # Drop rows with null values
    initial_len = len(df)
    df.dropna(subset=['label', 'message'], inplace=True)
    null_dropped = initial_len - len(df)
    if null_dropped > 0:
        print(f"Dropped {null_dropped} rows with null labels/messages.")
        
    # Drop duplicate rows
    initial_len = len(df)
    df.drop_duplicates(subset=['message'], inplace=True)
    duplicates_dropped = initial_len - len(df)
    if duplicates_dropped > 0:
        print(f"Dropped {duplicates_dropped} duplicate rows.")
        
    # Map label to standard categories (ham/spam)
    df['label'] = df['label'].astype(str).str.lower().str.strip()
    df = df[df['label'].isin(['ham', 'spam'])].copy()
    
    print(f"Cleaned dataset: {len(df)} rows remaining.")
    print(f"Class distribution:\n{df['label'].value_counts(normalize=True) * 100}")
    
    return df
