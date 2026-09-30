import os
import sys
import argparse
import joblib

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocess import preprocess_message
from src.features import transform_messages

def get_best_model_name(models_dir):
    metadata_path = os.path.join(models_dir, 'best_model_metadata.txt')
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if line.startswith("Model Name:"):
                    return line.split(":", 1)[1].strip()
    return "Trained Model"

def predict(message_text):
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(project_dir, 'models')
    
    vectorizer_path = os.path.join(models_dir, 'vectorizer.pkl')
    best_model_path = os.path.join(models_dir, 'best_model.pkl')
    
    if not os.path.exists(vectorizer_path) or not os.path.exists(best_model_path):
        print("Error: Model files not found. Please run the training pipeline first: python src/train.py")
        sys.exit(1)
        
    # Load assets
    vectorizer = joblib.load(vectorizer_path)
    model = joblib.load(best_model_path)
    model_name = get_best_model_name(models_dir)
    
    # Preprocess message
    processed = preprocess_message(message_text)
    
    # Vectorize
    vectorized = transform_messages(vectorizer, [processed])
    
    # Predict
    prediction = model.predict(vectorized)[0]
    
    # Predict Confidence (Probability)
    confidence = 100.0
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(vectorized)[0]
        classes = list(model.classes_)
        pred_idx = classes.index(prediction)
        confidence = probs[pred_idx] * 100
        
    return prediction, confidence, model_name

def main():
    parser = argparse.ArgumentParser(description="Classify a text message as Spam or Ham (legitimate).")
    parser.add_argument("message", type=str, help="The raw message text to classify.")
    args = parser.parse_args()
    
    print("\n" + "="*50)
    print("Email/SMS Spam Classifier CLI")
    print("="*50)
    print(f"Message: \"{args.message}\"")
    print("-"*50)
    
    label, confidence, model_name = predict(args.message)
    
    print(f"Prediction: {label.upper()}")
    print(f"Confidence: {confidence:.2f}%")
    print(f"Model Used: {model_name}")
    print("="*50 + "\n")

if __name__ == '__main__':
    main()
