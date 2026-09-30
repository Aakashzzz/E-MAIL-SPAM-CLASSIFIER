import os
import sys
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.download_data import download_data
from src.preprocess import load_and_clean_data, preprocess_message
from src.features import create_tfidf_features, transform_messages, save_vectorizer
from src.evaluate import compute_metrics, plot_confusion_matrices, plot_top_keywords

def main():
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(project_dir, 'data', 'spam.csv')
    models_dir = os.path.join(project_dir, 'models')
    outputs_dir = os.path.join(project_dir, 'outputs')
    
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(outputs_dir, exist_ok=True)
    
    # 1. Download data if it doesn't exist
    if not os.path.exists(data_path):
        download_data()
        
    # 2. Load and clean data
    df = load_and_clean_data(data_path)
    
    # 3. Text Preprocessing
    print("Preprocessing text messages (lowercasing, stopword removal, stemming)...")
    # Apply NLP preprocessing
    df['processed_message'] = df['message'].apply(preprocess_message)
    
    # Check for empty messages after preprocessing (e.g. messages that contained only stopwords or numbers)
    df = df[df['processed_message'].str.strip() != ''].copy()
    
    # 4. Train/Test Split (80/20, stratified)
    print("Splitting dataset into 80% train and 20% test, stratified by label...")
    X_train, X_test, y_train, y_test = train_test_split(
        df['processed_message'],
        df['label'],
        test_size=0.20,
        random_state=42,
        stratify=df['label']
    )
    
    print(f"Training set size: {len(X_train)} (Spam: {sum(y_train == 'spam')}, Ham: {sum(y_train == 'ham')})")
    print(f"Test set size: {len(X_test)} (Spam: {sum(y_test == 'spam')}, Ham: {sum(y_test == 'ham')})")
    
    # 5. Feature Extraction (TF-IDF vectorizer fitted ONLY on train data)
    X_train_vec, vectorizer = create_tfidf_features(X_train, max_features=3000)
    X_test_vec = transform_messages(vectorizer, X_test)
    
    # Save vectorizer
    vectorizer_path = os.path.join(models_dir, 'vectorizer.pkl')
    save_vectorizer(vectorizer, vectorizer_path)
    
    # 6. Train all 3 models
    print("\nTraining models...")
    models = {
        'Multinomial Naive Bayes': MultinomialNB(),
        'Logistic Regression': LogisticRegression(C=1.0, max_iter=1000, random_state=42),
        'Support Vector Machine': SVC(kernel='linear', probability=True, random_state=42)
    }
    
    trained_models = {}
    evaluation_results = {}
    model_predictions = {}
    
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train_vec, y_train)
        trained_models[name] = model
        
        # Predictions
        y_pred = model.predict(X_test_vec)
        
        # Predict probabilities (if supported) for ROC-AUC
        if hasattr(model, "predict_proba"):
            # probability of 'spam' (which is the second class usually, check classes_)
            spam_idx = list(model.classes_).index('spam')
            y_prob = model.predict_proba(X_test_vec)[:, spam_idx]
        else:
            y_prob = None
            
        model_predictions[name] = {
            'y_true': y_test,
            'y_pred': y_pred
        }
        
        # Compute metrics
        metrics = compute_metrics(y_test, y_pred, y_prob)
        evaluation_results[name] = metrics
        
        # Print classification report
        from sklearn.metrics import classification_report
        print(f"\n--- {name} Classification Report ---")
        print(classification_report(y_test, y_pred, digits=4))
        
    # 7. Compare Models
    comparison_df = pd.DataFrame(evaluation_results).T
    print("\n=== Model Comparison Table ===")
    print(comparison_df.to_string())
    
    # 8. Selection and Saving of the Best Model
    # Choose best model based on F1-Score (since we have class imbalance)
    # If F1-Scores are extremely close, we look at Precision (low false-positive rate is critical)
    best_model_name = comparison_df['F1-Score (Spam)'].idxmax()
    best_model = trained_models[best_model_name]
    
    print(f"\nBest Model selected based on F1-Score: {best_model_name}")
    
    # Double check if another model has significantly better precision (lower false positive rate)
    # which is critical for email systems.
    for name, metrics in evaluation_results.items():
        if name != best_model_name:
            # If the F1-Score is within 1% and the alternative model has better Precision:
            f1_diff = best_model_metrics_f1 = evaluation_results[best_model_name]['F1-Score (Spam)'] - metrics['F1-Score (Spam)']
            if f1_diff < 0.01 and metrics['Precision (Spam)'] > evaluation_results[best_model_name]['Precision (Spam)']:
                print(f"NOTE: {name} is within 1% of the best F1-score but has higher Precision ({metrics['Precision (Spam)']:.4f} vs {evaluation_results[best_model_name]['Precision (Spam)']:.4f}).")
                print("In spam filtering, minimizing false positives (marking ham as spam) is highly critical.")
                print(f"Switching final model selection to: {name}")
                best_model_name = name
                best_model = trained_models[name]
                break
                
    best_model_path = os.path.join(models_dir, 'best_model.pkl')
    joblib.dump(best_model, best_model_path)
    print(f"\nSaved best model ({best_model_name}) to {best_model_path}")
    
    # Write choice metadata file
    metadata_path = os.path.join(models_dir, 'best_model_metadata.txt')
    with open(metadata_path, 'w') as f:
        f.write(f"Model Name: {best_model_name}\n")
        f.write(f"F1-Score: {evaluation_results[best_model_name]['F1-Score (Spam)']:.4f}\n")
        f.write(f"Precision: {evaluation_results[best_model_name]['Precision (Spam)']:.4f}\n")
        f.write(f"Recall: {evaluation_results[best_model_name]['Recall (Spam)']:.4f}\n")
        
    # 9. Plot and save Confusion Matrices
    cm_path = os.path.join(outputs_dir, 'confusion_matrices.png')
    plot_confusion_matrices(model_predictions, cm_path)
    
    # 10. Visualize Keyword Importance for the Best Model
    kw_path = os.path.join(outputs_dir, 'top_spam_keywords.png')
    plot_top_keywords(best_model, vectorizer, best_model_name, kw_path, top_n=15)
    
    print("\nTraining pipeline execution complete.")

if __name__ == '__main__':
    main()
