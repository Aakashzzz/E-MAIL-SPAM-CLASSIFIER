import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

def compute_metrics(y_true, y_pred, y_prob=None):
    """
    Computes classification metrics, mapping labels to 0 and 1.
    ham = 0, spam = 1.
    """
    # Map labels to numeric if they are strings
    y_true_num = np.where(y_true == 'spam', 1, 0)
    y_pred_num = np.where(y_pred == 'spam', 1, 0)
    
    metrics = {
        'Accuracy': accuracy_score(y_true_num, y_pred_num),
        'Precision (Spam)': precision_score(y_true_num, y_pred_num, zero_division=0),
        'Recall (Spam)': recall_score(y_true_num, y_pred_num, zero_division=0),
        'F1-Score (Spam)': f1_score(y_true_num, y_pred_num, zero_division=0),
    }
    
    if y_prob is not None:
        metrics['ROC-AUC'] = roc_auc_score(y_true_num, y_prob)
    else:
        metrics['ROC-AUC'] = np.nan
        
    return metrics

def plot_confusion_matrices(models_results, save_path):
    """
    Plots a multi-panel figure containing the confusion matrix for each trained model.
    models_results is a dict: {model_name: {'y_true': y_true, 'y_pred': y_pred}}
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    n_models = len(models_results)
    
    # Establish a premium aesthetic style
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4.5))
    
    if n_models == 1:
        axes = [axes]
        
    for ax, (name, res) in zip(axes, models_results.items()):
        cm = confusion_matrix(res['y_true'], res['y_pred'], labels=['ham', 'spam'])
        
        # Calculate percentages
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        
        # Create annotation labels with both count and percentage
        labels = [
            [f"{cm[0,0]}\n({cm_percent[0,0]:.1f}%)", f"{cm[0,1]}\n({cm_percent[0,1]:.1f}%)"],
            [f"{cm[1,0]}\n({cm_percent[1,0]:.1f}%)", f"{cm[1,1]}\n({cm_percent[1,1]:.1f}%)"]
        ]
        
        sns.heatmap(
            cm, 
            annot=labels, 
            fmt="", 
            cmap='Blues', 
            cbar=False,
            xticklabels=['Ham (Legit)', 'Spam'], 
            yticklabels=['Ham (Legit)', 'Spam'],
            ax=ax,
            annot_kws={"size": 12, "weight": "bold"}
        )
        
        ax.set_title(f"{name} Confusion Matrix", fontsize=13, pad=10, weight='semibold')
        ax.set_xlabel('Predicted Label', fontsize=11)
        ax.set_ylabel('True Label', fontsize=11)
        
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Confusion matrices saved successfully to {save_path}")

def plot_top_keywords(model, vectorizer, model_name, save_path, top_n=15):
    """
    Plots the top N words that most strongly indicate Spam.
    Supported models: MultinomialNB (using log probability ratios),
    LogisticRegression (using positive coefficients), LinearSVC (using coefficients).
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    feature_names = vectorizer.get_feature_names_out()
    
    importance = []
    
    # Extract feature importance depending on model type
    class_name = model.__class__.__name__
    
    if 'LogisticRegression' in class_name:
        # Coefs represent log-odds. Larger coefficients mean strong spam predictors.
        coefs = model.coef_
        if hasattr(coefs, "toarray"):
            coefs = coefs.toarray()
        importance = coefs.ravel()
        title = "Logistic Regression Coefficients (Spam Indicator)"
    elif 'SVC' in class_name:
        # Coefs represent linear weights in SVM decision boundary
        coefs = model.coef_
        if hasattr(coefs, "toarray"):
            coefs = coefs.toarray()
        importance = coefs.ravel()
        title = "Linear SVM Coefficients (Spam Indicator)"
    elif 'MultinomialNB' in class_name:
        # Compute difference in log probabilities: log(P(w | spam)) - log(P(w | ham))
        # feature_log_prob_[0] -> ham log prob, feature_log_prob_[1] -> spam log prob
        log_prob_diff = model.feature_log_prob_[1] - model.feature_log_prob_[0]
        importance = log_prob_diff
        title = "Naive Bayes Log Probability Difference (Spam vs Ham)"
    else:
        print(f"Model type {class_name} not directly supported for keyword visualization.")
        return
        
    # Combine with feature names
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    })
    
    # Sort to find top spam words (largest positive values)
    top_spam_words = feature_importance.sort_values(by='Importance', ascending=False).head(top_n)
    
    # Premium bar plot
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")
    
    # Create gradient of colors
    colors = sns.color_palette("rocket", n_colors=top_n)
    
    sns.barplot(
        x='Importance', 
        y='Feature', 
        data=top_spam_words, 
        palette=colors,
        hue='Feature',
        legend=False
    )
    
    plt.title(f"Top {top_n} Spam-Indicating Words ({model_name})", fontsize=14, weight='bold', pad=15)
    plt.xlabel('Importance (Model Metric)', fontsize=12)
    plt.ylabel('Preprocessed Word', fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=11)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Top keywords plot saved to {save_path}")
