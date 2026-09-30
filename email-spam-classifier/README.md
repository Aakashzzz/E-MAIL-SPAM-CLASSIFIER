# SentriSpam AI - Email & SMS Spam Classifier

A production-grade, supervised machine learning system that classifies email and SMS messages into **Spam** or **Ham (Legitimate)**. The system is built in Python, using NLP text extraction (TF-IDF) and scikit-learn classifiers.

## Tech Stack
*   **Language**: Python 3.10+
*   **Machine Learning**: scikit-learn
*   **Data Handling**: pandas, NumPy
*   **NLP**: NLTK (Tokenization, Stopwords, Porter Stemmer)
*   **Visualization**: Matplotlib, Seaborn
*   **Application UI**: Streamlit

---

## Project Directory Structure

```text
email-spam-classifier/
├── data/
│   └── spam.csv           # Raw dataset
├── models/
│   ├── vectorizer.pkl     # Saved TF-IDF Vectorizer
│   ├── best_model.pkl     # Winning classifier model
│   └── best_model_metadata.txt # Details of the selected model
├── notebooks/
│   └── eda.ipynb          # Jupyter Notebook for Exploratory Data Analysis
├── outputs/
│   ├── confusion_matrices.png  # Heatmaps for Naive Bayes, Logistic Regression, SVM
│   └── top_spam_keywords.png   # Visualisation of key spam indicator words
├── src/
│   ├── download_data.py   # Utility script to acquire dataset
│   ├── preprocess.py      # Text processing pipeline (stopword removal, stemming)
│   ├── features.py        # TF-IDF feature extraction
│   ├── train.py           # Training, comparison, selection, and saving
│   ├── evaluate.py        # Custom metric computing & plotting functions
│   ├── predict.py         # CLI inference script
│   └── create_notebook.py # Programmatic generator for eda.ipynb
├── app.py                 # Live Streamlit application for real-time predictions
├── requirements.txt       # Pinned library dependencies
└── README.md              # Documentation & metrics
```

---

## Setup & Installation

1.  **Clone / Copy the directory** and navigate to the project directory:
    ```bash
    cd email-spam-classifier
    ```

2.  **Create a Virtual Environment**:
    ```bash
    python -m venv .venv
    ```

3.  **Activate the Virtual Environment**:
    *   **Windows (PowerShell)**:
        ```powershell
        .venv\Scripts\Activate.ps1
        ```
    *   **Windows (CMD)**:
        ```cmd
        .venv\Scripts\activate.bat
        ```
    *   **macOS / Linux**:
        ```bash
        source .venv/bin/activate
        ```

4.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

---

## Running the Project

### 1. Execute Training Pipeline
This script downloads the dataset, runs text cleaning, performs feature extraction (TF-IDF), trains Multinomial Naive Bayes, Logistic Regression, and Linear SVM models, and selects the best model.
```bash
python src/train.py
```

### 2. Generate Jupyter Notebook (Optional)
Generates the pre-configured EDA notebook `notebooks/eda.ipynb`:
```bash
python src/create_notebook.py
```

### 3. CLI Prediction Demo
Use the command line to predict if a message is spam or ham:
```bash
python src/predict.py "URGENT! You have won a 1-week free membership to our club. Call 0800123456 to claim."
```

### 4. Run Streamlit Web Application
Launch the graphical interface to interact with the classifier:
```bash
streamlit run app.py
```

---

## Model Evaluation Results

After running the training pipeline, the model performance comparison is summarized below:

| Model | Accuracy | Precision (Spam) | Recall (Spam) | F1-Score (Spam) | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Multinomial Naive Bayes** | 97.29% | 99.05% | 79.39% | 0.8814 | 0.9857 |
| **Logistic Regression** | 96.22% | 98.94% | 70.99% | 0.8267 | 0.9928 |
| **Support Vector Machine (Linear)** | **98.45%** | **99.15%** | **88.55%** | **0.9355** | **0.9950** |

### Win Criteria & Discussion
*   The final model is selected primarily using the **F1-Score** due to the imbalance between Ham (~86%) and Spam (~14%).
*   **Precision** is heavily prioritized (minimizing False Positives) to ensure legitimate, important messages are not flagged as spam.
