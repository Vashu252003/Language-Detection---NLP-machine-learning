import pandas as pd
import re
import string
import joblib
import os
import sys  # Re-imported sys to read command-line arguments
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier

# --- Config ---
DATA_PATH = "data/Language Detection.csv"
MODEL_PATH = "saved_models/best_langid_model.joblib"

# --- Preprocessing ---
def get_vectorizer():
    return TfidfVectorizer(analyzer="char", ngram_range=(1,4), max_features=20000)

# --- Candidate Models ---
def get_models():
    return {
        "MultinomialNB": MultinomialNB(),
        "LogisticRegression": LogisticRegression(max_iter=2000, solver="liblinear"),
        "LinearSVC": LinearSVC(max_iter=2000, dual=True),
        "RandomForest": RandomForestClassifier(n_estimators=200, n_jobs=-1),
        "SGD_Log": SGDClassifier(loss="log_loss", max_iter=2000)
    }

# --- Training Function ---
def train():
    print(f"Loading data from {DATA_PATH}...")
    try:
        df = pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        print(f"Error: Data file not found at '{DATA_PATH}'")
        return

    X = df["Text"].astype(str)
    y = df["Language"].astype(str)
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    vectorizer = get_vectorizer()
    models = get_models()
    best_model = None
    best_score = 0.0

    print("\n=== Cross-validation results ===")
    for name, clf in models.items():
        pipe = Pipeline([("tfidf", vectorizer), ("clf", clf)])
        scores = cross_val_score(pipe, X_train, y_train, cv=3, scoring="accuracy", n_jobs=-1)
        mean_score = scores.mean()
        print(f"{name:15s} CV accuracy = {mean_score:.4f}")
        if mean_score > best_score:
            best_score = mean_score
            best_model = pipe

    print(f"\nTraining the best model on the full dataset: {best_model.named_steps['clf'].__class__.__name__}")
    best_model.fit(X, y) # Train on all data for final model

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)
    print(f"\n✅ Training complete. Model saved to {MODEL_PATH}")


# --- Text preprocessing ---
def normalize_text(text):
    """
    Normalize input text for language detection:
    - Lowercase
    - Remove emojis and rare symbols
    - Keep letters, numbers, and common punctuation
    """
    # Convert to lowercase
    text = text.lower()

    # Keep letters, numbers, common punctuation, and whitespace
    allowed_chars = string.ascii_lowercase + string.digits + string.punctuation + " "
    text = "".join([c if c in allowed_chars else " " for c in text])

    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text).strip()
    
    return text

# --- Interactive prediction ---
def predict_interactive():
    print("Loading model for prediction...")
    try:
        model = joblib.load(MODEL_PATH)
    except FileNotFoundError:
        print(f"Error: Model file not found at '{MODEL_PATH}'")
        print("Please run 'python app.py train' first.")
        return

    print("✅ Model loaded. You can now start predicting.")
    print("-" * 50)

    while True:
        text = input("Enter text to predict (or type 'quit' to exit): ")
        if text.lower() in ['quit', 'exit']:
            break
        if not text.strip():
            continue

        # Normalize text
        clean_text = normalize_text(text)

        prediction = model.predict([clean_text])
        print(f"'{text}' → Predicted Language: {prediction[0]}\n")

    print("Exiting application. Goodbye! 👋")

# --- Main Controller ---
if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1].lower() == 'train':
        train()
    elif len(sys.argv) == 2 and sys.argv[1].lower() == 'predict':
        predict_interactive()
    else:
        print("Usage:")
        print("  To train the model: python app.py train")
        print("  To start predicting: python app.py predict")