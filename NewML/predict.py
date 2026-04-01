import os, sys, re, joblib, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sentence_transformers import SentenceTransformer

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Suppress prints from third-party tools
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("punkt", quiet=True)

MODEL_DIR = "models"
if not os.path.exists(os.path.join(MODEL_DIR, "spam_classifier_model.joblib")):
    print(f"Error: Model not found. Run export_model.py first to export models into {MODEL_DIR}/.")
    sys.exit(1)

print("Loading models (TF-IDF, SBERT, Logistic Regression)...")
tfidf = joblib.load(os.path.join(MODEL_DIR, "tfidf_vectorizer.joblib"))
clf = joblib.load(os.path.join(MODEL_DIR, "spam_classifier_model.joblib"))
sbert = SentenceTransformer("all-MiniLM-L6-v2")

lemmatizer = WordNetLemmatizer()
STOP_WORDS = set(stopwords.words("english"))

def extract_features(texts: list) -> csr_matrix:
    data = []
    
    for text in texts:
        orig_text = str(text)
        
        num_urls = len(re.findall(r"http\S+|www\S+", orig_text))
        num_caps = sum(1 for c in orig_text if c.isupper())
        caps_ratio = num_caps / (len(orig_text) + 1)
        num_excl = orig_text.count("!")
        
        t = orig_text.lower()
        t = re.sub(r"http\S+|www\S+", " [URL] ", t)
        t = re.sub(r"\S+@\S+", " [EMAIL] ", t)
        t = re.sub(r"\d+", " [NUM] ", t)
        t = re.sub(r"[^a-z\s\[\]]", " ", t)
        t = re.sub(r"\s+", " ", t).strip()
        
        clean_words = [lemmatizer.lemmatize(w) for w in t.split() if w not in STOP_WORDS or w.startswith("[")]
        clean_text = " ".join(clean_words)
        num_words = len(clean_words)
        
        data.append({
            "clean_text": clean_text,
            "num_urls": float(num_urls),
            "caps_ratio": float(caps_ratio),
            "num_excl": float(num_excl),
            "num_words": float(num_words)
        })
        
    df = pd.DataFrame(data)
    
    X_tfidf = tfidf.transform(df["clean_text"])
    hc_feats = csr_matrix(df[["num_urls", "caps_ratio", "num_excl", "num_words"]].values)
    X_emb = sbert.encode(df["clean_text"].tolist(), batch_size=64, show_progress_bar=False, convert_to_numpy=True)
    
    X_combined = hstack([X_tfidf, hc_feats, csr_matrix(X_emb)])
    return X_combined

def predict_spam(texts: list):
    X = extract_features(texts)
    preds = clf.predict(X)
    probs = clf.predict_proba(X)[:, 1]
    
    results = []
    for text, pred, prob in zip(texts, preds, probs):
        label = "SPAM" if pred == 1 else "HAM"
        confidence = prob if pred == 1 else 1 - prob
        results.append({"text": text, "label": label, "confidence": confidence})
        
    return results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Predict if a text is SPAM or HAM.")
    parser.add_argument("texts", nargs="*", default=[], help="Text strings to classify.")
    args = parser.parse_args()

    sample_texts = [
        "Congratulations! You've won a $1,000 Walmart gift card. Click here to claim your prize.",
        "Hi John, are we still meeting tomorrow at 10 AM for the project sync? Let me know.",
        "URGENT: Your bank account has been locked. Verify your identity at http://secure-update-id.com",
        "Attached is the invoice for the services provided last month. Please process payment."
    ]
    
    texts_to_predict = args.texts if args.texts else sample_texts
    
    print(f"\nAnalyzing {len(texts_to_predict)} message(s)...\n")
    results = predict_spam(texts_to_predict)
    
    for i, res in enumerate(results):
        print(f"[{res['label']}] (Conf: {res['confidence']:.2%})")
        print(f"Text: \"{res['text']}\"\n")
