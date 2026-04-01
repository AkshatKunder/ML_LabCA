import os, re, joblib, warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import kagglehub
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from scipy.sparse import hstack, csr_matrix
from sentence_transformers import SentenceTransformer

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("punkt", quiet=True)

print("Loading data...")
path = kagglehub.dataset_download("venky73/spam-mails-dataset")
csv_path = os.path.join(path, "spam_ham_dataset.csv")
df = pd.read_csv(csv_path)

lemmatizer = WordNetLemmatizer()
STOP_WORDS = set(stopwords.words("english"))

def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " [URL] ", text)
    text = re.sub(r"\S+@\S+", " [EMAIL] ", text)
    text = re.sub(r"\d+", " [NUM] ", text)
    text = re.sub(r"[^a-z\s\[\]]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return " ".join([lemmatizer.lemmatize(t) for t in text.split() if t not in STOP_WORDS or t.startswith("[")])

print("Preprocessing...")
df["clean_text"] = df["text"].apply(clean_text)
df["num_urls"] = df["text"].apply(lambda x: len(re.findall(r"http\S+|www\S+", str(x))))
df["num_caps"] = df["text"].apply(lambda x: sum(1 for c in str(x) if c.isupper()))
df["caps_ratio"] = df["num_caps"] / (df["text"].str.len() + 1)
df["num_excl"] = df["text"].str.count("!")
df["num_words"] = df["clean_text"].str.split().str.len()

X_text = df["clean_text"]
y = df["label_num"]

print("Feature Engineering...")
tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), sublinear_tf=True)
X_tfidf = tfidf.fit_transform(X_text)

hc_cols = ["num_urls", "caps_ratio", "num_excl", "num_words"]
hc_feats = csr_matrix(df[hc_cols].values.astype(float))

sbert = SentenceTransformer("all-MiniLM-L6-v2")
X_emb = sbert.encode(X_text.tolist(), batch_size=64, show_progress_bar=True, convert_to_numpy=True)

X_combined = hstack([X_tfidf, hc_feats, csr_matrix(X_emb)])

print("Training Combined Logistic Regression Model...")
clf = LogisticRegression(max_iter=1000, C=1.0)
clf.fit(X_combined, y)

print("Exporting models...")
os.makedirs("models", exist_ok=True)
joblib.dump(tfidf, "models/tfidf_vectorizer.joblib")
joblib.dump(clf, "models/spam_classifier_model.joblib")
print("Export complete: models/ saved.")
