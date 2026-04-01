"""
=======================================================================
EMAIL SPAM CLASSIFIER — Complete Pipeline
=======================================================================
Phases covered:
  1. Setup & Imports
  2. Data Loading & EDA
  3. Preprocessing
  4A. TF-IDF / N-gram features
  4B. Sentence-BERT embeddings  (combined track)
  5. Model Training (NB, LR, SVM, XGBoost, MLP)
  6. Evaluation (F1, AUC-ROC, Confusion Matrix)
  7. Hyperparameter Tuning (GridSearchCV)
  8. Analysis (SHAP + t-SNE / UMAP)
  9. Final Report / comparison table
=======================================================================
"""

# ─────────────────────────────────────────────────────────────────────
# PHASE 1 — Imports & NLTK downloads
# ─────────────────────────────────────────────────────────────────────
import os, re, sys, warnings
warnings.filterwarnings("ignore")

# Ensure UTF-8 output on Windows
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")           # non-interactive backend (works without GUI)
import matplotlib.pyplot as plt
import seaborn as sns

import nltk
nltk.download("stopwords",  quiet=True)
nltk.download("wordnet",    quiet=True)
nltk.download("punkt",      quiet=True)
from nltk.corpus import stopwords
from nltk.stem  import WordNetLemmatizer

from sklearn.model_selection      import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model         import LogisticRegression
from sklearn.naive_bayes          import MultinomialNB
from sklearn.svm                  import LinearSVC
from sklearn.neural_network       import MLPClassifier
from sklearn.metrics              import (classification_report, confusion_matrix,
                                          roc_auc_score, f1_score, ConfusionMatrixDisplay)
from sklearn.pipeline             import Pipeline
from scipy.sparse                 import hstack, csr_matrix

import xgboost as xgb

try:
    from sentence_transformers import SentenceTransformer
    SBERT_AVAILABLE = True
except ImportError:
    SBERT_AVAILABLE = False
    print("[WARN] sentence-transformers not installed -- SBERT track skipped.")

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("[WARN] shap not installed -- SHAP analysis skipped.")

try:
    from sklearn.manifold import TSNE
    TSNE_AVAILABLE = True
except ImportError:
    TSNE_AVAILABLE = False

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("[OK] Phase 1 complete -- all imports loaded.")

# ─────────────────────────────────────────────────────────────────────
# PHASE 2 — Load Dataset & EDA
# ─────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("PHASE 2 — Loading dataset & EDA")
print("="*60)

import kagglehub
path = kagglehub.dataset_download("venky73/spam-mails-dataset")
csv_path = os.path.join(path, "spam_ham_dataset.csv")
df = pd.read_csv(csv_path)

# Drop unnamed index column if present
df.drop(columns=[c for c in df.columns if "Unnamed" in c], inplace=True)

print(f"\nDataset shape : {df.shape}")
print(f"Columns       : {list(df.columns)}")
print(f"\nClass distribution:\n{df['label'].value_counts()}")
print(f"\nClass balance  : {df['label'].value_counts(normalize=True).round(3).to_dict()}")

# — EDA plots ——————————————————————————————————————————————————————
# 1. Class distribution bar chart
fig, axes = plt.subplots(1, 3, figsize=(16, 4))

label_counts = df["label"].value_counts()
axes[0].bar(label_counts.index, label_counts.values,
            color=["#4CAF50", "#F44336"], edgecolor="white", linewidth=1.2)
axes[0].set_title("Class Distribution", fontsize=13, fontweight="bold")
axes[0].set_xlabel("Label"); axes[0].set_ylabel("Count")
for i, v in enumerate(label_counts.values):
    axes[0].text(i, v + 30, str(v), ha="center", fontweight="bold")

# 2. Email length distribution
df["text_len"] = df["text"].str.len()
for label, color in zip(["ham", "spam"], ["#4CAF50", "#F44336"]):
    sub = df[df["label"] == label]["text_len"]
    axes[1].hist(sub, bins=60, alpha=0.6, label=label, color=color, edgecolor="white")
axes[1].set_title("Email Length Distribution", fontsize=13, fontweight="bold")
axes[1].set_xlabel("Character Count"); axes[1].set_ylabel("Frequency")
axes[1].legend(); axes[1].set_xlim(0, 20000)

# 3. Top 15 words in spam vs ham (character-level proxy via TF-IDF)
from sklearn.feature_extraction.text import CountVectorizer
for idx, label in enumerate(["ham", "spam"]):
    subset = df[df["label"] == label]["text"].str.lower()
    cv = CountVectorizer(max_features=15, stop_words="english")
    cv.fit(subset)
    freq = cv.transform(subset).toarray().sum(axis=0)
    words = cv.get_feature_names_out()
    color = "#4CAF50" if label == "ham" else "#F44336"
    sorted_idx = np.argsort(freq)[::-1]
    # subplot 3 — show spam words only for brevity
    if label == "spam":
        axes[2].barh(words[sorted_idx], freq[sorted_idx], color=color)
        axes[2].set_title("Top 15 words in SPAM", fontsize=13, fontweight="bold")
        axes[2].set_xlabel("Frequency")

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/eda_plots.png", dpi=150)
plt.close()
print(f"\n[PLOT] EDA saved --> {OUTPUT_DIR}/eda_plots.png")

# ─────────────────────────────────────────────────────────────────────
# PHASE 3 — Preprocessing
# ─────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("PHASE 3 — Text Preprocessing")
print("="*60)

lemmatizer = WordNetLemmatizer()
STOP_WORDS  = set(stopwords.words("english"))

def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+",   " [URL] ",   text)   # URLs
    text = re.sub(r"\S+@\S+",           " [EMAIL] ", text)   # Email addresses
    text = re.sub(r"\d+",              " [NUM] ",   text)    # Numbers
    text = re.sub(r"[^a-z\s\[\]]",     " ",         text)   # Special chars
    text = re.sub(r"\s+",              " ",         text).strip()
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in STOP_WORDS or t.startswith("[")]
    return " ".join(tokens)

df["clean_text"] = df["text"].apply(clean_text)
print("Sample after cleaning:")
print(df[["text", "clean_text"]].head(2).to_string())

# Handcrafted features
df["num_urls"]    = df["text"].apply(lambda x: len(re.findall(r"http\S+|www\S+", str(x))))
df["num_caps"]    = df["text"].apply(lambda x: sum(1 for c in str(x) if c.isupper()))
df["caps_ratio"]  = df["num_caps"] / (df["text"].str.len() + 1)
df["num_excl"]    = df["text"].str.count("!")
df["num_words"]   = df["clean_text"].str.split().str.len()

print(f"\n[OK] Preprocessing complete.  Dataset: {df.shape}")

# ─────────────────────────────────────────────────────────────────────
# PHASE 4A — TF-IDF Features
# ─────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("PHASE 4A — TF-IDF Feature Engineering")
print("="*60)

X_text = df["clean_text"]
y      = df["label_num"]   # 0 = ham, 1 = spam

X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X_text, y, test_size=0.2, random_state=42, stratify=y
)

# Unigram + bigram TF-IDF (10 000 features)
tfidf = TfidfVectorizer(max_features=10_000, ngram_range=(1, 2), sublinear_tf=True)
X_train_tfidf = tfidf.fit_transform(X_train_raw)   # ← fit only on TRAIN
X_test_tfidf  = tfidf.transform(X_test_raw)

print(f"TF-IDF train: {X_train_tfidf.shape}  |  test: {X_test_tfidf.shape}")

# Handcrafted features aligned to train/test split
hc_cols = ["num_urls", "caps_ratio", "num_excl", "num_words"]
hc_train = csr_matrix(df.loc[X_train_raw.index, hc_cols].values.astype(float))
hc_test  = csr_matrix(df.loc[X_test_raw.index,  hc_cols].values.astype(float))

X_train_tfidf_hc = hstack([X_train_tfidf, hc_train])
X_test_tfidf_hc  = hstack([X_test_tfidf,  hc_test])

print(f"TF-IDF + Handcrafted train: {X_train_tfidf_hc.shape}")

# ─────────────────────────────────────────────────────────────────────
# PHASE 4B — Sentence-BERT Embeddings
# ─────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("PHASE 4B — Sentence-BERT Embeddings")
print("="*60)

if SBERT_AVAILABLE:
    sbert = SentenceTransformer("all-MiniLM-L6-v2")
    print("Encoding training set…")
    X_train_emb = sbert.encode(X_train_raw.tolist(), batch_size=64,
                               show_progress_bar=True, convert_to_numpy=True)
    print("Encoding test set…")
    X_test_emb  = sbert.encode(X_test_raw.tolist(),  batch_size=64,
                               show_progress_bar=True, convert_to_numpy=True)
    print(f"Embedding shape: {X_train_emb.shape}")

    # Combined: TF-IDF + SBERT + handcrafted
    X_train_combined = hstack([X_train_tfidf_hc, csr_matrix(X_train_emb)])
    X_test_combined  = hstack([X_test_tfidf_hc,  csr_matrix(X_test_emb)])
    print(f"Combined feature matrix train: {X_train_combined.shape}")
else:
    X_train_emb, X_test_emb = None, None
    X_train_combined = X_train_tfidf_hc
    X_test_combined  = X_test_tfidf_hc

# ─────────────────────────────────────────────────────────────────────
# PHASE 5 — Model Training
# ─────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("PHASE 5 — Model Training")
print("="*60)

models = {
    "Naive Bayes (TF-IDF)":             (MultinomialNB(),                          X_train_tfidf,    X_test_tfidf),
    "Logistic Regression (TF-IDF)":     (LogisticRegression(max_iter=1000, C=1.0), X_train_tfidf,    X_test_tfidf),
    "SVM (TF-IDF)":                     (LinearSVC(max_iter=2000),                 X_train_tfidf,    X_test_tfidf),
    "XGBoost (TF-IDF+HC)":             (xgb.XGBClassifier(n_estimators=200,
                                           max_depth=6,
                                           eval_metric="logloss", verbosity=0),    X_train_tfidf_hc, X_test_tfidf_hc),
    "MLP (TF-IDF)":                     (MLPClassifier(hidden_layer_sizes=(256, 128),
                                           max_iter=300, random_state=42),         X_train_tfidf,    X_test_tfidf),
}

if SBERT_AVAILABLE:
    models["Logistic Regression (Combined)"] = (
        LogisticRegression(max_iter=1000, C=1.0), X_train_combined, X_test_combined
    )

results = {}
for name, (model, X_tr, X_te) in models.items():
    print(f"\n  Training: {name} …")
    model.fit(X_tr, y_train)
    y_pred = model.predict(X_te)

    # AUC-ROC (handle models without predict_proba)
    try:
        y_prob = model.predict_proba(X_te)[:, 1]
    except AttributeError:
        y_prob = model.decision_function(X_te)

    f1  = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    results[name] = {"F1": round(f1, 4), "AUC": round(auc, 4),
                     "model": model, "preds": y_pred, "probs": y_prob}
    print(f"  → F1={f1:.4f}  AUC={auc:.4f}")

# ─────────────────────────────────────────────────────────────────────
# PHASE 6 — Evaluation
# ─────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("PHASE 6 — Evaluation")
print("="*60)

best_name  = max(results, key=lambda k: results[k]["F1"])
best_model = results[best_name]
best_preds = best_model["preds"]

print(f"\n[BEST] Best model: {best_name}  (F1={best_model['F1']}  AUC={best_model['AUC']})")
print(classification_report(y_test, best_preds, target_names=["Ham", "Spam"]))

# Confusion matrix for every model
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()
for ax, (name, res) in zip(axes, results.items()):
    cm = confusion_matrix(y_test, res["preds"])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Ham","Spam"], yticklabels=["Ham","Spam"],
                linewidths=0.5, linecolor="white",
                annot_kws={"size": 12, "weight": "bold"})
    ax.set_title(f"{name}\nF1={res['F1']}  AUC={res['AUC']}", fontsize=9, fontweight="bold")
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")

# Hide any extra subplots
for ax in axes[len(results):]:
    ax.set_visible(False)

plt.suptitle("Confusion Matrices — All Models", fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/confusion_matrices.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"\n[PLOT] Confusion matrices saved --> {OUTPUT_DIR}/confusion_matrices.png")

# ROC curves
plt.figure(figsize=(9, 6))
from sklearn.metrics import roc_curve
colors = plt.cm.tab10.colors
for i, (name, res) in enumerate(results.items()):
    fpr, tpr, _ = roc_curve(y_test, res["probs"])
    plt.plot(fpr, tpr, color=colors[i % 10],
             label=f"{name}  (AUC={res['AUC']:.3f})", linewidth=1.8)
plt.plot([0,1],[0,1],"k--", linewidth=1)
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
plt.title("ROC Curves — All Models", fontsize=13, fontweight="bold")
plt.legend(fontsize=7); plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/roc_curves.png", dpi=150)
plt.close()
print(f"[PLOT] ROC curves saved --> {OUTPUT_DIR}/roc_curves.png")

# ─────────────────────────────────────────────────────────────────────
# PHASE 7 — Hyperparameter Tuning (Logistic Regression on TF-IDF)
# ─────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("PHASE 7 — Hyperparameter Tuning (GridSearchCV)")
print("="*60)

pipe = Pipeline([
    ("tfidf", TfidfVectorizer(sublinear_tf=True)),
    ("clf",   LogisticRegression(max_iter=1000))
])

param_grid = {
    "tfidf__max_features": [5000, 10000],
    "tfidf__ngram_range":  [(1,1), (1,2)],
    "clf__C":              [0.1, 1.0, 10.0],
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid = GridSearchCV(pipe, param_grid, cv=cv, scoring="f1", n_jobs=-1, verbose=1)
grid.fit(X_train_raw, y_train)

best_params = grid.best_params_
best_cv_f1  = round(grid.best_score_, 4)
tuned_preds = grid.predict(X_test_raw)
tuned_f1    = round(f1_score(y_test, tuned_preds), 4)

print(f"\n  Best params : {best_params}")
print(f"  CV F1       : {best_cv_f1}")
print(f"  Test F1     : {tuned_f1}")

# Update results table with tuned model
results["LR Tuned (GridSearchCV)"] = {
    "F1": tuned_f1, "AUC": round(roc_auc_score(y_test, grid.predict_proba(X_test_raw)[:,1]),4),
    "model": grid, "preds": tuned_preds,
    "probs": grid.predict_proba(X_test_raw)[:,1]
}

# ─────────────────────────────────────────────────────────────────────
# PHASE 8 — Analysis: SHAP + t-SNE
# ─────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("PHASE 8 — Analysis (SHAP Feature Importance + t-SNE)")
print("="*60)

# — 8A  SHAP for Logistic Regression ——————————————————————
if SHAP_AVAILABLE:
    print("\n  Computing SHAP values (Linear explainer on LR)…")
    lr_model = results["Logistic Regression (TF-IDF)"]["model"]
    # Use a small background sample for speed
    bg_size   = min(200, X_train_tfidf.shape[0])
    bg_sample = X_train_tfidf[:bg_size]
    explainer = shap.LinearExplainer(lr_model, bg_sample, feature_perturbation="interventional")
    test_sample = X_test_tfidf[:200]
    shap_vals = explainer.shap_values(test_sample)

    feature_names = tfidf.get_feature_names_out()
    mean_shap = np.abs(shap_vals).mean(axis=0)
    top20_idx  = np.argsort(mean_shap)[::-1][:20]

    plt.figure(figsize=(10, 6))
    plt.barh(feature_names[top20_idx][::-1], mean_shap[top20_idx][::-1],
             color="#5C6BC0", edgecolor="white")
    plt.xlabel("|SHAP value|  (mean over test sample)")
    plt.title("Top 20 Features — SHAP (Logistic Regression)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/shap_importance.png", dpi=150)
    plt.close()
    print(f"  [PLOT] SHAP plot saved --> {OUTPUT_DIR}/shap_importance.png")
else:
    print("  [WARN] SHAP not available -- skipping SHAP analysis.")

# — 8B  t-SNE on SBERT embeddings ————————————————————————
if SBERT_AVAILABLE and X_train_emb is not None and TSNE_AVAILABLE:
    print("\n  Running t-SNE on SBERT embeddings (this may take ~60 s)…")
    # Use a 1000-point subsample from train for speed
    np.random.seed(42)
    idx = np.random.choice(len(X_train_emb), size=min(1000, len(X_train_emb)), replace=False)
    emb_sub   = X_train_emb[idx]
    label_sub = y_train.values[idx]

    tsne = TSNE(n_components=2, perplexity=40, max_iter=1000, random_state=42)
    emb_2d = tsne.fit_transform(emb_sub)

    plt.figure(figsize=(9, 7))
    for cls, color, name_cls in [(0,"#43A047","Ham"), (1,"#E53935","Spam")]:
        mask = label_sub == cls
        plt.scatter(emb_2d[mask,0], emb_2d[mask,1],
                    c=color, label=name_cls, alpha=0.55, s=18, edgecolors="none")
    plt.title("t-SNE of SBERT Embeddings (train sample)", fontsize=13, fontweight="bold")
    plt.legend(fontsize=11); plt.axis("off"); plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/tsne_embeddings.png", dpi=150)
    plt.close()
    print(f"  [PLOT] t-SNE plot saved --> {OUTPUT_DIR}/tsne_embeddings.png")
else:
    print("  [WARN] SBERT embeddings not available -- skipping t-SNE.")

# ─────────────────────────────────────────────────────────────────────
# PHASE 9 — Final Report / Comparison Table
# ─────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("PHASE 9 — Final Model Comparison Table")
print("="*60)

summary_df = pd.DataFrame({
    name: {"F1-Score": v["F1"], "AUC-ROC": v["AUC"]}
    for name, v in results.items()
}).T.sort_values("F1-Score", ascending=False)

print("\n" + summary_df.to_string())

# Bar chart comparison
fig, ax = plt.subplots(figsize=(11, 5))
x = np.arange(len(summary_df))
w = 0.35
bars1 = ax.bar(x - w/2, summary_df["F1-Score"],  width=w, label="F1-Score",  color="#42A5F5", edgecolor="white")
bars2 = ax.bar(x + w/2, summary_df["AUC-ROC"],   width=w, label="AUC-ROC",   color="#EF5350", edgecolor="white")
ax.set_xticks(x); ax.set_xticklabels(summary_df.index, rotation=25, ha="right", fontsize=8)
ax.set_ylabel("Score"); ax.set_ylim(0.7, 1.02)
ax.set_title("Model Comparison: F1-Score & AUC-ROC", fontsize=13, fontweight="bold")
ax.legend()

for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
            f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=7, fontweight="bold")
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
            f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=7, fontweight="bold")

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/model_comparison.png", dpi=150)
plt.close()
print(f"\n[PLOT] Comparison chart saved --> {OUTPUT_DIR}/model_comparison.png")

# ── Save summary CSV ──────────────────────────────────────────────
summary_df.to_csv(f"{OUTPUT_DIR}/model_summary.csv")
print(f"[FILE] Summary CSV saved  --> {OUTPUT_DIR}/model_summary.csv")

print("\n" + "="*60)
print("*** PIPELINE COMPLETE! ***")
print("="*60)
print(f"\n[BEST] Best Model : {summary_df.index[0]}")
print(f"   F1-Score   : {summary_df['F1-Score'].iloc[0]}")
print(f"   AUC-ROC    : {summary_df['AUC-ROC'].iloc[0]}")
print(f"\nAll outputs saved in: ./{OUTPUT_DIR}/")
