"""
train_bert.py
=============
BERT-based Spam Classifier using sentence-transformers.

Pipeline:
    Raw text -> SentenceTransformer('all-MiniLM-L6-v2') -> 384-dim embeddings
             -> Logistic Regression / LinearSVM / MLP -> SPAM or HAM

Usage:
    python train_bert.py
"""

import os, io, zipfile, urllib.request, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report, RocCurveDisplay,
)

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, "data")
MODEL_DIR  = os.path.join(BASE_DIR, "models")
RESULT_DIR = os.path.join(BASE_DIR, "results")
for d in [DATA_DIR, MODEL_DIR, RESULT_DIR]:
    os.makedirs(d, exist_ok=True)

# ── Dataset ───────────────────────────────────────────────────────────────────
DATASET_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "00228/smsspamcollection.zip"
)
CSV_PATH = os.path.join(DATA_DIR, "spam.csv")

def download_dataset():
    print("[INFO] Downloading SMS Spam Collection ...")
    with urllib.request.urlopen(DATASET_URL) as resp:
        raw = resp.read()
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        with z.open("SMSSpamCollection") as f:
            content = f.read().decode("utf-8")
    rows = [line.split("\t", 1) for line in content.strip().splitlines()]
    df = pd.DataFrame(rows, columns=["label", "text"])
    df.to_csv(CSV_PATH, index=False)
    print(f"   Saved -> {CSV_PATH}  ({len(df)} rows)")
    return df

def load_data():
    if os.path.exists(CSV_PATH):
        df = pd.read_csv(CSV_PATH)
        if "label" not in df.columns:
            df = download_dataset()
    else:
        df = download_dataset()
    df = df[["label", "text"]].dropna()
    df["label"] = df["label"].str.strip()
    return df

# ── Minimal pre-processing (BERT handles tokenisation internally) ──────────────
import re

def preprocess(text: str) -> str:
    """Light clean only -- preserve natural language for BERT."""
    text = re.sub(r"http\S+|www\S+", "URL", text)
    text = text.strip()
    return text

# ── BERT Embedding ─────────────────────────────────────────────────────────────
def get_bert_embeddings(texts, model_name="all-MiniLM-L6-v2", batch_size=64):
    from sentence_transformers import SentenceTransformer
    print(f"\n[BERT] Loading model: {model_name} ...")
    encoder = SentenceTransformer(model_name)
    print(f"   Encoding {len(texts)} messages (batch_size={batch_size}) ...")
    embeddings = encoder.encode(
        list(texts),
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    print(f"   Embedding shape: {embeddings.shape}")
    return embeddings, encoder

# ── Models ────────────────────────────────────────────────────────────────────
def build_models():
    return {
        "Logistic Regression": LogisticRegression(
            C=5, max_iter=1000, random_state=42
        ),
        "Linear SVM": CalibratedClassifierCV(
            LinearSVC(C=1, max_iter=3000, random_state=42)
        ),
        "MLP Neural Network": MLPClassifier(
            hidden_layer_sizes=(256, 128),
            activation="relu",
            max_iter=300,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
        ),
    }

# ── Evaluation ─────────────────────────────────────────────────────────────────
def evaluate(name, clf, X_tr, X_te, y_tr, y_te):
    clf.fit(X_tr, y_tr)
    y_pred  = clf.predict(X_te)
    y_proba = clf.predict_proba(X_te)[:, 1] if hasattr(clf, "predict_proba") else None

    acc  = accuracy_score(y_te, y_pred)
    prec = precision_score(y_te, y_pred, pos_label=1)
    rec  = recall_score(y_te, y_pred, pos_label=1)
    f1   = f1_score(y_te, y_pred, pos_label=1)
    auc  = roc_auc_score(y_te, y_proba) if y_proba is not None else float("nan")

    print(f"  [OK] {name:<26}  Acc={acc:.4f}  Prec={prec:.4f}  Rec={rec:.4f}  F1={f1:.4f}  AUC={auc:.4f}")
    return dict(name=name, clf=clf, y_pred=y_pred, y_proba=y_proba,
                accuracy=acc, precision=prec, recall=rec, f1=f1, roc_auc=auc)

# ── Plots ──────────────────────────────────────────────────────────────────────
def plot_confusion_matrices(results, y_te, label_names):
    n    = len(results)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]
    for ax, res in zip(axes, results):
        cm = confusion_matrix(y_te, res["y_pred"])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=label_names, yticklabels=label_names,
                    ax=ax, annot_kws={"size": 14})
        ax.set_title(res["name"], fontsize=11, fontweight="bold")
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    plt.suptitle("Confusion Matrices -- BERT Classifiers", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(RESULT_DIR, "bert_confusion_matrices.png")
    plt.savefig(path, dpi=120, bbox_inches="tight"); plt.close()
    print(f"   [PLOT] Saved -> {path}")

def plot_roc(results, y_te):
    fig, ax = plt.subplots(figsize=(9, 7))
    colors  = plt.cm.Set1.colors
    for i, res in enumerate(results):
        if res["y_proba"] is not None:
            RocCurveDisplay.from_predictions(
                y_te, res["y_proba"],
                name=f"{res['name']} (AUC={res['roc_auc']:.4f})",
                ax=ax, color=colors[i],
            )
    ax.plot([0, 1], [0, 1], "k--", lw=1.2)
    ax.set_title("ROC Curves -- BERT Classifiers", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    plt.tight_layout()
    path = os.path.join(RESULT_DIR, "bert_roc_curves.png")
    plt.savefig(path, dpi=120); plt.close()
    print(f"   [PLOT] Saved -> {path}")

def plot_comparison(results_list):
    """Bar chart comparing all models across metrics."""
    df_plot = pd.DataFrame([{
        "Model": r["name"],
        "Accuracy": r["accuracy"],
        "Precision": r["precision"],
        "Recall": r["recall"],
        "F1": r["f1"],
        "ROC-AUC": r["roc_auc"],
    } for r in results_list])
    metrics = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
    ax = df_plot.set_index("Model")[metrics].plot(
        kind="bar", figsize=(11, 5), colormap="Set2", width=0.7, edgecolor="white"
    )
    ax.set_ylim(0.85, 1.005)
    ax.set_xticklabels(df_plot["Model"], rotation=15, ha="right", fontsize=11)
    ax.set_title("BERT Classifier Comparison", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    plt.tight_layout()
    path = os.path.join(RESULT_DIR, "bert_model_comparison.png")
    plt.savefig(path, dpi=120); plt.close()
    print(f"   [PLOT] Saved -> {path}")

def plot_tsne(X_emb, y, label_names):
    try:
        from sklearn.manifold import TSNE
        print("   Running t-SNE (may take ~30 s) ...")
        tsne = TSNE(n_components=2, perplexity=40, random_state=42, max_iter=1000)
        X2   = tsne.fit_transform(X_emb[:2000])
        y2   = y[:2000]
        colors = ["#4ECDC4", "#FF6B6B"]
        fig, ax = plt.subplots(figsize=(9, 7))
        for idx, (label, color) in enumerate(zip(label_names, colors)):
            mask = y2 == idx
            ax.scatter(X2[mask, 0], X2[mask, 1], c=color,
                       label=label.capitalize(), alpha=0.55, s=18, edgecolors="none")
        ax.set_title("t-SNE of BERT Embeddings (spam vs. ham)", fontsize=13, fontweight="bold")
        ax.legend(fontsize=12)
        ax.set_xlabel("t-SNE dim 1"); ax.set_ylabel("t-SNE dim 2")
        plt.tight_layout()
        path = os.path.join(RESULT_DIR, "bert_tsne.png")
        plt.savefig(path, dpi=120); plt.close()
        print(f"   [PLOT] Saved -> {path}")
    except Exception as e:
        print(f"   [WARN] t-SNE skipped: {e}")

# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print("\n" + "=" * 65)
    print("  BERT Spam Classifier - Training & Evaluation")
    print("=" * 65)

    # 1. Load data
    print("\n[DATA] Loading dataset ...")
    df = load_data()
    spam_n = (df["label"] == "spam").sum()
    ham_n  = (df["label"] == "ham").sum()
    print(f"   Total: {len(df)}  |  Spam: {spam_n}  |  Ham: {ham_n}")

    # 2. Encode labels
    le = LabelEncoder()
    y  = le.fit_transform(df["label"])   # ham=0, spam=1
    label_names = le.classes_.tolist()

    # 3. Light pre-process
    df["text_clean"] = df["text"].apply(preprocess)

    # 4. BERT embeddings for full dataset
    X_emb, encoder = get_bert_embeddings(df["text_clean"])

    # 5. t-SNE visualisation
    print("\n[PLOT] Generating t-SNE visualisation ...")
    plot_tsne(X_emb, y, label_names)

    # 6. Train/test split
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_emb, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\n[SPLIT] Train: {len(X_tr)}  |  Test: {len(X_te)}")

    # 7. Train classifiers
    print("\n[TRAIN] Training classifiers on BERT embeddings ...")
    results = []
    for name, clf in build_models().items():
        r = evaluate(name, clf, X_tr, X_te, y_tr, y_te)
        results.append(r)

    # 8. Detailed report for best model
    best = max(results, key=lambda r: r["f1"])
    print(f"\n[REPORT] Detailed report - {best['name']}")
    print(classification_report(y_te, best["y_pred"], target_names=label_names))

    # 9. Summary table
    df_res = pd.DataFrame([{
        "Model": r["name"], "Accuracy": r["accuracy"],
        "Precision": r["precision"], "Recall": r["recall"],
        "F1": r["f1"], "ROC-AUC": r["roc_auc"]}
        for r in results])
    print("=" * 65)
    print("  Results Summary")
    print("=" * 65)
    print(df_res.to_string(index=False, float_format="{:.4f}".format))

    # 10. Save best model (before plots, so a plot error doesn't block saving)
    print(f"\n[BEST] Best model: {best['name']}  (F1={best['f1']:.4f}  AUC={best['roc_auc']:.4f})")
    joblib.dump(best["clf"],  os.path.join(MODEL_DIR, "bert_best_model.pkl"))
    joblib.dump(encoder,      os.path.join(MODEL_DIR, "bert_encoder.pkl"))
    joblib.dump(le,           os.path.join(MODEL_DIR, "label_encoder.pkl"))
    print("   Models saved to models/")

    # 11. Plots
    print("\n[PLOT] Generating plots ...")
    plot_confusion_matrices(results, y_te, label_names)
    plot_roc(results, y_te)
    plot_comparison(results)

    print("\n[DONE] All complete! Check results/ for all plots.\n")

if __name__ == "__main__":
    main()
