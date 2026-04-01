"""
predict.py
==========
CLI tool to classify an email/SMS as SPAM or HAM using BERT embeddings.

Usage:
    python predict.py "Free entry! Win cash now!"
    python predict.py           (interactive REPL mode)

Requires training first:
    python train_bert.py
"""

import os, sys, re, joblib

# Ensure UTF-8 output on Windows consoles
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

# ── Light pre-processing (mirrors train_bert.py) ──────────────────────────────
def preprocess(text: str) -> str:
    text = re.sub(r"http\S+|www\S+", "URL", text)
    return text.strip()

# ── Load saved artifacts ───────────────────────────────────────────────────────
def load_artifacts():
    required = ["bert_best_model.pkl", "bert_encoder.pkl", "label_encoder.pkl"]
    missing  = [f for f in required if not os.path.exists(os.path.join(MODEL_DIR, f))]
    if missing:
        print("❌  Model files not found. Please run train_bert.py first.")
        print("    Missing:", ", ".join(missing))
        sys.exit(1)

    clf     = joblib.load(os.path.join(MODEL_DIR, "bert_best_model.pkl"))
    encoder = joblib.load(os.path.join(MODEL_DIR, "bert_encoder.pkl"))
    le      = joblib.load(os.path.join(MODEL_DIR, "label_encoder.pkl"))
    return clf, encoder, le

# ── Prediction ─────────────────────────────────────────────────────────────────
def predict(text: str, clf, encoder, le) -> dict:
    clean   = preprocess(text)
    X       = encoder.encode([clean], convert_to_numpy=True)
    label   = le.inverse_transform(clf.predict(X))[0]
    proba   = clf.predict_proba(X)[0] if hasattr(clf, "predict_proba") else None
    spam_idx = list(le.classes_).index("spam")
    confidence = proba[spam_idx] if proba is not None else None
    return {"label": label.upper(), "confidence": confidence}

# ── Display ────────────────────────────────────────────────────────────────────
def display_result(text: str, result: dict):
    label = result["label"]
    conf  = result["confidence"]
    BAR   = "#"

    if label == "SPAM":
        icon, color = "[!!]", "\033[91m"
    else:
        icon, color = "[OK]", "\033[92m"
    reset = "\033[0m"

    print("\n" + "-" * 58)
    print(f"  Input  : {text[:82]}{'...' if len(text) > 82 else ''}")
    print(f"  Result : {color}{icon}  {label}{reset}")
    if conf is not None:
        spam_fill = int(conf * 40)
        ham_fill  = 40 - spam_fill
        bar = f"\033[91m{BAR * spam_fill}\033[92m{BAR * ham_fill}{reset}"
        print(f"  Spam   : {color}{conf*100:5.1f}%{reset}  [{bar}]  Ham: {color}{(1-conf)*100:5.1f}%{reset}")
    print("-" * 58 + "\n")

# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    clf, encoder, le = load_artifacts()

    if len(sys.argv) > 1:
        text   = " ".join(sys.argv[1:])
        result = predict(text, clf, encoder, le)
        display_result(text, result)
    else:
        print("\nBERT Spam Classifier - Interactive Mode")
        print("    Type a message and press Enter. Type 'quit' to exit.\n")
        while True:
            try:
                text = input("  Enter message: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nBye!"); break
            if not text:
                continue
            if text.lower() in ("quit", "exit", "q"):
                print("Bye!"); break
            result = predict(text, clf, encoder, le)
            display_result(text, result)

if __name__ == "__main__":
    main()
