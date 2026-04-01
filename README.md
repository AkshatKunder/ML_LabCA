# 📧 BERT Email Spam Classifier

Classifies emails / SMS as **Spam** or **Ham** using BERT sentence embeddings (`all-MiniLM-L6-v2`) + scikit-learn classifiers.

---

## Pipeline

```
Raw Text
  ↓  Minimal cleaning (URL normalisation)
  ↓  SentenceTransformer('all-MiniLM-L6-v2')
  ↓  384-dimensional semantic embedding
  ↓  Logistic Regression / SVM / MLP
SPAM or HAM
```

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train (downloads BERT model ~80 MB on first run)
python train_bert.py

# 3. Predict
python predict.py "Congratulations! You've won a FREE prize!"
python predict.py "Can we reschedule tomorrow's meeting?"

# 4. Full notebook walkthrough
jupyter notebook spam_classifier.ipynb
```

---

## Project Structure

```
ML-LabCA/
├── data/spam.csv              # Auto-downloaded dataset
├── models/
│   ├── bert_best_model.pkl    # Best trained classifier
│   ├── bert_encoder.pkl       # SentenceTransformer encoder
│   └── label_encoder.pkl      # ham/spam label encoder
├── results/                   # All plots (PNG)
├── spam_classifier.ipynb      # Notebook (EDA → BERT → evaluation)
├── train_bert.py              # Training script
├── predict.py                 # CLI prediction tool
├── requirements.txt
└── README.md
```

---

## Dataset

[UCI SMS Spam Collection](https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection) — 5 572 labelled messages (auto-downloaded).

---

## Model

| Property | Detail |
|---|---|
| **BERT model** | `all-MiniLM-L6-v2` (22M params) |
| **Embedding dim** | 384 |
| **Classifiers** | Logistic Regression · Linear SVM · MLP |
| **Accuracy** | ~98–99% |
| **F1 (spam)** | ~96–98% |
| **ROC-AUC** | ~99% |
