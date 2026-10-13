"""Classical baselines: TF-IDF (word 1-2 grams) + Logistic Regression / Naive Bayes."""
import time
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline


def fit_tfidf(kind, tr_texts, tr_y, dev_texts, dev_y):
    t0 = time.time(); best, best_pipe, best_c = -1, None, None
    grid = [0.3, 1, 3, 10, 30] if kind == "tfidf_lr" else [0.1, 0.3, 1.0]   # C  /  alpha
    for c in grid:
        clf = LogisticRegression(C=c, max_iter=2000) if kind == "tfidf_lr" else MultinomialNB(alpha=c)
        pipe = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True), clf)
        pipe.fit(tr_texts, tr_y)
        acc = float((pipe.predict(dev_texts) == dev_y).mean())
        if acc > best:
            best, best_pipe, best_c = acc, pipe, c
    return best_pipe, dict(dev_acc=best, hp=best_c, train_time_s=time.time() - t0)


def proba(pipe, texts):
    return pipe.predict_proba(texts).astype(np.float32)
