"""Non-transformer model families, evaluated under exactly the same protocol as the transformers.

Families (all cheap on CPU; every one uses the same stratified training subsets, dev-set model selection,
perturbed test sets and out-of-domain reviews as the neural models):

  vader_lexicon    rule/lexicon baseline, NO task labels (VADER compound score)            [zero-label anchor]
  tfidf_svm        word 1-2-gram TF-IDF + linear SVM                                       [sparse, word level]
  tfidf_char_lr    character 2-5-gram (word-boundary) TF-IDF + logistic regression         [sparse, sub-word level]
  w2v_avg_lr       skip-gram word2vec trained on OUR 1.2M unlabeled words, averaged + LR  [static, pre-trained on same data]
  fasttext_avg_lr  fastText (sub-word) skip-gram on the same words, averaged + LR          [static, pre-trained on same data]

The static-embedding families use the *same unlabeled corpus* as our MLM pre-training, which makes them a controlled
comparison of ways to learn from the same text (static vs contextual).
Word-level TF-IDF + LogReg / NaiveBayes live in baselines.py (stage 'curve').
"""
import time
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from config import MODELS
from data import load_movie_reviews, subsample

FAMILY_MODELS = ["vader_lexicon", "tfidf_svm", "tfidf_char_lr", "w2v_avg_lr", "fasttext_avg_lr"]
PSEUDO_PROB = {"vader_lexicon", "tfidf_svm"}          # scores squashed to (0,1): fine for ranking/accuracy, not for calibration


def unlabeled_sentences(chunk=40):
    """Pre-training documents cut into 40-token pseudo-sentences (the corpus used for MLM)."""
    docs, _ = load_movie_reviews()
    out = []
    for d in docs:
        w = d.split()
        out += [w[i:i + chunk] for i in range(0, len(w), chunk)]
    return out


def get_embeddings(domain="sst2", own_texts=None, seed=0):
    """Train (or load from cache) skip-gram word2vec and fastText on THIS domain's own unlabeled text.
    sst2 (default): the original 1.2M-word movie-review pre-training corpus (unchanged from earlier runs).
    finance/airline: `own_texts` (that domain's own train+dev sentences, labels ignored) -- each row is
    already sentence-length, so no 40-word chunking is needed."""
    from gensim.models import FastText, Word2Vec
    suffix = "" if domain == "sst2" else f"_{domain}"
    p_w, p_f = MODELS / f"w2v{suffix}.model", MODELS / f"fasttext{suffix}.model"
    if p_w.exists() and p_f.exists():
        return Word2Vec.load(str(p_w)).wv, FastText.load(str(p_f)).wv
    sents = unlabeled_sentences() if domain == "sst2" else [t.split() for t in own_texts]
    t0 = time.time()
    w2v = Word2Vec(sents, vector_size=100, window=5, min_count=2, sg=1, epochs=5, workers=1, seed=seed)
    ft = FastText(sents, vector_size=100, window=5, min_count=2, sg=1, epochs=5, workers=1, seed=seed, min_n=3, max_n=5, bucket=50000)
    print(f"[families] {domain}: trained word2vec + fastText on {sum(map(len, sents))/1e6:.2f}M tokens in {time.time()-t0:.0f}s", flush=True)
    w2v.save(str(p_w)); ft.save(str(p_f))
    return w2v.wv, ft.wv


def _avg(wv, subword):
    def f(texts):
        out = np.zeros((len(texts), wv.vector_size), dtype=np.float32)
        for i, t in enumerate(texts):
            ws = [w for w in t.split() if (subword or w in wv.key_to_index)]
            if ws:
                out[i] = np.mean([wv[w] for w in ws], 0)
        return out
    return f


def _sigmoid_probs(d):
    p = 1.0 / (1.0 + np.exp(-d))
    return np.stack([1 - p, p], 1).astype(np.float32)


def _pick(make, Xtr, ytr, Xdev, ydev, grid):
    best_acc, best_m, best_c = -1, None, None
    for c in grid:
        m = make(c).fit(Xtr, ytr)
        acc = float((m.predict(Xdev) == ydev).mean())
        if acc > best_acc:
            best_acc, best_m, best_c = acc, m, c
    return best_m, best_acc, best_c


def vader_probs_fn():
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    an = SentimentIntensityAnalyzer()

    def f(texts):
        c = np.array([an.polarity_scores(t)["compound"] for t in texts])
        p = (c + 1) / 2
        return np.stack([1 - p, p], 1).astype(np.float32)
    return f


def fit_family(name, B, n, seed, emb=None):
    """Fit one family on the (n, seed) subset. Returns (probability function, info dict)."""
    idx = subsample(B.tr_y, n, seed)
    tr_t = [B.tr_t[i] for i in idx]; tr_y = B.tr_y[idx]
    t0 = time.time()
    if name == "vader_lexicon":
        return vader_probs_fn(), dict(dev_acc=float((vader_probs_fn()(B.dev_t).argmax(1) == B.dev_y).mean()), hp=None, train_time_s=0.0)
    if name == "tfidf_svm":
        vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True).fit(tr_t)
        m, acc, c = _pick(lambda c: LinearSVC(C=c), vec.transform(tr_t), tr_y, vec.transform(B.dev_t), B.dev_y, [0.03, 0.1, 0.3, 1])
        return (lambda t: _sigmoid_probs(m.decision_function(vec.transform(t)))), dict(dev_acc=acc, hp=c, train_time_s=time.time() - t0)
    if name == "tfidf_char_lr":
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True, min_df=2).fit(tr_t)
        m, acc, c = _pick(lambda c: LogisticRegression(C=c, max_iter=2000), vec.transform(tr_t), tr_y, vec.transform(B.dev_t), B.dev_y, [1, 10, 30])
        return (lambda t: m.predict_proba(vec.transform(t)).astype(np.float32)), dict(dev_acc=acc, hp=c, train_time_s=time.time() - t0)
    if name in ("w2v_avg_lr", "fasttext_avg_lr"):
        wv = emb[0] if name == "w2v_avg_lr" else emb[1]
        fx = _avg(wv, subword=(name == "fasttext_avg_lr"))
        m, acc, c = _pick(lambda c: make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=3000)),
                          fx(tr_t), tr_y, fx(B.dev_t), B.dev_y, [0.01, 0.1, 1])
        return (lambda t: m.predict_proba(fx(t)).astype(np.float32)), dict(dev_acc=acc, hp=c, train_time_s=time.time() - t0)
    raise ValueError(name)
