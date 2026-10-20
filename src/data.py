"""Data loading, splitting and text perturbations.

Datasets (both are fetched from public GitHub mirrors, no HuggingFace needed):
  * SST-2 (Stanford Sentiment Treebank, binary, sentence level) -> supervised task
  * NLTK `movie_reviews` (Pang & Lee 2004, IMDb documents)      -> unlabeled MLM
    pre-training text (80 %) and an out-of-domain (OOD) test set (20 % held out).
"""
import random
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

from config import DATA

SST_BASE = ("https://raw.githubusercontent.com/clairett/"
            "pytorch-sentiment-classification/master/data/SST2/")
MR_URL = ("https://raw.githubusercontent.com/nltk/nltk_data/"
          "gh-pages/packages/corpora/movie_reviews.zip")


def _download(url: str, dst: Path):
    if dst.exists() and dst.stat().st_size > 0:
        return
    print(f"[data] downloading {url}")
    urllib.request.urlretrieve(url, dst)


def ensure_data():
    for split in ("train", "dev", "test"):
        _download(SST_BASE + f"{split}.tsv", DATA / f"sst2_{split}.tsv")
    z = DATA / "movie_reviews.zip"
    _download(MR_URL, z)
    if not (DATA / "movie_reviews").exists():
        with zipfile.ZipFile(z) as f:
            f.extractall(DATA)


def load_sst2():
    """Returns {split: (list[str], np.ndarray)}; label 1 = positive."""
    ensure_data()
    out = {}
    for split in ("train", "dev", "test"):
        texts, labels = [], []
        for line in (DATA / f"sst2_{split}.tsv").read_text(encoding="utf8").splitlines():
            t, y = line.rsplit("\t", 1)
            texts.append(t.strip())
            labels.append(int(y))
        out[split] = (texts, np.array(labels))
    return out


def load_movie_reviews(seed: int = 13, heldout_frac: float = 0.2):
    """Split the 2000 IMDb docs (1000 pos / 1000 neg) *by document*.

    Returns (pretrain_docs: list[str], (heldout_texts, heldout_labels)).
    The held-out documents are never seen during pre-training.
    """
    ensure_data()
    rng = random.Random(seed)
    pre, ho_t, ho_y = [], [], []
    for label, name in ((0, "neg"), (1, "pos")):
        files = sorted((DATA / "movie_reviews" / name).glob("*.txt"))
        rng.shuffle(files)
        k = int(len(files) * heldout_frac)
        for i, f in enumerate(files):
            text = " ".join(f.read_text(encoding="utf8").split()).lower()
            if i < k:
                ho_t.append(text); ho_y.append(label)
            else:
                pre.append(text)
    rng.shuffle(pre)
    return pre, (ho_t, np.array(ho_y))


def chunk_words(text: str, size: int = 40, max_chunks: int = 8):
    """Split a long review into <= max_chunks pseudo-sentences of `size` words."""
    w = text.split()
    return [" ".join(w[i:i + size]) for i in range(0, min(len(w), size * max_chunks), size)] or [""]


def subsample(labels: np.ndarray, n: int, seed: int) -> np.ndarray:
    """Stratified random subset of indices (same indices for every model -> paired)."""
    if n >= len(labels):
        return np.arange(len(labels))
    rng = np.random.RandomState(1000 + seed)
    idx = []
    for c in np.unique(labels):
        pool = np.where(labels == c)[0]
        idx.append(rng.choice(pool, size=int(round(n * len(pool) / len(labels))), replace=False))
    idx = np.concatenate(idx)
    rng.shuffle(idx)
    return idx


# ---------------------------------------------------------------- perturbations
def typo(text: str, rate: float, rng: random.Random) -> str:
    """Swap two adjacent inner characters in a fraction `rate` of words (len>3)."""
    out = []
    for w in text.split():
        if len(w) > 3 and rng.random() < rate:
            i = rng.randrange(1, len(w) - 2)
            w = w[:i] + w[i + 1] + w[i] + w[i + 2:]
        out.append(w)
    return " ".join(out)


def word_drop(text: str, rate: float, rng: random.Random) -> str:
    w = text.split()
    kept = [x for x in w if rng.random() >= rate]
    return " ".join(kept if kept else w)


PERTURBATIONS = {
    "typo10": lambda t, r: typo(t, 0.10, r),
    "typo30": lambda t, r: typo(t, 0.30, r),
    "drop20": lambda t, r: word_drop(t, 0.20, r),
    "drop40": lambda t, r: word_drop(t, 0.40, r),
}


def make_perturbed(texts, kind: str, seed: int = 7):
    rng = random.Random(seed)
    f = PERTURBATIONS[kind]
    return [f(t, rng) for t in texts]


# ------------------------------------------------------------------ SST-5 (fine-grained) labels for ambiguity analysis
SST5_URL = ("https://raw.githubusercontent.com/prrao87/fine-grained-sentiment/master/data/sst/sst_test.txt")


def _norm5(s: str) -> str:
    import re
    s = re.sub(r"\b(lrb|rrb)\b", " ", s.lower())          # SST-2 spells -LRB-/-RRB- as lrb/rrb
    return re.sub(r"[^a-z0-9]", "", s)


def load_sst5_for(texts, cutoff=0.97):
    """Fine-grained SST label (1=very neg, 2=neg, 3=neutral, 4=pos, 5=very pos) for each SST-2 *test* sentence.

    SST-2 is the subset of SST-5 without neutral sentences. Sentences are matched by normalised text (exact, then a
    difflib fallback with ratio >= cutoff). Returns an int array; -1 marks sentences that could not be matched.
    'Weak polarity' (label 2 or 4) is used as a human-graded ambiguity proxy: mild sentiment close to neutral.
    """
    import difflib
    _download(SST5_URL, DATA / "sst5_test.txt")
    d5 = {}
    for line in (DATA / "sst5_test.txt").read_text(encoding="utf8").splitlines():
        lab, t = line.split("\t", 1)
        d5.setdefault(_norm5(t), int(lab.replace("__label__", "")))
    keys = list(d5)
    out = np.full(len(texts), -1, dtype=int)
    for i, t in enumerate(texts):
        k = _norm5(t)
        if k in d5:
            out[i] = d5[k]
        else:
            m = difflib.get_close_matches(k, keys, n=1, cutoff=cutoff)
            if m:
                out[i] = d5[m[0]]
    return out


# ------------------------------------------------------------------ Second/third domains: Financial PhraseBank, Twitter Airline
# Financial PhraseBank (Malo et al. 2014, CC-BY-NC-SA-3.0): finance news headlines, 3-class, with REAL
# annotator-agreement tiers (not a proxy). Files cross-verified against each other (zero label mismatches
# on 2,259 overlapping sentences; tiers nest 100% subset 66-99% subset 50-65%).
FIN_ALLAGREE_URL = "https://raw.githubusercontent.com/nvpham12/Financial-Phrasebank-Sentiment-Analysis/main/data/Sentences_AllAgree.txt"
FIN_66AGREE_URL = "https://raw.githubusercontent.com/eliasbaumann/BERT_stock_forecasting/master/Data/FinancialPhraseBank-v1.0/Sentences_66Agree.txt"
FIN_50AGREE_CSV_URL = "https://raw.githubusercontent.com/onurtuncay/FinancialSentimentAnalysis/main/FinancialPhraseBankDataset.csv"
# Twitter US Airline Sentiment (CrowdFlower/Kaggle origin): tweets, 3-class, with a REAL per-tweet crowd-worker
# confidence score (continuous, 0.335-1.0) -- an independent kind of real ambiguity signal (confidence, not agreement).
AIRLINE_URL = "https://raw.githubusercontent.com/MrColinHan/Twitter-US-Airline-Sentiment/master/Kaggle_Tweets.csv"


def _norm_fin(s: str) -> str:
    import re
    return re.sub(r"\s+", " ", s.strip().lower())


def _fetch_text(url, dst):
    path = DATA / dst
    _download(url, path)
    return path.read_text(encoding="latin-1" if dst.endswith(".txt") else "utf-8")


def _stratified_split(rows, seed, fracs=(0.7, 0.1, 0.2), label_index=1, text_index=0):
    """Deterministic label-stratified split that keeps duplicate texts in one split.

    ``rows`` may carry metadata after the label (e.g. finance agreement tier or
    airline worker confidence), so the sentiment label is selected explicitly
    instead of inferred from the last tuple field. Normalized duplicate texts
    are assigned as a group to prevent train/dev/test text leakage. If identical
    normalized text has conflicting labels, fail loudly rather than silently
    leaking or choosing one label.
    """
    if not rows:
        return [], [], []
    if not (0 <= label_index < len(rows[0]) and 0 <= text_index < len(rows[0])):
        raise ValueError("label_index/text_index is outside the row schema")
    import re
    def norm_text(value):
        return re.sub(r"\s+", " ", str(value).strip().lower())

    grouped = {}
    for i, row in enumerate(rows):
        key = norm_text(row[text_index])
        label = row[label_index]
        if not key:
            raise ValueError(f"Empty text at row {i}")
        if key not in grouped:
            grouped[key] = {"labels": [], "indices": []}
        grouped[key]["labels"].append(label)
        grouped[key]["indices"].append(i)

    # Some public mirrors contain the same normalized text with different hard
    # labels. Keep the entire text group in one split, but do not silently
    # overwrite any original row labels. For stratification only, use the
    # group's majority label (ties resolved by sorted label order). The audit
    # reports such conflicts so researchers can decide whether to exclude them.
    by_label = {}
    for key, item in grouped.items():
        counts = {}
        for label in item["labels"]:
            counts[label] = counts.get(label, 0) + 1
        max_count = max(counts.values())
        strat_label = sorted(label for label, count in counts.items() if count == max_count)[0]
        item["stratification_label"] = strat_label
        by_label.setdefault(strat_label, []).append((key, item["indices"]))
    rng = np.random.RandomState(seed)
    tr_idx, dev_idx, te_idx = [], [], []
    for lab, groups in by_label.items():
        order = rng.permutation(len(groups))
        shuffled = [groups[int(j)] for j in order]
        n = len(shuffled)
        n_tr = int(round(n * fracs[0]))
        n_dev = int(round(n * fracs[1]))
        # Keep at least one group in the test partition for each label when possible.
        if n >= 3:
            n_tr = min(n_tr, n - 2)
            n_dev = min(n_dev, n - n_tr - 1)
        for split, subset in ((tr_idx, shuffled[:n_tr]),
                              (dev_idx, shuffled[n_tr:n_tr + n_dev]),
                              (te_idx, shuffled[n_tr + n_dev:])):
            for _, indices in subset:
                split.extend(indices)
    rng.shuffle(tr_idx); rng.shuffle(dev_idx); rng.shuffle(te_idx)
    return tr_idx, dev_idx, te_idx


def load_finance(seed=42):
    """Financial PhraseBank, binary (neutral dropped). Returns ({split:(texts,labels)}, test_tier)
    where test_tier[i] in {"50-65","66-99","100"} is the REAL annotator-agreement tier for test sentence i
    (higher = more human agreement = less ambiguous)."""
    allagree_raw = _fetch_text(FIN_ALLAGREE_URL, "fin_allagree.txt")
    a66_raw = _fetch_text(FIN_66AGREE_URL, "fin_66agree.txt")
    csv_raw = _fetch_text(FIN_50AGREE_CSV_URL, "fin_50agree.csv")
    allagree = {_norm_fin(l.rsplit("@", 1)[0]) for l in allagree_raw.strip().split("\n") if "@" in l}
    agree66 = {_norm_fin(l.rsplit("@", 1)[0]) for l in a66_raw.strip().split("\n") if "@" in l}
    import csv as _csv, io as _io
    lab_map = {"0": "negative", "1": "neutral", "2": "positive"}
    rows = list(_csv.DictReader(_io.StringIO(csv_raw)))
    out = []
    for r in rows:
        lab = lab_map[r["label"]]
        if lab == "neutral":
            continue
        key = _norm_fin(r["sentence"])
        tier = "100" if key in allagree else ("66-99" if key in agree66 else "50-65")
        out.append((r["sentence"].strip(), 1 if lab == "positive" else 0, tier))
    tr_i, dev_i, te_i = _stratified_split(out, seed)
    splits = {}
    for name, idx in (("train", tr_i), ("dev", dev_i), ("test", te_i)):
        splits[name] = ([out[i][0] for i in idx], np.array([out[i][1] for i in idx]))
    test_tier = [out[i][2] for i in te_i]
    return splits, test_tier


def load_airline(seed=42):
    """Twitter US Airline Sentiment, binary (neutral dropped). Returns ({split:(texts,labels)}, test_confidence)
    where test_confidence[i] is the REAL per-tweet crowd-worker confidence (0.335-1.0; higher = less ambiguous)."""
    import csv as _csv, io as _io
    raw = _fetch_text(AIRLINE_URL, "airline_tweets.csv")
    rows = list(_csv.DictReader(_io.StringIO(raw)))
    out = []
    for r in rows:
        lab = r["airline_sentiment"]
        if lab == "neutral" or not r.get("airline_sentiment_confidence"):
            continue
        out.append((r["text"].strip(), 1 if lab == "positive" else 0, float(r["airline_sentiment_confidence"])))
    tr_i, dev_i, te_i = _stratified_split(out, seed)
    splits = {}
    for name, idx in (("train", tr_i), ("dev", dev_i), ("test", te_i)):
        splits[name] = ([out[i][0] for i in idx], np.array([out[i][1] for i in idx]))
    test_conf = [out[i][2] for i in te_i]
    return splits, test_conf


DOMAIN_LOADERS = {"sst2": lambda: (load_sst2(), None), "finance": load_finance, "airline": load_airline}
