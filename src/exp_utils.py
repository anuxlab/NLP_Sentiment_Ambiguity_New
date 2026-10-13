"""Evaluation sets + single-run helpers used by run_experiments.py and hf_benchmark.py.

Multi-domain (RQ8): `Bundle(domain=...)` supports "sst2" (default, unchanged filenames/behaviour for full
backward compatibility with existing results), "finance" (Financial PhraseBank, real annotator-agreement
tiers) and "airline" (Twitter US Airline Sentiment, real per-tweet confidence scores). Every Bundle also
loads the *test* sets of the other two domains so every run reports in-domain AND cross-domain accuracy
(cross_<other domain>_acc) in the same result record -- no extra training or storage needed for the
cross-domain transfer matrix.
"""
import json, time
import numpy as np
import torch

from baselines import fit_tfidf, proba
from config import DOMAIN_TRAIN_SIZES, DOMAINS, MODELS, PREDS, RES
from data import DOMAIN_LOADERS, PERTURBATIONS, chunk_words, load_movie_reviews, load_sst2, make_perturbed, subsample
from finetune import build_model, fit, load_tokenizer, metrics, predict_proba

RESULTS_FILE = RES / "results.jsonl"
_CROSS_CACHE = {}   # domain -> (texts, labels) for the OTHER domains' test sets; loaded at most once per process


def _other_test_sets(domain):
    for d in DOMAINS:
        if d not in _CROSS_CACHE:
            if d == "sst2":
                _CROSS_CACHE[d] = load_sst2()["test"]
            else:
                splits, _ = DOMAIN_LOADERS[d]()
                _CROSS_CACHE[d] = splits["test"]
    return {d: _CROSS_CACHE[d] for d in DOMAINS if d != domain}


class Bundle:
    """Loads one domain's data and builds all evaluation sets (clean, perturbed, OOD/cross-domain)."""

    def __init__(self, domain="sst2"):
        self.domain = domain
        if domain == "sst2":
            self.sst = load_sst2()
            self.tr_t, self.tr_y = self.sst["train"]
            self.dev_t, self.dev_y = self.sst["dev"]
            te_t, te_y = self.sst["test"]
            self.test_meta = None   # real-signal ambiguity metadata: SST-2 uses the separate SST-5 proxy (ambiguity.py)
        else:
            splits, test_meta = DOMAIN_LOADERS[domain]()
            self.tr_t, self.tr_y = splits["train"]
            self.dev_t, self.dev_y = splits["dev"]
            te_t, te_y = splits["test"]
            self.test_meta = test_meta   # tier (finance) or confidence (airline): REAL ambiguity signal, not a proxy
        self.sets = {"test": (te_t, te_y)}
        for k in PERTURBATIONS:
            self.sets[k] = (make_perturbed(te_t, k), te_y)
        if domain == "sst2":                                    # keep the original single-domain OOD test unchanged
            _, (ho_t, ho_y) = load_movie_reviews()
            self.imdb_y = ho_y
            self.imdb_chunks, self.imdb_owner = [], []
            for i, t in enumerate(ho_t):
                c = chunk_words(t)
                self.imdb_chunks += c; self.imdb_owner += [i] * len(c)
            self.imdb_owner = np.array(self.imdb_owner)
        self.cross = _other_test_sets(domain)                    # {other_domain: (texts, labels)}, for RQ8


def evaluate(pfn, B: Bundle):
    """pfn: list[str] -> (N,2) class probabilities. Returns (metrics dict, clean test probs)."""
    out = {}
    for name, (texts, y) in B.sets.items():
        p = pfn(texts)
        m = metrics(y, p)
        out[f"{name}_acc"] = m["acc"]
        if name == "test":
            out["test_f1"] = m["f1"]; test_p = p
    if B.domain == "sst2":
        pc = pfn(B.imdb_chunks)                                # OOD: average chunk probabilities per document
        doc = np.stack([pc[B.imdb_owner == i].mean(0) for i in range(len(B.imdb_y))])
        out["imdb_acc"] = metrics(B.imdb_y, doc)["acc"]
    for other, (o_t, o_y) in B.cross.items():                  # RQ8: cross-domain transfer, same trained model
        m = metrics(o_y, pfn(o_t))
        out[f"cross_{other}_acc"] = m["acc"]; out[f"cross_{other}_f1"] = m["f1"]
    return out, test_p


def log(rec):
    with open(RESULTS_FILE, "a") as f:
        f.write(json.dumps(rec) + "\n")


def done_keys():
    if not RESULTS_FILE.exists():
        return set()
    return {(r["stage"], r["model"], r["n"], r["seed"], r.get("tag", ""), r.get("domain", "sst2"))
            for r in map(json.loads, open(RESULTS_FILE))}


def _pred_path(stage, model_name, tag, domain, n, seed):
    """sst2 keeps the ORIGINAL filenames (no domain suffix) so existing results/preds/*.npy stay valid;
    other domains get a domain suffix to avoid collisions."""
    if domain == "sst2":
        return PREDS / f"{stage}_{model_name}_{tag}_{n}_{seed}.npy"
    return PREDS / f"{stage}_{model_name}_{tag}_{domain}_{n}_{seed}.npy"


def run_neural(B, stage, model_name, init, source, n, seed, lr, tag="", save_model=None, tok_source=None):
    tok = load_tokenizer(tok_source or source)
    idx = subsample(B.tr_y, n, seed)
    tr_t = [B.tr_t[i] for i in idx]; tr_y = B.tr_y[idx]
    model = build_model(init, source, seed)
    info = fit(model, tok, tr_t, tr_y, B.dev_t, B.dev_y, lr=lr, seed=seed)
    ev, test_p = evaluate(lambda t: predict_proba(model, tok, t), B)
    np.save(_pred_path(stage, model_name, tag, B.domain, n, seed), test_p)
    if save_model:
        torch.save(model.state_dict(), save_model)
    rec = dict(stage=stage, domain=B.domain, model=model_name, n=n, seed=seed, lr=lr, tag=tag,
               dev_acc=info["dev_acc"], best_epoch=info["best_epoch"], epochs=info["epochs"],
               train_time_s=info["train_time_s"], **ev)
    return rec


def run_tfidf(B, kind, n, seed):
    idx = subsample(B.tr_y, n, seed)
    tr_t = [B.tr_t[i] for i in idx]; tr_y = B.tr_y[idx]
    pipe, info = fit_tfidf(kind, tr_t, tr_y, B.dev_t, B.dev_y)
    ev, test_p = evaluate(lambda t: proba(pipe, t), B)
    if B.domain == "sst2":
        np.save(PREDS / f"curve_{kind}__{n}_{seed}.npy", test_p)
    else:
        np.save(PREDS / f"curve_{kind}__{B.domain}_{n}_{seed}.npy", test_p)
    return dict(stage="curve", domain=B.domain, model=kind, n=n, seed=seed, lr=None, tag="", hp=info["hp"],
                dev_acc=info["dev_acc"], train_time_s=info["train_time_s"], **ev)


@torch.no_grad()
def linear_probe(ckpt, B, n, seed, batch=256):
    """Frozen-encoder probe: logistic regression on mean-pooled hidden states."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from finetune import collate, encode
    tok = load_tokenizer(ckpt)
    enc = build_model("mlm", ckpt, seed).enc.eval()

    def feats(texts):
        ids = encode(tok, texts); out = []
        for i in range(0, len(ids), batch):
            x, a = collate(ids[i:i + batch], tok.pad_token_id)
            h = enc(input_ids=x, attention_mask=a).last_hidden_state
            m = a.unsqueeze(-1).float()
            out.append(((h * m).sum(1) / m.sum(1)).numpy())
        return np.concatenate(out)

    idx = subsample(B.tr_y, n, seed)
    Xtr = feats([B.tr_t[i] for i in idx]); sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(C=1.0, max_iter=3000).fit(sc.transform(Xtr), B.tr_y[idx])
    Xte = feats(B.sets["test"][0])
    return float((clf.predict(sc.transform(Xte)) == B.sets["test"][1]).mean())
