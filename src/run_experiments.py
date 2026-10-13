"""Run every experiment of the study (resumable: finished runs are skipped).

    python run_experiments.py --stage all          # pilot -> curve -> ckpt
    python run_experiments.py --stage curve

Stages
  pilot  learning-rate selection on the dev set (n=300 & 3000, seed 0)
  curve  learning curves: {tfidf_lr, tfidf_nb, tiny-scratch, tiny-mlm} x n x seeds
         (+ robustness to perturbations and OOD IMDb accuracy for every run)
  ckpt   pre-training-steps ablation at n=1000 (fine-tune + frozen linear probe)
  families  non-transformer model families (lexicon, SVM, char n-grams, static embeddings) -> results/families.jsonl
"""
import argparse, json, time
import numpy as np

from config import DOMAIN_TRAIN_SIZES, MODELS, SEEDS, TRAIN_SIZES
import json as _json
from config import PREDS, RES
from exp_utils import Bundle, RESULTS_FILE, done_keys, evaluate, linear_probe, log, run_neural, run_tfidf

LR_GRID = [2e-4, 5e-4, 1e-3, 2e-3, 4e-3]
CKPT_STEPS = [0, 100, 300, 1000, 2500, 5000]
SCRATCH_SRC = str(MODELS / "mlm_step0")     # only its config (architecture) + tokenizer are used
MLM_SRC = str(MODELS / "mlm_final")


def best_lrs(domain="sst2"):
    recs = [json.loads(l) for l in open(RESULTS_FILE)]
    out = {}
    for m in ("tiny_scratch", "tiny_mlm"):
        acc = {}
        for r in recs:
            if r["stage"] == "pilot" and r["model"] == m and r.get("domain", "sst2") == domain:
                acc.setdefault(r["lr"], []).append(r["dev_acc"])
        out[m] = max(acc, key=lambda k: np.mean(acc[k])) if acc else 2e-4
    return out


def pilot_sizes(domain):
    sizes = DOMAIN_TRAIN_SIZES[domain]
    return (sizes[1], sizes[3]) if len(sizes) >= 4 else (sizes[0], sizes[-1])


def stage_pilot(B, done):
    for init, name, src in (("scratch", "tiny_scratch", SCRATCH_SRC), ("mlm", "tiny_mlm", MLM_SRC)):
        for n in pilot_sizes(B.domain):
            for lr in LR_GRID:
                if ("pilot", name, n, 0, f"lr{lr}", B.domain) in done:
                    continue
                try:
                    r = run_neural(B, "pilot", name, init, src, n, 0, lr, tag=f"lr{lr}")
                except Exception as e:
                    print(f"[pilot] SKIPPED {name} n={n} lr={lr:.0e}: {type(e).__name__}: {e}", flush=True)
                    continue
                log(r); print(f"[pilot] {B.domain:8s} {name:13s} n={n:5d} lr={lr:.0e} dev={r['dev_acc']:.4f}", flush=True)


def stage_curve(B, done):
    lrs = best_lrs(B.domain); print(f"[curve] {B.domain} selected learning rates:", lrs)
    sizes = DOMAIN_TRAIN_SIZES[B.domain]
    ood_key = "imdb_acc" if B.domain == "sst2" else f"cross_{[d for d in ('sst2','finance','airline') if d != B.domain][0]}_acc"
    for n in sizes:
        for seed in SEEDS:
            for kind in ("tfidf_lr", "tfidf_nb"):
                if ("curve", kind, n, seed, "", B.domain) not in done:
                    try:
                        r = run_tfidf(B, kind, n, seed); log(r)
                        print(f"[curve] {B.domain:8s} {kind:13s} n={n:5d} seed={seed} test={r['test_acc']:.4f} "
                              f"{ood_key}={r.get(ood_key, float('nan')):.4f}", flush=True)
                    except Exception as e:
                        print(f"[curve] SKIPPED {kind} n={n} seed={seed}: {type(e).__name__}: {e}", flush=True)
            for init, name, src in (("scratch", "tiny_scratch", SCRATCH_SRC), ("mlm", "tiny_mlm", MLM_SRC)):
                if ("curve", name, n, seed, "", B.domain) in done:
                    continue
                save = (MODELS / f"ft_{name}_full_seed0_{B.domain}.pt") if (n == sizes[-1] and seed == 0 and B.domain != "sst2") else                        (MODELS / f"ft_{name}_full_seed0.pt") if (n == sizes[-1] and seed == 0) else None
                try:
                    r = run_neural(B, "curve", name, init, src, n, seed, lrs[name], save_model=save)
                except Exception as e:
                    print(f"[curve] SKIPPED {name} n={n} seed={seed}: {type(e).__name__}: {e}", flush=True)
                    continue
                log(r)
                print(f"[curve] {B.domain:8s} {name:13s} n={n:5d} seed={seed} test={r['test_acc']:.4f} "
                      f"{ood_key}={r.get(ood_key, float('nan')):.4f} typo30={r['typo30_acc']:.4f} ({r['train_time_s']:.0f}s)", flush=True)


def stage_ckpt(B, done):
    if B.domain != "sst2":
        print(f"[ckpt] skipped for domain={B.domain}: the pre-training-length ablation uses the SST-2/movie-review "
              "MLM checkpoint ladder only (results/models/mlm_step*), not re-derived per domain.")
        return
    lr = best_lrs()["tiny_mlm"]
    for step in CKPT_STEPS:
        for seed in SEEDS:
            key = ("ckpt", "tiny_mlm", 1000, seed, f"step{step}", "sst2")
            if key in done:
                continue
            src = str(MODELS / f"mlm_step{step}")
            r = run_neural(B, "ckpt", "tiny_mlm", "mlm", src, 1000, seed, lr, tag=f"step{step}")
            r["probe_acc"] = linear_probe(src, B, 1000, seed)
            log(r)
            print(f"[ckpt] step={step:5d} seed={seed} finetune_test={r['test_acc']:.4f} probe={r['probe_acc']:.4f}", flush=True)


def stage_families(B):
    """Non-transformer families under the same protocol; results go to results/families.jsonl (resumable).
    Static embeddings (word2vec/fastText) are trained on THIS domain's own unlabeled text (train+dev),
    so every family, on every domain, only ever sees that domain's own words -- except VADER, which is a
    fixed general-purpose lexicon by design and applies unchanged to all domains."""
    from families import FAMILY_MODELS, fit_family, get_embeddings
    f = RES / "families.jsonl"
    done = {(r["model"], r["n"], r["seed"], r.get("domain", "sst2")) for r in map(_json.loads, open(f))} if f.exists() else set()
    emb = get_embeddings(B.domain, B.tr_t + B.dev_t if B.domain != "sst2" else None)
    zero = {}
    ood_key = "imdb_acc" if B.domain == "sst2" else f"cross_{[d for d in ('sst2','finance','airline') if d != B.domain][0]}_acc"
    for n in DOMAIN_TRAIN_SIZES[B.domain]:
        for seed in SEEDS:
            for name in FAMILY_MODELS:
                if (name, n, seed, B.domain) in done:
                    continue
                path = PREDS / (f"family_{name}__{n}_{seed}.npy" if B.domain == "sst2" else f"family_{name}__{B.domain}_{n}_{seed}.npy")
                if name == "vader_lexicon" and B.domain in zero:          # no training: evaluate once per domain, reuse for every (n, seed)
                    rec = dict(zero[B.domain][0], n=n, seed=seed)
                    np.save(path, zero[B.domain][1])
                else:
                    try:
                        pfn, info = fit_family(name, B, n, seed, emb)
                        ev, test_p = evaluate(pfn, B)
                    except Exception as e:
                        print(f"[family] SKIPPED {name} n={n} seed={seed}: {type(e).__name__}: {e}", flush=True)
                        continue
                    np.save(path, test_p)
                    rec = dict(stage="family", domain=B.domain, model=name, n=n, seed=seed, **info, **ev)
                    if name == "vader_lexicon":
                        zero[B.domain] = (rec, test_p)
                with open(f, "a") as fh:
                    fh.write(_json.dumps(rec) + "\n")
                print(f"[family] {B.domain:8s} {name:16s} n={n:5d} seed={seed} test={rec['test_acc']:.4f} "
                      f"typo30={rec['typo30_acc']:.4f} {ood_key}={rec.get(ood_key, float('nan')):.4f}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=["pilot", "curve", "ckpt", "families", "all"])
    ap.add_argument("--domain", default="sst2", choices=["sst2", "finance", "airline"],
                     help="sst2 (default, original study), finance (Financial PhraseBank) or airline (Twitter Airline Sentiment)")
    a = ap.parse_args(); B = Bundle(domain=a.domain)
    for st, fn in (("pilot", stage_pilot), ("curve", stage_curve), ("ckpt", stage_ckpt)):
        if a.stage in (st, "all"):
            fn(B, done_keys())
    if a.stage in ("families", "all"):
        stage_families(B)
