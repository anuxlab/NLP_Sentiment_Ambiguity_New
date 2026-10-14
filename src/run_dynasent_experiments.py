"""End-to-end DynaSent soft-label experiments (offline-first).

Quick CPU pilot (TF-IDF linear classifier):
  python src/run_dynasent_experiments.py --data data/dynasent-main.zip --round round1 --backend tfidf --budget 100 --eval-size 500 --seeds 0

Neural tiny-BERT run (requires the project's transformers/tokenizers dependencies):
  python src/run_dynasent_experiments.py --data data/dynasent-main.zip --round round1 --backend tiny-bert --checkpoint results/models/mlm_final --budget 100 --eval-size 500 --seeds 0

The three matched conditions use the same labeled training subset: majority-label CE,
human-distribution soft-target CE, and generic label smoothing. Examples with null
majority labels are excluded from this matched comparison; an optional separate
soft-target condition can include them using --include-null-soft-only.
"""
from __future__ import annotations
import argparse, csv, json, math, random, sys, time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import normalize

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
from dynasent_data import LABELS, load_dynasent_round
from soft_targets import hard_label_cross_entropy, jensen_shannon_divergence, soft_target_cross_entropy


def seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def macro_brier(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean(np.sum((pred - target) ** 2, axis=1)))


def ece_score(pred: np.ndarray, labels: np.ndarray, bins: int = 10) -> float:
    conf = pred.max(axis=1); guess = pred.argmax(axis=1)
    ece = 0.0
    edges = np.linspace(0, 1, bins + 1)
    for i in range(bins):
        mask = (conf > edges[i]) & (conf <= edges[i + 1] if i < bins - 1 else conf <= edges[i + 1])
        if mask.any(): ece += float(mask.mean()) * abs(float((guess[mask] == labels[mask]).mean()) - float(conf[mask].mean()))
    return ece


def evaluate(probs: np.ndarray, rows: list[dict[str, Any]]) -> dict[str, Any]:
    target = np.asarray([r["target_probs"] for r in rows], dtype=np.float32)
    js = jensen_shannon_divergence(torch.tensor(probs), torch.tensor(target), reduction="none").numpy()
    gold_mask = np.asarray([r["gold_label_id"] is not None for r in rows])
    result: dict[str, Any] = {
        "n_test": len(rows), "js_divergence_mean_bits": float(js.mean()),
        "js_divergence_median_bits": float(np.median(js)),
        "soft_brier_score": macro_brier(probs, target),
        "human_distribution_entropy_mean_bits": float(np.mean(-np.sum(target * np.log2(np.clip(target, 1e-12, 1)), axis=1))),
        "n_with_majority_gold": int(gold_mask.sum()),
    }
    if gold_mask.any():
        y = np.asarray([r["gold_label_id"] if r["gold_label_id"] is not None else -1 for r in rows])[gold_mask]
        p = probs[gold_mask]
        result.update({"accuracy": float(accuracy_score(y, p.argmax(1))),
                       "macro_f1": float(f1_score(y, p.argmax(1), labels=list(range(len(LABELS))), average="macro", zero_division=0)),
                       "ece_10bin": ece_score(p, y)})
    else:
        result.update({"accuracy": None, "macro_f1": None, "ece_10bin": None})
    return result


def sample_rows(rows: list[dict[str, Any]], n: int, seed: int) -> list[dict[str, Any]]:
    if n <= 0 or n >= len(rows): return list(rows)
    idx = np.random.RandomState(seed).choice(len(rows), size=n, replace=False)
    return [rows[int(i)] for i in idx]


def fit_tfidf(train_rows, eval_rows, condition, seed, args):
    """Train a CPU-friendly linear baseline using the same features across conditions."""
    seed_everything(seed)
    started = time.time()
    vectorizer = TfidfVectorizer(max_features=args.max_features, ngram_range=(1, 2), min_df=2,
                                 sublinear_tf=True, strip_accents="unicode")
    Xtr = vectorizer.fit_transform([r["text"] for r in train_rows]).toarray().astype(np.float32)
    Xev = vectorizer.transform([r["text"] for r in eval_rows]).toarray().astype(np.float32)
    target = np.asarray([r["target_probs"] for r in train_rows], dtype=np.float32)
    gold = np.asarray([r["gold_label_id"] if r["gold_label_id"] is not None else -1 for r in train_rows], dtype=np.int64)
    if condition in ("majority_ce", "label_smoothing") and (gold < 0).any():
        raise ValueError("Hard-label baselines require non-null gold labels")
    model = torch.nn.Linear(Xtr.shape[1], len(LABELS))
    torch.nn.init.xavier_uniform_(model.weight); torch.nn.init.zeros_(model.bias)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    xt = torch.from_numpy(Xtr); xv = torch.from_numpy(Xev); yt = torch.from_numpy(target); yg = torch.from_numpy(gold)
    rng = np.random.RandomState(seed)
    for _epoch in range(args.epochs):
        model.train()
        order = rng.permutation(len(train_rows))
        for start in range(0, len(order), args.batch_size):
            ids = torch.from_numpy(order[start:start + args.batch_size])
            logits = model(xt[ids])
            if condition.startswith("human_soft_targets"): loss = soft_target_cross_entropy(logits, yt[ids])
            elif condition == "majority_ce": loss = hard_label_cross_entropy(logits, yg[ids])
            elif condition == "label_smoothing": loss = hard_label_cross_entropy(logits, yg[ids], label_smoothing=args.label_smoothing)
            else: raise ValueError(condition)
            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    model.eval()
    with torch.no_grad(): probs = torch.softmax(model(xv), dim=-1).cpu().numpy()
    return probs, time.time() - started, {"vocabulary_size": len(vectorizer.vocabulary_), "feature_type": "tfidf_word_unigram_bigram"}


def fit_tiny_bert(train_rows, eval_rows, condition, seed, args):
    """Train local tiny BERT; no Hub calls are made."""
    try:
        from transformers import AutoTokenizer, BertModel, BertConfig
    except ImportError as exc:
        raise RuntimeError("tiny-bert backend needs transformers and tokenizers installed; use --backend tfidf for a CPU smoke run") from exc
    seed_everything(seed)
    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists(): raise FileNotFoundError(f"Local checkpoint not found: {checkpoint}")
    tok = AutoTokenizer.from_pretrained(str(checkpoint), local_files_only=True)
    enc = BertModel.from_pretrained(str(checkpoint), add_pooling_layer=False, local_files_only=True)
    class Head(torch.nn.Module):
        def __init__(self, encoder):
            super().__init__(); self.encoder=encoder; self.drop=torch.nn.Dropout(0.1); self.head=torch.nn.Linear(encoder.config.hidden_size, len(LABELS))
        def forward(self, ids, mask):
            h=self.encoder(input_ids=ids, attention_mask=mask).last_hidden_state
            m=mask.unsqueeze(-1).to(h.dtype); pooled=(h*m).sum(1)/m.sum(1).clamp(min=1)
            return self.head(self.drop(pooled))
    model=Head(enc); opt=torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.device == "cuda" and not torch.cuda.is_available(): raise RuntimeError("--device cuda requested but CUDA is unavailable")
    device=torch.device("cuda" if torch.cuda.is_available() and args.device != "cpu" else "cpu"); model.to(device)
    def batch(texts):
        return tok(texts, truncation=True, max_length=args.max_length, padding=True, return_tensors="pt")
    target=torch.tensor([r["target_probs"] for r in train_rows], dtype=torch.float32)
    gold=torch.tensor([r["gold_label_id"] if r["gold_label_id"] is not None else -1 for r in train_rows], dtype=torch.long)
    if condition in ("majority_ce", "label_smoothing") and (gold < 0).any(): raise ValueError("Hard-label baselines require non-null gold labels")
    rng=np.random.RandomState(seed); started=time.time()
    for _epoch in range(args.epochs):
        model.train(); order=rng.permutation(len(train_rows))
        for start in range(0,len(train_rows),args.batch_size):
            idx=order[start:start+args.batch_size]
            b=batch([train_rows[int(i)]["text"] for i in idx]); b={k:v.to(device) for k,v in b.items()}
            logits=model(b["input_ids"],b["attention_mask"]); ti=torch.tensor(idx, dtype=torch.long)
            if condition.startswith("human_soft_targets"): loss=soft_target_cross_entropy(logits,target[ti].to(device))
            elif condition=="majority_ce": loss=hard_label_cross_entropy(logits,gold[ti].to(device))
            else: loss=hard_label_cross_entropy(logits,gold[ti].to(device),label_smoothing=args.label_smoothing)
            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
    model.eval(); chunks=[]
    with torch.no_grad():
        for start in range(0,len(eval_rows),args.batch_size):
            b=batch([r["text"] for r in eval_rows[start:start+args.batch_size]]); b={k:v.to(device) for k,v in b.items()}
            chunks.append(torch.softmax(model(b["input_ids"],b["attention_mask"]),dim=-1).cpu().numpy())
    return np.concatenate(chunks,axis=0), time.time()-started, {"checkpoint":str(checkpoint),"device":str(device),"max_length":args.max_length}


def main():
    p=argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default=str(ROOT/"data"/"dynasent-main.zip")); p.add_argument("--round", choices=["round1","round2"], default="round1")
    p.add_argument("--backend", choices=["tfidf","tiny-bert"], default="tfidf"); p.add_argument("--checkpoint", default=str(ROOT/"results"/"models"/"mlm_final"))
    p.add_argument("--budget", type=int, default=100); p.add_argument("--eval-size", type=int, default=500, help="0 means full test split")
    p.add_argument("--seeds", type=int, nargs="+", default=[0]); p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=32); p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--weight-decay", type=float, default=0.001); p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--max-features", type=int, default=12000); p.add_argument("--max-length", type=int, default=64)
    p.add_argument("--device", choices=["auto","cpu","cuda"], default="auto"); p.add_argument("--include-null-soft-only", action="store_true")
    p.add_argument("--out-dir", default=str(ROOT/"results"/"dynasent_pilot")); args=p.parse_args()
    if args.budget < 1 or args.epochs < 1: p.error("--budget and --epochs must be positive")
    data=Path(args.data)
    if not data.exists(): p.error(f"DynaSent archive/path not found: {data}")
    splits=load_dynasent_round(data,args.round)
    train_eligible=[r for r in splits.train if r["gold_label_id"] is not None]
    if len(train_eligible)<args.budget: p.error(f"Requested budget {args.budget}, only {len(train_eligible)} train rows have majority labels")
    eval_rows=sample_rows(splits.test,args.eval_size,seed=20261014)
    out=Path(args.out_dir); out.mkdir(parents=True,exist_ok=True)
    run_rows=[]; prediction_rows=[]; audit={"round":args.round,"data":str(data),"backend":args.backend,"labels":list(LABELS),"loader_audit":splits.audit,"eligible_train_rows_with_gold":len(train_eligible),"test_rows_used":len(eval_rows),"eval_sample_seed":20261014,"budget":args.budget,"seeds":args.seeds,"conditions":["majority_ce","human_soft_targets","label_smoothing"],"matching_policy":"all three matched conditions use identical examples sampled from non-null gold-label training pool"}
    for seed in args.seeds:
        rng=np.random.RandomState(seed); ix=rng.choice(len(train_eligible),size=args.budget,replace=False)
        matched=[train_eligible[int(i)] for i in ix]
        conditions=[("majority_ce",matched),("human_soft_targets",matched),("label_smoothing",matched)]
        if args.include_null_soft_only:
            soft_all=[r for r in splits.train if r["num_votes"]>0]
            extra_n=min(args.budget,len(soft_all)); ix2=np.random.RandomState(seed+10000).choice(len(soft_all),size=extra_n,replace=False)
            conditions.append(("human_soft_targets_all_votes_including_null_gold",[soft_all[int(i)] for i in ix2]))
        for condition, rows in conditions:
            print(f"[run] round={args.round} backend={args.backend} seed={seed} condition={condition} n_train={len(rows)} n_eval={len(eval_rows)}",flush=True)
            fn=fit_tfidf if args.backend=="tfidf" else fit_tiny_bert
            probs, elapsed, extra=fn(rows,eval_rows,condition,seed,args)
            metrics=evaluate(probs,eval_rows)
            rec={"round":args.round,"backend":args.backend,"seed":seed,"condition":condition,"budget":len(rows),"epochs":args.epochs,"train_time_s":elapsed,**metrics,**extra}
            run_rows.append(rec); print(json.dumps(rec,sort_keys=True),flush=True)
            for row, prob in zip(eval_rows,probs):
                prediction_rows.append({"round":args.round,"backend":args.backend,"seed":seed,"condition":condition,"text_id":row.get("text_id"),"text":row["text"],"gold_label":row["gold_label"],"human_distribution":row["target_probs"],"predicted_distribution":prob.tolist(),"predicted_label":LABELS[int(prob.argmax())],"js_divergence_bits":float(jensen_shannon_divergence(torch.tensor(prob[None,:]),torch.tensor(np.asarray(row["target_probs"])[None,:])).item())})
    (out/"config_and_audit.json").write_text(json.dumps({"args":vars(args),"audit":audit},indent=2),encoding="utf-8")
    (out/"metrics.json").write_text(json.dumps(run_rows,indent=2),encoding="utf-8")
    if run_rows:
        with (out/"metrics.csv").open("w",newline="",encoding="utf-8") as f:
            w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in run_rows for k in r))); w.writeheader(); w.writerows(run_rows)
    with (out/"predictions.jsonl").open("w",encoding="utf-8") as f:
        for r in prediction_rows: f.write(json.dumps(r,ensure_ascii=False)+"\n")
    summary=["# DynaSent pilot results","",f"- Round: `{args.round}`",f"- Backend: `{args.backend}`",f"- Budget: `{args.budget}`",f"- Evaluation rows: `{len(eval_rows)}`",f"- Seeds: `{args.seeds}`","","> Pilot results are exploratory, not final research claims. The test subset is fixed by seed 20261014; use the full test split and multiple seeds for final evaluation.","","## Metrics","","| Condition | Seed | Train seconds | JS divergence ↓ | Soft Brier ↓ | Accuracy | Macro-F1 | ECE ↓ |","|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in run_rows: summary.append(f"| {r['condition']} | {r['seed']} | {r['train_time_s']:.2f} | {r['js_divergence_mean_bits']:.4f} | {r['soft_brier_score']:.4f} | {r['accuracy'] if r['accuracy'] is not None else 'NA'} | {r['macro_f1'] if r['macro_f1'] is not None else 'NA'} | {r['ece_10bin'] if r['ece_10bin'] is not None else 'NA'} |")
    (out/"summary.md").write_text("\n".join(summary)+"\n",encoding="utf-8")
    print(f"\n[done] wrote {len(run_rows)} metric rows and {len(prediction_rows)} prediction rows to {out}",flush=True)

if __name__=="__main__": main()
