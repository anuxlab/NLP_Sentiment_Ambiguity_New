"""Ambiguity-aware analysis of the saved test-set predictions (no retraining needed).

Three complementary ambiguity signals for the 1,821 SST-2 test sentences:
  A1  human-graded *polarity strength* from the fine-grained SST-5 labels: strong (very neg/very pos) vs weak (neg/pos,
      i.e. mild sentiment close to neutral).  This is a graded-intensity proxy for ambiguity, NOT annotator disagreement.
  A2  cross-model *consensus difficulty*: fraction of all model runs that classify the sentence correctly (used to check
      that A1 is not an arbitrary label: do hard-for-everyone sentences tend to be the weakly polar ones?).
  A3  simple *linguistic markers* of mixed / compositional sentiment: contrast words ("but", "although", ...) and negation.

Per model and label budget it reports accuracy on strong vs weak sentences, mean confidence on each, the
"ambiguity-awareness" AUROC (how well low confidence identifies weak-polarity sentences; 0.5 = chance), expected
calibration error (ECE), and accuracy on the 50 % most confident predictions.  Paired bootstrap intervals compare models
within each stratum.

    python ambiguity.py            # reads results/preds/*.npy, writes results/ambiguity.json
Public checkpoints are included automatically when results/preds/hf_<model>__<n>_<seed>.npy exist
(run hf_benchmark.py first, which stores predictions).
"""
import argparse, glob, json, re
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from config import DOMAIN_TRAIN_SIZES, PREDS, RES, SEEDS, TRAIN_SIZES
from data import load_airline, load_finance, load_sst2, load_sst5_for
from families import FAMILY_MODELS, PSEUDO_PROB

CONTRAST = {"but", "however", "although", "though", "yet", "despite", "whereas", "while"}
NEGATION = {"not", "n't", "no", "never", "nothing", "nobody", "neither", "nor", "without", "hardly", "barely"}
CORE = ["tfidf_lr", "tfidf_nb", "tiny_scratch", "tiny_mlm"] + FAMILY_MODELS


def ece(p, y, mask=None, bins=15):
    conf, pred = p.max(1), p.argmax(1)
    cor = (pred == y).astype(float)
    if mask is not None:
        conf, cor = conf[mask], cor[mask]
    edges = np.linspace(0.5, 1.0, bins + 1); tot = 0.0
    for a, b in zip(edges[:-1], edges[1:]):
        m = (conf > a) & (conf <= b)
        if m.any():
            tot += m.mean() * abs(cor[m].mean() - conf[m].mean())
    return tot


def acc_at_coverage(p, y, cov=0.5):
    conf = p.max(1); k = int(len(y) * cov); idx = np.argsort(-conf)[:k]
    return float((p.argmax(1)[idx] == y[idx]).mean())


def pred_file(model, n, seed, root, domain="sst2"):
    suf = "" if domain == "sst2" else f"{domain}_"           # matches exp_utils._pred_path / stage_families naming
    if model in ("tfidf_lr", "tfidf_nb", "tiny_scratch", "tiny_mlm"):
        return root / f"curve_{model}__{suf}{n}_{seed}.npy" if domain != "sst2" else root / f"curve_{model}__{n}_{seed}.npy"
    if model in FAMILY_MODELS:
        return root / f"family_{model}__{suf}{n}_{seed}.npy" if domain != "sst2" else root / f"family_{model}__{n}_{seed}.npy"
    return root / f"hf_{model}__{n}_{seed}.npy"                # public checkpoints: sst2-only for now


def strata_for_domain(domain):
    """Returns (test_labels, strong_mask, weak_mask, extra) with a REAL (finance/airline) or proxy (sst2)
    ambiguity signal. strong = clear-cut / high agreement / high confidence; weak = the opposite."""
    if domain == "sst2":
        te_t, y = load_sst2()["test"]
        lab5 = load_sst5_for(te_t)
        return y, (lab5 == 1) | (lab5 == 5), (lab5 == 2) | (lab5 == 4), dict(kind="sst5_polarity_strength (proxy)")
    if domain == "finance":
        splits, tier = load_finance(); y = splits["test"][1]; tier = np.array(tier)
        return y, tier == "100", tier == "50-65", dict(kind="real_annotator_agreement_tier", middle=int((tier == "66-99").sum()))
    if domain == "airline":
        splits, conf = load_airline(); y = splits["test"][1]; conf = np.array(conf)
        return y, conf >= 0.9, conf < 0.65, dict(kind="real_crowdworker_confidence", threshold_strong=0.9, threshold_weak=0.65)
    raise ValueError(domain)


def discover_hf(root):
    found = set()
    for f in glob.glob(str(root / "hf_*__*_*.npy")):
        m = re.match(r"^hf_(.+)__(\d+)_(\d+)\.npy$", Path(f).name)
        if m:
            found.add(m.group(1))
    return sorted(found)


def load_runs(model, n, root, domain="sst2"):
    out = []
    for s in SEEDS:
        f = pred_file(model, n, s, root, domain)
        if f.exists():
            out.append(np.load(f))
    return out


def model_metrics(runs, y, strong, weak, pseudo):
    m = strong | weak; rows = []
    for P in runs:
        cor = P.argmax(1) == y; conf = P.max(1)
        r = dict(acc=cor.mean(), acc_strong=cor[strong].mean(), acc_weak=cor[weak].mean(),
                 conf_strong=conf[strong].mean(), conf_weak=conf[weak].mean(),
                 auroc=roc_auc_score(weak[m], -conf[m]), acc50=acc_at_coverage(P, y))
        if not pseudo:
            r.update(ece=ece(P, y), ece_strong=ece(P, y, strong), ece_weak=ece(P, y, weak))
        rows.append(r)
    keys = rows[0].keys()
    return {k: [float(np.mean([r[k] for r in rows])), float(np.std([r[k] for r in rows]))] for k in keys}, len(rows)


def paired_bootstrap(runs_a, runs_b, y, mask, B=2000, seed=0):
    k = min(len(runs_a), len(runs_b))
    d = np.mean([((runs_a[i].argmax(1) == y).astype(float) - (runs_b[i].argmax(1) == y).astype(float))[mask] for i in range(k)], 0)
    rng = np.random.RandomState(seed)
    boots = np.array([d[rng.randint(0, len(d), len(d))].mean() for _ in range(B)])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return [float(d.mean()), float(lo), float(hi)]


def interaction_bootstrap(runs_a, runs_b, y, mask1, mask2, B=2000, seed=1):
    """Bootstrap of (A-B accuracy difference on mask1) - (A-B difference on mask2); strata resampled independently."""
    k = min(len(runs_a), len(runs_b))
    d = np.mean([((runs_a[i].argmax(1) == y).astype(float) - (runs_b[i].argmax(1) == y).astype(float)) for i in range(k)], 0)
    d1, d2 = d[mask1], d[mask2]; rng = np.random.RandomState(seed)
    boots = np.array([d1[rng.randint(0, len(d1), len(d1))].mean() - d2[rng.randint(0, len(d2), len(d2))].mean() for _ in range(B)])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return [float(d1.mean() - d2.mean()), float(lo), float(hi)]


def main(root=None, domain="sst2", output_dir=None):
    root = Path(root) if root else PREDS
    y, strong, weak, sig_info = strata_for_domain(domain)
    if domain == "sst2":
        te_t, _ = load_sst2()["test"]
    elif domain == "finance":
        te_t = load_finance()[0]["test"][0]
    else:
        te_t = load_airline()[0]["test"][0]
    toks = [set(t.lower().split()) for t in te_t]
    contrast = np.array([bool(t & CONTRAST) for t in toks]); negation = np.array([bool(t & NEGATION) for t in toks])
    ood_key = "imdb" if domain == "sst2" else [d for d in ("sst2", "finance", "airline") if d != domain][0]
    sizes = DOMAIN_TRAIN_SIZES[domain]
    eval_ns = (1000, sizes[-1]) if domain == "sst2" else (sizes[len(sizes) // 2], sizes[-1])
    models = (CORE + discover_hf(root)) if domain == "sst2" else CORE   # public checkpoints: sst2-only for now
    out = dict(domain=domain, signal=sig_info,
               counts=dict(total=len(y), strong=int(strong.sum()), weak=int(weak.sum()),
                           contrast=int(contrast.sum()), negation=int(negation.sum())),
               strata={}, bootstrap={}, validity={}, markers={})
    for n in eval_ns:
        out["strata"][str(n)] = {}
        for mdl in models:
            runs = load_runs(mdl, n, root, domain)
            if runs:
                out["strata"][str(n)][mdl], k = model_metrics(runs, y, strong, weak, mdl in PSEUDO_PROB)
                out["strata"][str(n)][mdl]["n_runs"] = k
        out["bootstrap"][str(n)] = {}
        pairs = [("tiny_mlm", "tiny_scratch"), ("tiny_mlm", "tfidf_lr"), ("tiny_scratch", "tfidf_lr"),
                 ("tiny_scratch", "fasttext_avg_lr"), ("tiny_mlm", "fasttext_avg_lr")]
        for a, b in pairs:
            ra, rb = load_runs(a, n, root, domain), load_runs(b, n, root, domain)
            if ra and rb:
                out["bootstrap"][str(n)][f"{a}-{b}"] = dict(
                    strong=paired_bootstrap(ra, rb, y, strong), weak=paired_bootstrap(ra, rb, y, weak),
                    strong_minus_weak=interaction_bootstrap(ra, rb, y, strong, weak),
                    contrast=paired_bootstrap(ra, rb, y, contrast), no_contrast=paired_bootstrap(ra, rb, y, ~contrast))
        cors = [(P.argmax(1) == y).astype(float) for mdl in CORE for P in load_runs(mdl, n, root, domain)]
        if cors:
            d = np.mean(cors, 0); m = strong | weak
            grp = {"hard ($<$25\\% of runs correct)": d < 0.25, "middle": (d >= 0.25) & (d < 0.75), "easy ($\\geq$75\\% correct)": d >= 0.75}
            out["validity"][str(n)] = dict(
                n_runs=len(cors), base_weak_share=float(weak[m].mean()),
                groups={k: dict(count=int((v & m).sum()), weak_share=float(weak[v & m].mean()) if (v & m).any() else None) for k, v in grp.items()},
                spearman_difficulty_vs_strength=float(spearmanr(d[m], strong[m]).correlation))
        out["markers"][str(n)] = {}
        for mdl in ["tfidf_lr", "tiny_scratch", "tiny_mlm", "tfidf_char_lr", "fasttext_avg_lr"]:
            runs = load_runs(mdl, n, root, domain)
            if not runs:
                continue
            def a(mask): return float(np.mean([(P.argmax(1) == y)[mask].mean() for P in runs]))
            out["markers"][str(n)][mdl] = dict(contrast=a(contrast), no_contrast=a(~contrast), negation=a(negation), no_negation=a(~negation))
        m_all = strong | weak
        out["markers"][str(n)]["weak_share"] = dict(
            contrast=float(weak[contrast & m_all].mean()) if (contrast & m_all).any() else None,
            no_contrast=float(weak[~contrast & m_all].mean()) if (~contrast & m_all).any() else None,
            negation=float(weak[negation & m_all].mean()) if (negation & m_all).any() else None,
            no_negation=float(weak[~negation & m_all].mean()) if (~negation & m_all).any() else None)
    fname = "ambiguity.json" if domain == "sst2" else f"ambiguity_{domain}.json"
    out_dir = Path(output_dir) if output_dir else RES
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / fname).write_text(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds_dir", default=None)
    ap.add_argument("--domain", default="sst2", choices=["sst2", "finance", "airline"])
    a = ap.parse_args()
    r = main(a.preds_dir, a.domain)
    print(json.dumps(r["counts"]))
    for n, d in r["strata"].items():
        print(f"\nn={n}")
        for mdl, v in d.items():
            print(f"  {mdl:18s} strong {100*v['acc_strong'][0]:5.1f} weak {100*v['acc_weak'][0]:5.1f}  AUROC {v['auroc'][0]:.3f}"
                  + (f"  ECE {100*v['ece'][0]:4.1f}" if 'ece' in v else ""))
