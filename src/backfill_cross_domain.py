"""One-off: backfill cross_finance_acc/cross_airline_acc for the SST-2 full-data (n=6920) models that were
trained before the multi-domain code existed. Regenerates only tfidf_lr/tiny_scratch/tiny_mlm at n=6920,
all 3 seeds (9 runs total, a few minutes) -- NOT a full SST-2 rerun. Safe to run once; resumable like the
other stages (skips rows that already have cross_ fields).
    python backfill_cross_domain.py
"""
import json
from config import MODELS, TRAIN_SIZES
from exp_utils import Bundle, done_keys, log, run_neural, run_tfidf
from run_experiments import best_lrs

SCRATCH_SRC = str(MODELS / "mlm_step0")
MLM_SRC = str(MODELS / "mlm_final")
N = TRAIN_SIZES[-1]  # 6920

if __name__ == "__main__":
    B = Bundle(domain="sst2")
    lrs = best_lrs("sst2")  # the actual pilot-selected rates (both were 2e-4 in the original run)
    done = {(r["stage"], r["model"], r["n"], r["seed"], r.get("tag", ""), r.get("domain", "sst2")) for r in
            (json.loads(l) for l in open("../results/results.jsonl"))
            if "cross_finance_acc" in r}   # only skip rows that ALREADY have the new fields
    for seed in (0, 1, 2):
        if ("curve", "tfidf_lr", N, seed, "", "sst2") not in done:
            r = run_tfidf(B, "tfidf_lr", N, seed); log(r)
            print(f"[backfill] tfidf_lr seed={seed} cross_finance={r['cross_finance_acc']:.4f} cross_airline={r['cross_airline_acc']:.4f}")
        for init, name, src in (("scratch", "tiny_scratch", SCRATCH_SRC), ("mlm", "tiny_mlm", MLM_SRC)):
            if ("curve", name, N, seed, "", "sst2") in done:
                continue
            save = MODELS / f"ft_{name}_full_seed0.pt" if seed == 0 else None
            try:
                r = run_neural(B, "curve", name, init, src, N, seed, lrs[name], save_model=save)
            except Exception as e:
                print(f"[backfill] SKIPPED {name} seed={seed}: {type(e).__name__}: {e}")
                continue
            log(r)
            print(f"[backfill] {name} seed={seed} cross_finance={r['cross_finance_acc']:.4f} cross_airline={r['cross_airline_acc']:.4f}")
    print("done.")
