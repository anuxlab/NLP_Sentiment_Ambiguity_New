"""Benchmark public pre-trained compact encoders from the HuggingFace Hub with the SAME protocol
(fine-tuning code, mean-pool head, evaluation sets) used for our own tiny BERTs.

Model aliases (see MODEL_ZOO): tiny, mini, small, medium, distilbert, minilm, tinybert, electra
Any other string is treated as a hub name or a local checkpoint directory.

    python hf_benchmark.py --list                                # show the model zoo
    python hf_benchmark.py --models mini,small,minilm            # default: sizes 1000,6920 x seeds 0,1,2
    python hf_benchmark.py --models ladder                       # presets: ladder, distilled, objectives, phase_a/b/c, all
    python hf_benchmark.py --models small --sizes 1000 --seeds 0 # quick check of one model
    python hf_benchmark.py --models mini --tune                  # pick lr on dev first (3 lrs, n=1000, seed 0)
    python hf_benchmark.py --models tiny,distilbert --rerun      # redo earlier runs (also stores predictions for ambiguity/calibration)
    python hf_benchmark.py --models /path/to/local_ckpt          # local HF-format checkpoint

Outputs (all resumable - finished (model,n,seed) runs are skipped):
    results/hf_results.jsonl      one line per run (same schema as results.jsonl, stage="hf"; pilots use stage="hf_pilot")
    results/hf_efficiency.jsonl   params / size / latency / throughput per model (single CPU thread)
    results/preds/hf_<model>__<n>_<seed>.npy   test-set probabilities (used for paired bootstrap in make_assets.py)
Needs internet access to huggingface.co the first time a hub model is used (weights are then cached).
"""
import argparse, io, json, time
import numpy as np
import torch

import exp_utils
from config import RES, SEEDS
from exp_utils import Bundle, run_neural
from finetune import build_model, load_tokenizer, predict_proba

MODEL_ZOO = {
    "tiny":       dict(name="google/bert_uncased_L-2_H-128_A-2", lr=3e-4, note="BERT-Tiny   (2 layers, H=128,  4.4M params)"),
    "mini":       dict(name="google/bert_uncased_L-4_H-256_A-4", lr=1e-4, note="BERT-Mini   (4 layers, H=256, 11.2M params)"),
    "small":      dict(name="google/bert_uncased_L-4_H-512_A-8", lr=1e-4, note="BERT-Small  (4 layers, H=512, 28.8M params)"),
    "medium":     dict(name="google/bert_uncased_L-8_H-512_A-8", lr=5e-5, note="BERT-Medium (8 layers, H=512, 41.4M params)"),
    "distilbert": dict(name="distilbert-base-uncased",           lr=5e-5, note="DistilBERT  (6 layers, H=768, 66.4M params)"),
    "minilm":     dict(name="microsoft/MiniLM-L12-H384-uncased", lr=5e-5, tokenizer="bert-base-uncased",
                       note="MiniLM      (12 layers, H=384, 33M params; uses the bert-base-uncased vocabulary)"),
    "tinybert":   dict(name="huawei-noah/TinyBERT_General_4L_312D", lr=1e-4, tokenizer="bert-base-uncased",
                       note="TinyBERT-4L (4 layers, H=312, 14.5M params; general distillation)"),
    "electra":    dict(name="google/electra-small-discriminator",  lr=1e-4, tokenizer="bert-base-uncased",
                       note="ELECTRA-small (12 layers, H=256, 14M params; replaced-token-detection pre-training)"),
}
# Presets: --models ladder  ==  --models tiny,mini,small,medium   (comma lists may mix presets and models)
GROUPS = {
    "ladder":     "tiny,mini,small,medium",          # same family / same recipe, 4.4M -> 41M parameters (size dose-response)
    "distilled":  "distilbert,minilm,tinybert",      # compressed from larger teachers
    "objectives": "tiny,electra,tinybert,distilbert",# different pre-training objectives / compression at comparable sizes
    "phase_a":    "mini,small",                      # completes the size ladder (cheapest, most informative)
    "phase_b":    "tinybert,electra",                # adds objective/distillation diversity
    "phase_c":    "medium,minilm",                   # expensive; consider --sizes 1000
    "all":        "tiny,mini,small,medium,distilbert,minilm,tinybert,electra",
}
DEFAULT_MODELS = "mini,small,minilm"


def expand(models_arg):
    """Expand presets (GROUPS) inside a comma-separated model list, keeping order and removing duplicates."""
    out = []
    for item in models_arg.split(","):
        for m in GROUPS.get(item.strip(), item.strip()).split(","):
            if m not in out:
                out.append(m)
    return out


def resolve(spec):
    """alias | hub name | local dir -> (short_name, source, default_lr, tokenizer_source_or_None)."""
    if spec in MODEL_ZOO:
        z = MODEL_ZOO[spec]
        return z["name"].rstrip("/").split("/")[-1], z["name"], z["lr"], z.get("tokenizer")
    for z in MODEL_ZOO.values():                       # allow the full hub name of a zoo model
        if z["name"] == spec:
            return spec.split("/")[-1], spec, z["lr"], z.get("tokenizer")
    return spec.rstrip("/").split("/")[-1], spec, 3e-4, None


def get_tokenizer(source, tok_override):
    try:
        return load_tokenizer(source), source
    except Exception as e:                             # e.g. checkpoints that ship without a tokenizer
        if tok_override is None:
            raise
        print(f"[hf] tokenizer of {source} unavailable ({type(e).__name__}); using {tok_override}")
        return load_tokenizer(tok_override), tok_override


@torch.no_grad()
def measure_efficiency(short, model, tok, B, n_lat=150, n_thr=512):
    """Params, fp32 size, batch-1 latency and batch-64 throughput on ONE CPU thread (weights do not affect speed)."""
    prev = torch.get_num_threads(); torch.set_num_threads(1); model.eval()
    texts = B.sets["test"][0]
    for t in texts[:10]:
        predict_proba(model, tok, [t], bs=1)
    ts = []
    for t in texts[10:10 + n_lat]:
        t0 = time.perf_counter(); predict_proba(model, tok, [t], bs=1); ts.append((time.perf_counter() - t0) * 1e3)
    t0 = time.perf_counter(); predict_proba(model, tok, texts[:n_thr], bs=64); thr = n_thr / (time.perf_counter() - t0)
    buf = io.BytesIO(); torch.save(model.state_dict(), buf)
    tot = sum(p.numel() for p in model.parameters())
    emb = sum(p.numel() for n, p in model.named_parameters() if "embeddings" in n)
    torch.set_num_threads(prev)
    return dict(model=short, params_total=tot, params_non_embedding=tot - emb, size_mb=buf.tell() / 1e6,
                latency_ms_bs1=float(np.mean(ts)), latency_ms_p95=float(np.percentile(ts, 95)),
                throughput_sent_per_s=float(thr), threads=1)


def read_jsonl(path):
    return [json.loads(l) for l in open(path)] if path.exists() else []


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default=None, help=f"comma-separated aliases / hub names / local dirs (default: {DEFAULT_MODELS})")
    ap.add_argument("--hf_names", default=None, help="deprecated alias of --models")
    ap.add_argument("--sizes", default="1000,6920")
    ap.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    ap.add_argument("--lr", type=float, default=None, help="override the learning rate of every model")
    ap.add_argument("--tune", action="store_true", help="select the lr per model on dev accuracy first (seed 0)")
    ap.add_argument("--tune_grid", default="3e-5,1e-4,3e-4")
    ap.add_argument("--tune_n", type=int, default=1000, help="training size used for the lr pilot")
    ap.add_argument("--no_efficiency", action="store_true")
    ap.add_argument("--rerun", action="store_true", help="discard earlier rows of the selected models and run them again (e.g. to store predictions)")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    if a.list:
        for k, z in MODEL_ZOO.items():
            print(f"{k:11s} {z['name']:46s} lr={z['lr']:<7g} {z['note']}")
        print("\npresets:", *[f"{k}={v}" for k, v in GROUPS.items()], sep="\n  ")
        return

    hf_file, eff_file = RES / "hf_results.jsonl", RES / "hf_efficiency.jsonl"
    exp_utils.RESULTS_FILE = hf_file
    B = Bundle()
    if a.rerun:                                        # drop earlier results of the selected models, then run them again
        shorts = {resolve(m.strip())[0] for m in expand(a.models or a.hf_names or DEFAULT_MODELS)}
        kept = [r for r in read_jsonl(hf_file) if r["model"] not in shorts]
        with open(hf_file, "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in kept)
        print(f"[hf] --rerun: removed earlier rows of {sorted(shorts)}; {len(kept)} rows kept", flush=True)
    done = {(r["stage"], r["model"], r["n"], r["seed"], r.get("tag", "")) for r in read_jsonl(hf_file)}
    eff_done = {r["model"] for r in read_jsonl(eff_file)}

    for spec in expand(a.models or a.hf_names or DEFAULT_MODELS):
        short, source, lr, tok_override = resolve(spec.strip())
        print(f"\n[hf] === {short}  ({source})  default lr={lr:g} ===", flush=True)
        tok, tok_src = get_tokenizer(source, tok_override)
        model = build_model("hf", source, 0)
        n_par = sum(p.numel() for p in model.parameters())
        if not a.no_efficiency and short not in eff_done:
            r = measure_efficiency(short, model, tok, B)
            with open(eff_file, "a") as f:
                f.write(json.dumps(r) + "\n")
            print(f"[hf] efficiency {short}: {r['params_total']/1e6:.1f}M params, {r['size_mb']:.1f} MB, "
                  f"{r['latency_ms_bs1']:.1f} ms/sentence, {r['throughput_sent_per_s']:.0f} sent/s", flush=True)
        del model
        if a.lr:
            lr = a.lr
        elif a.tune:
            best, best_lr = -1, lr
            for g in map(float, a.tune_grid.split(",")):
                key = ("hf_pilot", short, a.tune_n, 0, f"lr{g}")
                if key in done:
                    dev = next(r["dev_acc"] for r in read_jsonl(hf_file) if (r["stage"], r["model"], r["n"], r["seed"], r.get("tag", "")) == key)
                else:
                    r = run_neural(B, "hf_pilot", short, "hf", source, a.tune_n, 0, g, tag=f"lr{g}", tok_source=tok_src)
                    r["params_total"] = n_par; exp_utils.log(r); dev = r["dev_acc"]
                    print(f"[hf-pilot] {short:36s} lr={g:.0e} dev={dev:.4f}", flush=True)
                if dev > best:
                    best, best_lr = dev, g
            lr = best_lr
            print(f"[hf] selected lr for {short}: {lr:g}", flush=True)
        for n in map(int, a.sizes.split(",")):
            for seed in map(int, a.seeds.split(",")):
                if ("hf", short, n, seed, "") in done:
                    continue
                t0 = time.time()
                r = run_neural(B, "hf", short, "hf", source, n, seed, lr, tok_source=tok_src)
                r["params_total"] = n_par
                exp_utils.log(r)
                print(f"[hf] {short:36s} n={n:5d} seed={seed} lr={lr:g} test={r['test_acc']:.4f} imdb={r['imdb_acc']:.4f} "
                      f"typo30={r['typo30_acc']:.4f}  ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
