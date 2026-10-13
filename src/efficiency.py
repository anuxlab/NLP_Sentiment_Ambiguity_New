"""Efficiency study: parameters, size, latency, throughput and INT8 dynamic quantization.

Requires run_experiments.py --stage curve to have saved results/models/ft_tiny_*_full_seed0.pt
    python efficiency.py
"""
import io, json, time
import numpy as np
import torch, torch.nn as nn
import pickle

from baselines import fit_tfidf, proba
from config import MODELS, RES
from exp_utils import Bundle, evaluate
from finetune import build_model, encode, collate, load_tokenizer, predict_proba

torch.set_num_threads(1)


def n_params(model):
    tot = sum(p.numel() for p in model.parameters())
    emb = sum(p.numel() for n, p in model.named_parameters() if "embeddings" in n)
    return tot, tot - emb


def try_quantize(model):
    """INT8 dynamic quantization. Needs a quantized engine (fbgemm/x86 on Intel/AMD, qnnpack on some ARM builds).
    Returns (quantized_model, engine) or (None, None) if this PyTorch build has no usable engine
    (e.g. some Apple-Silicon wheels raise 'NoQEngine')."""
    import warnings
    warnings.filterwarnings("ignore")
    for eng in ("fbgemm", "x86", "onednn", "qnnpack"):
        if eng not in torch.backends.quantized.supported_engines:
            continue
        try:
            torch.backends.quantized.engine = eng
            return torch.ao.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8), eng
        except Exception as e:                      # engine listed but unusable
            print(f"[quant] engine {eng} failed: {str(e)[:80]}")
    return None, None


def size_mb(obj):
    buf = io.BytesIO(); torch.save(obj, buf); return buf.tell() / 1e6


def latency_bs1(pfn_single, texts, warm=20):
    for t in texts[:warm]:
        pfn_single(t)
    ts = []
    for t in texts[warm:]:
        t0 = time.perf_counter(); pfn_single(t); ts.append((time.perf_counter() - t0) * 1e3)
    return float(np.mean(ts)), float(np.percentile(ts, 95))


def throughput(pfn, texts):
    t0 = time.perf_counter(); pfn(texts); return len(texts) / (time.perf_counter() - t0)


def measure_neural(name, model, tok, B, quant=False):
    model.eval()
    texts = B.sets["test"][0]
    lat_mean, lat_p95 = latency_bs1(lambda t: predict_proba(model, tok, [t], bs=1), texts[:220])
    ev, _ = evaluate(lambda t: predict_proba(model, tok, t), B)
    tot, non_emb = n_params(model) if not quant else (None, None)
    return dict(model=name, params_total=tot, params_non_embedding=non_emb, size_mb=size_mb(model.state_dict()),
                latency_ms_bs1=lat_mean, latency_ms_p95=lat_p95,
                throughput_sent_per_s=throughput(lambda t: predict_proba(model, tok, t, bs=64), texts),
                **ev)


def main():
    B = Bundle(); out = []
    tok = load_tokenizer(str(MODELS / "mlm_step0"))
    for name in ("tiny_scratch", "tiny_mlm"):
        m = build_model("scratch", str(MODELS / "mlm_step0"), 0)
        m.load_state_dict(torch.load(MODELS / f"ft_{name}_full_seed0.pt")); m.eval()
        r = measure_neural(f"{name}_fp32", m, tok, B); out.append(r)
        print(f"{r['model']:20s} acc={r['test_acc']:.4f} size={r['size_mb']:.2f}MB lat={r['latency_ms_bs1']:.2f}ms "
              f"thr={r['throughput_sent_per_s']:.0f}/s", flush=True)
        if name == "tiny_mlm":
            q, eng = try_quantize(m)
            if q is None:
                print("[quant] no quantized engine available on this machine -> INT8 row skipped "
                      f"(supported: {torch.backends.quantized.supported_engines})", flush=True)
                continue
            rq = measure_neural("tiny_mlm_int8", q, tok, B, quant=True)
            rq["params_total"], rq["params_non_embedding"] = r["params_total"], r["params_non_embedding"]
            rq["quant_engine"] = eng
            out.append(rq)
            print(f"{rq['model']:20s} acc={rq['test_acc']:.4f} size={rq['size_mb']:.2f}MB lat={rq['latency_ms_bs1']:.2f}ms "
                  f"thr={rq['throughput_sent_per_s']:.0f}/s  (engine={eng})", flush=True)
    # TF-IDF + LR on the full training set
    pipe, info = fit_tfidf("tfidf_lr", B.tr_t, B.tr_y, B.dev_t, B.dev_y)
    texts = B.sets["test"][0]
    lm, lp = latency_bs1(lambda t: proba(pipe, [t]), texts[:220])
    ev, _ = evaluate(lambda t: proba(pipe, t), B)
    voc = len(pipe[0].vocabulary_)
    out.append(dict(model="tfidf_lr", params_total=voc + 1, params_non_embedding=voc + 1,
                    size_mb=len(pickle.dumps(pipe)) / 1e6, latency_ms_bs1=lm, latency_ms_p95=lp,
                    throughput_sent_per_s=throughput(lambda t: proba(pipe, t), texts), **ev))
    r = out[-1]; print(f"tfidf_lr             acc={r['test_acc']:.4f} size={r['size_mb']:.2f}MB lat={lm:.2f}ms", flush=True)
    json.dump(out, open(RES / "efficiency.json", "w"), indent=1)


if __name__ == "__main__":
    main()
