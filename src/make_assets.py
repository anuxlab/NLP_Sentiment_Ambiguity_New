"""Turn raw results into figures (PNG+PDF), LaTeX tables and a Markdown summary.

    python make_assets.py

Writes to results/figures/, report/figures/, report/tables/ and results/summary.md
"""
import json, shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import FIGS, PREDS, RES, ROOT, SEEDS, TRAIN_SIZES, DOMAIN_TRAIN_SIZES
from data import load_sst2
import argparse
from pathlib import Path

CORE = RES   # folder holding results.jsonl, efficiency.json, logs/pretrain.jsonl, preds/ (override with --results_dir)

REP_FIG = ROOT / "report" / "figures"; REP_TAB = ROOT / "report" / "tables"
REP_FIG.mkdir(parents=True, exist_ok=True); REP_TAB.mkdir(parents=True, exist_ok=True)

NAMES = {"tfidf_lr": "TF-IDF + LogReg", "tfidf_nb": "TF-IDF + NaiveBayes",
         "tiny_scratch": "Tiny-BERT (random init)", "tiny_mlm": "Tiny-BERT (+MLM pre-training)"}
COL = {"tfidf_lr": "#7f7f7f", "tfidf_nb": "#bcbd22", "tiny_scratch": "#d95f02", "tiny_mlm": "#1b9e77"}
MK = {"tfidf_lr": "s", "tfidf_nb": "^", "tiny_scratch": "o", "tiny_mlm": "D"}
PERT = ["test", "typo10", "typo30", "drop20", "drop40"]
PERT_LABEL = {"test": "clean", "typo10": "typos 10%", "typo30": "typos 30%",
              "drop20": "word-drop 20%", "drop40": "word-drop 40%", "imdb": "OOD IMDb"}
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": .25, "figure.dpi": 130})


def savefig(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(FIGS / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    shutil.copy(FIGS / f"{name}.pdf", REP_FIG / f"{name}.pdf")
    shutil.copy(FIGS / f"{name}.png", REP_FIG / f"{name}.png")
    plt.close(fig)


def ms(x, pct=True):
    x = np.asarray(x) * (100 if pct else 1)
    return f"{x.mean():.1f}$\\pm${x.std(ddof=0):.1f}"


def only_domain(df, domain):
    """Keep rows of one domain; rows written before the multi-domain code have no 'domain' field = sst2.
    domain=None returns everything."""
    if df is None or domain is None or len(df) == 0:
        return df
    col = df["domain"].fillna("sst2") if "domain" in df else pd.Series(["sst2"] * len(df), index=df.index)
    return df[col == domain].copy()


def load(domain="sst2"):
    """results.jsonl rows for ONE domain (default sst2, so all the original SST-2 tables/figures stay unchanged
    when finance/airline rows are present). load(None) returns every domain."""
    return only_domain(pd.DataFrame([json.loads(l) for l in open(CORE / "results.jsonl")]), domain)


def fig_pretrain():
    log = pd.DataFrame([json.loads(l) for l in open(CORE / "logs" / "pretrain.jsonl")])
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
    ax[0].plot(log.step, log.train_loss, label="train", c="#d95f02"); ax[0].plot(log.step, log.val_loss, label="held-out", c="#1b9e77")
    ax[0].set_xlabel("pre-training step"); ax[0].set_ylabel("MLM cross-entropy"); ax[0].legend(); ax[0].set_title("Masked-LM loss")
    ax[1].plot(log.step, 100 * log.val_acc, c="#1b9e77"); ax[1].set_xlabel("pre-training step")
    ax[1].set_ylabel("masked-token accuracy (%)"); ax[1].set_title("Held-out masked-token accuracy")
    fig.tight_layout(); savefig(fig, "fig_pretrain")
    return log


def fig_learning(df):
    c = df[df.stage == "curve"]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    for panel, metric, title in ((0, "test_acc", "SST-2 test (in-domain)"), (1, "imdb_acc", "IMDb held-out docs (OOD)")):
        for m in NAMES:
            g = c[c.model == m].groupby("n")[metric].agg(["mean", "std"]).reindex(TRAIN_SIZES)
            ax[panel].plot(g.index, 100 * g["mean"], marker=MK[m], c=COL[m], label=NAMES[m])
            ax[panel].fill_between(g.index, 100 * (g["mean"] - g["std"]), 100 * (g["mean"] + g["std"]), color=COL[m], alpha=.15)
        ax[panel].set_xscale("log"); ax[panel].set_xlabel("# labeled training sentences (log)")
        ax[panel].set_ylabel("accuracy (%)"); ax[panel].set_title(title)
    ax[0].legend(fontsize=8, loc="lower right"); fig.tight_layout(); savefig(fig, "fig_learning_curve")


def fig_robust(df):
    c = df[df.stage == "curve"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
    for ax, n in zip(axes, (1000, TRAIN_SIZES[-1])):
        keys = PERT + ["imdb"]; w = 0.2
        for i, m in enumerate(NAMES):
            g = c[(c.model == m) & (c.n == n)]
            mu = [100 * g[f"{k}_acc"].mean() for k in keys]; sd = [100 * g[f"{k}_acc"].std(ddof=0) for k in keys]
            ax.bar(np.arange(len(keys)) + (i - 1.5) * w, mu, w, yerr=sd, color=COL[m], label=NAMES[m], capsize=1.5)
        ax.set_xticks(range(len(keys))); ax.set_xticklabels([PERT_LABEL[k] for k in keys], rotation=25, ha="right")
        ax.set_ylim(45, 90); ax.set_title(f"n = {n} labeled sentences"); ax.set_ylabel("accuracy (%)")
    axes[0].legend(fontsize=7, loc="upper right"); fig.tight_layout(); savefig(fig, "fig_robustness")


def fig_ckpt(df):
    c = df[df.stage == "ckpt"].copy(); c["step"] = c.tag.str.replace("step", "").astype(int)
    g = c.groupby("step")[["test_acc", "probe_acc", "imdb_acc"]].agg(["mean", "std"])
    fig, ax = plt.subplots(figsize=(5.6, 3.6)); xs = np.maximum(g.index.values, 30)
    for col, lab, colr in (("test_acc", "fine-tuned (n=1000)", "#1b9e77"), ("probe_acc", "frozen encoder + linear probe", "#7570b3"),
                           ("imdb_acc", "fine-tuned, OOD IMDb", "#e6ab02")):
        ax.errorbar(xs, 100 * g[col]["mean"], 100 * g[col]["std"], marker="o", label=lab, c=colr, capsize=2)
    ax.set_xscale("log"); ax.set_xlabel("MLM pre-training steps (0 plotted at 30)"); ax.set_ylabel("accuracy (%)")
    ax.legend(fontsize=8); ax.set_title("Does more pre-training help downstream?"); fig.tight_layout(); savefig(fig, "fig_ckpt")


def fig_eff(eff, df):
    fig, ax = plt.subplots(figsize=(6, 3.8))
    lab = {"tfidf_lr": "TF-IDF+LR", "tiny_scratch_fp32": "Tiny scratch", "tiny_mlm_fp32": "Tiny +MLM", "tiny_mlm_int8": "Tiny +MLM INT8"}
    colr = {"tfidf_lr": "#7f7f7f", "tiny_scratch_fp32": "#d95f02", "tiny_mlm_fp32": "#1b9e77", "tiny_mlm_int8": "#1f78b4"}
    for r in eff:
        ax.scatter(r["latency_ms_bs1"], 100 * r["test_acc"], s=40 + 60 * r["size_mb"], c=colr[r["model"]], alpha=.8, edgecolor="k")
        dx, dy, ha = {"tfidf_lr": (10, 4, "left"), "tiny_scratch_fp32": (12, 2, "left"), "tiny_mlm_fp32": (-12, 2, "right"), "tiny_mlm_int8": (12, 2, "left")}[r["model"]]
        ax.annotate(lab[r["model"]], (r["latency_ms_bs1"], 100 * r["test_acc"]), textcoords="offset points", xytext=(dx, dy), ha=ha, va="center", fontsize=8)
    ax.set_xscale("log"); ax.set_xlim(0.2, 6); ax.set_xlabel("CPU latency, batch=1 (ms, log)"); ax.set_ylabel("SST-2 test accuracy (%)")
    ax.set_title("Accuracy vs latency (marker area ~ model size)"); fig.tight_layout(); savefig(fig, "fig_efficiency")


# ------------------------------------------------------------------ tables
def tex_table(path, header, rows, colfmt, caption, label):
    esc = lambda x: str(x).replace("\\%", "%").replace("%", "\\%")
    header = [esc(h) for h in header]; rows = [[esc(c) for c in r] for r in rows]
    body = "\n".join(" & ".join(map(str, r)) + r" \\" for r in rows)
    (REP_TAB / path).write_text(
        "\\begin{table}[t]\n\\centering\\small\n\\caption{" + caption + "}\\label{" + label + "}\n"
        "\\resizebox{\\linewidth}{!}{\\begin{tabular}{" + colfmt + "}\n\\toprule\n" + " & ".join(header) + " \\\\\n\\midrule\n" + body +
        "\n\\bottomrule\n\\end{tabular}}\n\\end{table}\n")


def tables(df, eff):
    c = df[df.stage == "curve"]; md = []
    # main learning-curve table
    rows = []
    for m in NAMES:
        rows.append([NAMES[m]] + [ms(c[(c.model == m) & (c.n == n)].test_acc) for n in TRAIN_SIZES])
    tex_table("tab_main.tex", ["Model"] + [f"$n{{=}}{n}$" for n in TRAIN_SIZES], rows, "l" + "c" * len(TRAIN_SIZES),
              "SST-2 test accuracy (\\%), mean$\\pm$std over 3 seeds, as a function of labeled training size $n$.", "tab:main")
    md.append("## SST-2 test accuracy (%)\n" + pd.DataFrame(rows, columns=["model"] + [str(n) for n in TRAIN_SIZES]).to_markdown(index=False))
    # robustness tables (n=1000 and full)
    for n, nm in ((1000, "tab_robust_1000.tex"), (TRAIN_SIZES[-1], "tab_robust_full.tex")):
        rows = []
        for m in NAMES:
            g = c[(c.model == m) & (c.n == n)]
            rows.append([NAMES[m]] + [ms(g[f"{k}_acc"]) for k in PERT + ["imdb"]])
        tex_table(nm, ["Model"] + [PERT_LABEL[k] for k in PERT + ["imdb"]], rows, "l" + "c" * 6,
                  f"Accuracy (\\%) under input perturbations and on out-of-domain IMDb documents, models trained on $n={n}$ sentences.",
                  f"tab:robust{n}")
        md.append(f"## Robustness, n={n}\n" + pd.DataFrame(rows, columns=["model"] + [PERT_LABEL[k] for k in PERT + ['imdb']]).to_markdown(index=False))
    # relative degradation at full data
    rows = []
    for m in NAMES:
        g = c[(c.model == m) & (c.n == TRAIN_SIZES[-1])]
        base = g.test_acc.mean()
        rows.append([NAMES[m]] + [f"{100*(g[f'{k}_acc'].mean()-base):+.1f}" for k in PERT[1:]])
    tex_table("tab_drop.tex", ["Model"] + [PERT_LABEL[k] for k in PERT[1:]], rows, "lcccc",
              "Change in accuracy (percentage points) relative to clean test, full training set.", "tab:drop")
    md.append("## Accuracy change vs clean (pp), full data\n" + pd.DataFrame(rows, columns=["model"] + [PERT_LABEL[k] for k in PERT[1:]]).to_markdown(index=False))
    # LR pilot
    p = df[df.stage == "pilot"]
    rows = []
    for m in ("tiny_scratch", "tiny_mlm"):
        for n in (300, 3000):
            g = p[(p.model == m) & (p.n == n)].set_index("lr").dev_acc
            rows.append([NAMES[m], n] + [f"{100*g[l]:.1f}" for l in sorted(g.index)])
    lrs = sorted(p.lr.unique())
    tex_table("tab_pilot.tex", ["Model", "$n$"] + [f"{l:.0e}".replace("e-0", "e-") for l in lrs], rows, "lc" + "c" * len(lrs),
              "Learning-rate pilot: dev accuracy (\\%) (seed 0). The rate with the best mean over the two sizes is used in all later experiments.", "tab:pilot")
    md.append("## LR pilot (dev acc %)\n" + pd.DataFrame(rows, columns=["model", "n"] + [str(l) for l in lrs]).to_markdown(index=False))
    # ckpt table
    k = df[df.stage == "ckpt"].copy(); k["step"] = k.tag.str.replace("step", "").astype(int)
    rows = [[s, ms(g.probe_acc), ms(g.test_acc), ms(g.imdb_acc)] for s, g in k.groupby("step")]
    tex_table("tab_ckpt.tex", ["MLM steps", "Linear probe", "Fine-tuned", "Fine-tuned OOD IMDb"], rows, "rccc",
              "Effect of the amount of pre-training ($n{=}1000$ labeled sentences, 3 seeds, test accuracy \\%).", "tab:ckpt")
    md.append("## Pre-training steps ablation\n" + pd.DataFrame(rows, columns=["steps", "probe", "finetune", "imdb"]).to_markdown(index=False))
    # efficiency
    if eff:
        lab = {"tfidf_lr": "TF-IDF + LogReg", "tiny_scratch_fp32": "Tiny-BERT scratch, FP32", "tiny_mlm_fp32": "Tiny-BERT +MLM, FP32", "tiny_mlm_int8": "Tiny-BERT +MLM, INT8 dyn."}
        rows = [[lab[r["model"]], f"{r['params_total']/1e6:.2f}", f"{r['size_mb']:.2f}", f"{r['latency_ms_bs1']:.2f}",
                 f"{r['latency_ms_p95']:.2f}", f"{r['throughput_sent_per_s']:.0f}", f"{100*r['test_acc']:.1f}", f"{100*r['typo30_acc']:.1f}"] for r in eff]
        tex_table("tab_eff.tex", ["Model", "Params (M)", "Size (MB)", "Lat. mean (ms)", "Lat. p95 (ms)", "Sent./s (bs 64)", "Acc.", "Acc. typo30"],
                  rows, "lccccccc", "Efficiency on a single CPU thread (seed-0 models trained on the full training set).", "tab:eff")
        md.append("## Efficiency\n" + pd.DataFrame(rows, columns=["model", "params M", "size MB", "lat ms", "p95 ms", "sent/s", "acc", "acc typo30"]).to_markdown(index=False))
    return md


def hf_table():
    """Optional: table for public HF checkpoints run with hf_benchmark.py (results/hf_results.jsonl)."""
    f = RES / "hf_results.jsonl"
    if not f.exists():
        return ""
    h = pd.DataFrame([json.loads(l) for l in open(f)]); h = h[h.stage == "hf"]
    rows, md = [], []
    for (m, n), g in h.groupby(["model", "n"]):
        rows.append([m.replace("_", "\\_"), f"{g.params_total.iloc[0]/1e6:.1f}", n, ms(g.test_acc), ms(g.typo30_acc), ms(g.drop20_acc), ms(g.imdb_acc)])
        md.append([m, f"{g.params_total.iloc[0]/1e6:.1f}", n, ms(g.test_acc), ms(g.typo30_acc), ms(g.drop20_acc), ms(g.imdb_acc)])
    tex_table("tab_hf.tex", ["Checkpoint", "Params (M)", "$n$", "Clean", "typos 30%", "word-drop 20%", "OOD IMDb"], rows, "lcccccc",
              "Public pre-trained compact checkpoints fine-tuned with the same protocol (accuracy \\%, mean$\\pm$std over the seeds run).", "tab:hf")
    return "## Public HF checkpoints\n" + pd.DataFrame(md, columns=["model", "params M", "n", "clean", "typo30", "drop20", "imdb"]).to_markdown(index=False)


def fig_hf(df, eff):
    f = RES / "hf_results.jsonl"
    if not f.exists():
        return
    h = pd.DataFrame([json.loads(l) for l in open(f)]); h = h[h.stage == "hf"]; c = df[df.stage == "curve"]
    tiny_p = next((r["params_total"] for r in (eff or []) if r["model"] == "tiny_mlm_fp32"), 1.19e6)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=False)
    for ax, n in zip(axes, sorted(h.n.unique())[:2]):
        for m, lab, colr in (("tiny_scratch", "Tiny-BERT random init (ours)", COL["tiny_scratch"]), ("tiny_mlm", "Tiny-BERT +MLM (ours)", COL["tiny_mlm"])):
            g = c[(c.model == m) & (c.n == n)].test_acc
            ax.errorbar(tiny_p / 1e6, 100 * g.mean(), 100 * g.std(ddof=0), marker="o", c=colr, capsize=3, label=lab)
        for name, g in h[h.n == n].groupby("model"):
            ax.errorbar(g.params_total.iloc[0] / 1e6, 100 * g.test_acc.mean(), 100 * g.test_acc.std(ddof=0), marker="*", ms=12, c="#1f78b4", capsize=3)
            ax.annotate(name.replace("bert_uncased_", "").replace("-base-uncased", ""), (g.params_total.iloc[0] / 1e6, 100 * g.test_acc.mean()),
                        textcoords="offset points", xytext=(-4, -14), fontsize=8, ha="right")
        t = c[(c.model == "tfidf_lr") & (c.n == n)].test_acc.mean() * 100
        ax.axhline(t, ls="--", c="#7f7f7f", label="TF-IDF + LogReg"); ax.set_xscale("log")
        ax.set_xlabel("parameters (millions, log)"); ax.set_ylabel("SST-2 test accuracy (%)"); ax.set_title(f"n = {n} labeled sentences")
    axes[0].legend(fontsize=7, loc="lower right"); fig.tight_layout(); savefig(fig, "fig_hf")


def hf_eff_table(eff):
    """Efficiency of public checkpoints (results/hf_efficiency.jsonl) next to our models (single CPU thread)."""
    f = RES / "hf_efficiency.jsonl"
    if not f.exists():
        return ""
    rows, md = [], []
    for r in [json.loads(l) for l in open(f)]:
        rows.append([r["model"].replace("_", "\\_"), f"{r['params_total']/1e6:.1f}", f"{r['size_mb']:.1f}", f"{r['latency_ms_bs1']:.1f}", f"{r['throughput_sent_per_s']:.0f}"])
        md.append([r["model"], f"{r['params_total']/1e6:.1f}", f"{r['size_mb']:.1f}", f"{r['latency_ms_bs1']:.1f}", f"{r['throughput_sent_per_s']:.0f}"])
    for r in (eff or []):
        if r["model"] in ("tiny_mlm_fp32", "tfidf_lr"):
            nm = "Tiny-BERT (ours)" if r["model"] == "tiny_mlm_fp32" else "TF-IDF + LogReg"
            row = [nm, f"{r['params_total']/1e6:.2f}", f"{r['size_mb']:.1f}", f"{r['latency_ms_bs1']:.2f}", f"{r['throughput_sent_per_s']:.0f}"]
            rows.append(row); md.append(row)
    tex_table("tab_hf_eff.tex", ["Model", "Params (M)", "Size (MB)", "Latency (ms)", "Sent./s (bs 64)"], rows, "lcccc",
              "Cost of public checkpoints vs our models (single CPU thread, FP32).", "tab:hf_eff")
    return "## Efficiency of public checkpoints\n" + pd.DataFrame(md, columns=["model", "params M", "size MB", "lat ms", "sent/s"]).to_markdown(index=False)


def hf_bootstrap(B=2000):
    """Paired bootstrap: public checkpoint minus TF-IDF+LogReg (needs results/preds/hf_*.npy from hf_benchmark.py)."""
    f = RES / "hf_results.jsonl"
    if not f.exists():
        return ""
    h = pd.DataFrame([json.loads(l) for l in open(f)]); h = h[h.stage == "hf"]
    y = load_sst2()["test"][1]; rng = np.random.RandomState(0); rows, md = [], []
    for (m, n), g in h.groupby(["model", "n"]):
        try:
            a = np.stack([(np.load(PREDS / f"hf_{m}__{n}_{s}.npy").argmax(1) == y) for s in g.seed]).astype(float)
            b = np.stack([(np.load(CORE / "preds" / f"curve_tfidf_lr__{n}_{s}.npy").argmax(1) == y) for s in g.seed]).astype(float)
        except FileNotFoundError:
            continue
        d = (a - b).mean(0)
        boots = np.array([d[rng.randint(0, len(d), len(d))].mean() for _ in range(B)]); lo, hi = np.percentile(boots, [2.5, 97.5])
        cell = f"{100*d.mean():+.1f} [{100*lo:+.1f}, {100*hi:+.1f}]" + ("$^*$" if lo > 0 or hi < 0 else "")
        rows.append([m.replace("_", "\\_"), n, cell]); md.append([m, n, cell])
    if not rows:
        return ""
    tex_table("tab_hf_stats.tex", ["Checkpoint", "$n$", "Accuracy diff. vs TF-IDF+LR (pp) [95\\% CI]"], rows, "lcc",
              "Paired bootstrap of public checkpoints against TF-IDF + LogReg (test sentences resampled 2000 times, pooled over seeds). $^*$ = CI excludes 0.", "tab:hf_stats")
    return "## Public checkpoints vs TF-IDF+LR (paired bootstrap)\n" + pd.DataFrame(md, columns=["model", "n", "diff pp [CI]"]).to_markdown(index=False)


# ------------------------------------------------------------------ model families + ambiguity (added)
FAM_NAMES = {"vader_lexicon": "VADER lexicon (no task labels)", "tfidf_svm": "TF-IDF word + SVM", "tfidf_char_lr": "TF-IDF char n-gram + LogReg",
             "w2v_avg_lr": "word2vec (same 1.2M words) + LogReg", "fasttext_avg_lr": "fastText (same 1.2M words) + LogReg"}
ALL_NAMES = {**NAMES, **FAM_NAMES}
FAM_COL = {"vader_lexicon": "#8c564b", "tfidf_svm": "#17becf", "tfidf_char_lr": "#9467bd", "w2v_avg_lr": "#e377c2", "fasttext_avg_lr": "#bcbd22"}
ALL_COL = {**COL, **FAM_COL}
ALL_COL["tfidf_nb"] = "#c7c7c7"


def load_families(domain="sst2"):
    f = CORE / "families.jsonl"
    return only_domain(pd.DataFrame([json.loads(l) for l in open(f)]), domain) if f.exists() else None


def combined(df, fam):
    c = df[df.stage == "curve"]
    return pd.concat([c, fam], ignore_index=True) if fam is not None else c


def equiv_labels(allr, acc):
    """TF-IDF+LogReg labels needed to reach accuracy `acc` (log-linear interpolation of the seed-mean curve).
    Returns "N/A" if `acc` is NaN or the TF-IDF reference curve itself has missing points, instead of crashing."""
    if acc != acc:  # NaN check (NaN != NaN is True), avoids importing math/np.isnan for a one-liner
        return "N/A"
    ns = TRAIN_SIZES
    tf = [100 * allr[(allr.model == "tfidf_lr") & (allr.n == n)].test_acc.mean() for n in ns]
    if any(t != t for t in tf):
        return "N/A"
    if acc < tf[0]:
        return f"<{ns[0]}"
    if acc > tf[-1]:
        return f"$>${ns[-1]:,}"
    return f"{int(round(float(10 ** np.interp(acc, tf, np.log10(ns))), -1)):,}"


def fig_families(df, fam):
    if fam is None:
        return
    allr = combined(df, fam)
    hfp = RES / "hf_results.jsonl"
    h = pd.DataFrame([json.loads(l) for l in open(hfp)]) if hfp.exists() else None
    if h is not None:
        h = h[h.stage == "hf"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={"width_ratios": [1.15, 1]})
    for m in ["tfidf_lr", "tfidf_char_lr", "fasttext_avg_lr", "w2v_avg_lr", "tiny_scratch", "tiny_mlm"]:
        g = allr[allr.model == m].groupby("n").test_acc.agg(["mean", "std"]).reindex(TRAIN_SIZES)
        ax[0].plot(g.index, 100 * g["mean"], marker=MK.get(m, "o"), c=ALL_COL[m], label=ALL_NAMES[m])
    v = 100 * allr[allr.model == "vader_lexicon"].test_acc.mean()
    ax[0].axhline(v, ls="--", c=FAM_COL["vader_lexicon"], label=FAM_NAMES["vader_lexicon"])
    if h is not None and len(h):
        for name, g in h.groupby("model"):
            for n, gg in g.groupby("n"):
                ax[0].scatter(n, 100 * gg.test_acc.mean(), marker="*", s=110, c="#1f78b4", zorder=5)
        ax[0].scatter([], [], marker="*", s=110, c="#1f78b4", label="public checkpoints")
    ax[0].set_xscale("log"); ax[0].set_xlabel("# labeled training sentences (log)"); ax[0].set_ylabel("SST-2 test accuracy (%)")
    ax[0].set_title("Learning curves by model family")
    ms_ = ["vader_lexicon", "tfidf_lr", "tfidf_char_lr", "fasttext_avg_lr", "tiny_scratch", "tiny_mlm"]
    keys = [("test_acc", "clean"), ("typo30_acc", "30% typos"), ("imdb_acc", "OOD IMDb")]
    w = 0.13
    for i, m in enumerate(ms_):
        g = allr[(allr.model == m) & (allr.n == (TRAIN_SIZES[-1] if m != "vader_lexicon" else 1000))]
        ax[1].bar(np.arange(3) + (i - 2.5) * w, [100 * g[k].mean() for k, _ in keys], w, color=ALL_COL[m], label=ALL_NAMES[m])
    ax[1].set_xticks(range(3)); ax[1].set_xticklabels([l for _, l in keys]); ax[1].set_ylim(50, 85)
    ax[1].set_ylabel("accuracy (%)"); ax[1].set_title("Robustness at n = 6,920")
    h_, l_ = ax[0].get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0.17, 1, 1)); fig.legend(h_, l_, loc="lower center", ncol=4, fontsize=7.5, frameon=False); savefig(fig, "fig_families")


def families_tables(df, fam):
    if fam is None:
        return []
    allr = combined(df, fam); md = []
    order = ["vader_lexicon", "tfidf_lr", "tfidf_nb", "tfidf_svm", "tfidf_char_lr", "w2v_avg_lr", "fasttext_avg_lr", "tiny_scratch", "tiny_mlm"]
    rows = []
    for m in order:
        n1, n2 = (1000, 1000) if m == "vader_lexicon" else (1000, TRAIN_SIZES[-1])
        g1, g2 = allr[(allr.model == m) & (allr.n == n1)], allr[(allr.model == m) & (allr.n == n2)]
        rows.append([ALL_NAMES[m], ms(g1.test_acc), ms(g2.test_acc), ms(g2.typo30_acc), ms(g2.drop20_acc), ms(g2.imdb_acc)])
    tex_table("tab_families.tex", ["Model family", "$n{=}1000$", "$n{=}6920$", "30\\% typos", "20\\% drop", "OOD IMDb"], rows, "lccccc",
              "Model families under one protocol (accuracy \\%, mean$\\pm$std over 3 seeds; VADER uses no task labels, so its row is independent of $n$; robustness columns at $n{=}6920$).", "tab:families")
    md.append("## Model families\n" + pd.DataFrame(rows, columns=["model", "n=1000", "n=6920", "typo30", "drop20", "OOD"]).to_markdown(index=False))
    rows2 = []
    for m in order[1:] if False else order:
        if m == "tfidf_lr":
            continue
        a1 = 100 * allr[(allr.model == m) & (allr.n == 1000)].test_acc.mean(); a2 = 100 * allr[(allr.model == m) & (allr.n == TRAIN_SIZES[-1])].test_acc.mean()
        rows2.append([ALL_NAMES[m], equiv_labels(allr, a1), equiv_labels(allr, a2) if m != "vader_lexicon" else "--"])
    hfp = RES / "hf_results.jsonl"
    if hfp.exists():
        h = pd.DataFrame([json.loads(l) for l in open(hfp)]); h = h[h.stage == "hf"]
        for name, g in h.groupby("model"):
            vals = {n: 100 * gg.test_acc.mean() for n, gg in g.groupby("n")}
            rows2.append([name.replace("_", "\\_"), equiv_labels(allr, vals[1000]) if 1000 in vals else "--", equiv_labels(allr, vals[6920]) if 6920 in vals else "--"])
    tex_table("tab_equiv.tex", ["Model", "$n{=}1000$", "$n{=}6920$"], rows2, "lcc",
              "Label-equivalence: number of TF-IDF+LogReg labels needed to match each model's accuracy (log-linear interpolation of the TF-IDF curve).", "tab:equiv")
    md.append("## Label-equivalence (TF-IDF+LR labels)\n" + pd.DataFrame(rows2, columns=["model", "n=1000", "n=6920"]).to_markdown(index=False))
    return md


SHORT = {"tfidf_lr": "TF-IDF\nword LR", "tfidf_char_lr": "TF-IDF\nchar LR", "tfidf_svm": "TF-IDF\nword SVM", "vader_lexicon": "VADER\nlexicon",
         "fasttext_avg_lr": "fastText\navg+LR", "w2v_avg_lr": "word2vec\navg+LR", "tiny_scratch": "Tiny\nrandom", "tiny_mlm": "Tiny\n+MLM"}
AMB_ORDER = ["tfidf_lr", "tfidf_char_lr", "tfidf_svm", "vader_lexicon", "fasttext_avg_lr", "w2v_avg_lr", "tiny_scratch", "tiny_mlm"]


def ambiguity_assets():
    try:
        import ambiguity
        r = ambiguity.main(CORE / "preds", output_dir=CORE)
    except Exception as e:                                   # e.g. no predictions available
        print(f"[make_assets] ambiguity analysis skipped ({type(e).__name__}: {e})")
        return []
    md = []
    names = {**ALL_NAMES}
    hf_models = [m for m in r["strata"]["1000"] if m not in AMB_ORDER + ["tfidf_nb"]]
    for m in hf_models:
        names[m] = m.replace("_", "\\_")
    for n, fn in (("1000", "tab_amb_1000.tex"), ("6920", "tab_amb_full.tex")):
        rows = []
        for m in AMB_ORDER + hf_models:
            v = r["strata"][n].get(m)
            if v is None:
                continue
            e = f"{100*v['ece'][0]:.1f}" if "ece" in v else "--"
            rows.append([names[m], f"{100*v['acc_strong'][0]:.1f}", f"{100*v['acc_weak'][0]:.1f}", f"{100*(v['acc_strong'][0]-v['acc_weak'][0]):.1f}",
                         f"{v['auroc'][0]:.3f}", e, f"{100*v['acc50'][0]:.1f}"])
        tex_table(fn, ["Model", "Acc.\\ strong", "Acc.\\ weak", "Gap", "Ambiguity AUROC", "ECE", "Acc.@50\\% cov."], rows, "lcccccc",
                  f"Accuracy (\\%) on strong- vs weak-polarity sentences, ambiguity-awareness AUROC (0.5 = chance), calibration error (ECE, \\%; -- for score-only models) and accuracy on the 50\\% most confident predictions; $n={n}$ labels, mean over 3 seeds (strong: 678 sentences, weak: 1\\,143).",
                  f"tab:amb{n}")
        md.append(f"## Ambiguity strata, n={n}\n" + pd.DataFrame([[str(re_).replace("\\", "") for re_ in x] for x in rows], columns=["model", "acc strong", "acc weak", "gap", "AUROC", "ECE", "acc@50%"]).to_markdown(index=False))
    # bootstrap table
    rows = []
    lab = {"tiny_mlm-tiny_scratch": "+MLM $-$ random init", "tiny_mlm-tfidf_lr": "+MLM $-$ TF-IDF LR", "tiny_scratch-tfidf_lr": "random init $-$ TF-IDF LR",
           "tiny_scratch-fasttext_avg_lr": "random init $-$ fastText", "tiny_mlm-fasttext_avg_lr": "+MLM $-$ fastText"}
    def cell(x): return f"{100*x[0]:+.1f} [{100*x[1]:+.1f}, {100*x[2]:+.1f}]" + ("$^*$" if x[1] > 0 or x[2] < 0 else "")
    for n in ("1000", "6920"):
        for k, d in r["bootstrap"][n].items():
            rows.append([n, lab[k]] + [cell(d[s]) for s in ("strong", "weak", "strong_minus_weak", "contrast", "no_contrast")])
    tex_table("tab_amb_ci.tex", ["$n$", "Comparison", "Strong", "Weak", "Strong$-$weak", "Contrast", "No contrast"], rows, "llccccc",
              "Paired bootstrap (95\\% CI, points) of accuracy differences within ambiguity strata; ``Strong$-$weak'' tests whether the difference is larger on strong than on weak sentences. $^*$ = CI excludes 0.", "tab:ambci")
    md.append("## Ambiguity bootstrap\n" + pd.DataFrame([[str(re_).replace("$^*$", "*") for re_ in x] for x in rows], columns=["n", "cmp", "strong", "weak", "strong-weak", "contrast", "no contrast"]).to_markdown(index=False))
    # validity table
    rows = []
    for n in ("1000", "6920"):
        v = r["validity"][n]
        for g, x in v["groups"].items():
            rows.append([n, g, x["count"], f"{100*x['weak_share']:.1f}\\%"])
        rows.append([n, "all sentences (base rate)", sum(x["count"] for x in v["groups"].values()), f"{100*v['base_weak_share']:.1f}\\%"])
    tex_table("tab_amb_valid.tex", ["$n$", "Consensus difficulty group", "Sentences", "Share weak polarity"], rows, "llcc",
              "Convergent validity of the polarity-strength proxy: share of weak-polarity sentences among those that few/many of the 27 model runs classify correctly.", "tab:ambvalid")
    md.append("## Consensus vs polarity strength\n" + pd.DataFrame([[str(re_).replace("\\", "") for re_ in x] for x in rows], columns=["n", "group", "count", "weak share"]).to_markdown(index=False))
    # figure
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    ms_ = [m for m in AMB_ORDER if m in r["strata"]["6920"]]
    xs = np.arange(len(ms_)); w = 0.38
    ax[0].bar(xs - w / 2, [100 * r["strata"]["6920"][m]["acc_strong"][0] for m in ms_], w, color="#2c7fb8", label="strong polarity")
    ax[0].bar(xs + w / 2, [100 * r["strata"]["6920"][m]["acc_weak"][0] for m in ms_], w, color="#fdae61", label="weak polarity (ambiguous)")
    ax[0].set_xticks(xs); ax[0].set_xticklabels([SHORT.get(m, m[:10]) for m in ms_], fontsize=6.5, rotation=0)
    ax[0].set_ylim(55, 95); ax[0].set_ylabel("accuracy (%)"); ax[0].set_title("Accuracy by ambiguity stratum (n = 6,920)"); ax[0].legend(fontsize=8)
    for j, (n, off) in enumerate((("1000", -w / 2), ("6920", w / 2))):
        ax[1].bar(xs + off, [r["strata"][n][m]["auroc"][0] for m in ms_ if m in r["strata"][n]], w, color=["#7fcdbb", "#41b6c4"][j], label=f"n = {n}")
    ax[1].axhline(0.5, ls="--", c="k", lw=0.8); ax[1].set_ylim(0.45, 0.66); ax[1].set_xticks(xs)
    ax[1].set_xticklabels([SHORT.get(m, m[:10]) for m in ms_], fontsize=6.5)
    ax[1].set_ylabel("AUROC: low confidence -> weak polarity"); ax[1].set_title("Ambiguity-awareness (0.5 = chance)"); ax[1].legend(fontsize=8)
    fig.tight_layout(); savefig(fig, "fig_ambiguity")
    return md


def bootstrap_tests(df, B=2000):
    y = load_sst2()["test"][1]; rng = np.random.RandomState(0); rows = []; md = []
    def corr(model, n, tag):
        arr = []
        for s in SEEDS:
            f = (CORE / "preds") / (f"curve_{model}__{n}_{s}.npy" if model.startswith("tfidf") else f"curve_{model}_{tag}_{n}_{s}.npy")
            arr.append((np.load(f).argmax(1) == y).astype(float))
        return np.stack(arr)
    for a, b in (("tiny_mlm", "tiny_scratch"), ("tiny_mlm", "tfidf_lr"), ("tiny_scratch", "tfidf_lr")):
        cells = []
        for n in TRAIN_SIZES:
            d = (corr(a, n, "") - corr(b, n, "")).mean(0)
            boots = np.array([d[rng.randint(0, len(d), len(d))].mean() for _ in range(B)])
            lo, hi = np.percentile(boots, [2.5, 97.5])
            cells.append(f"{100*d.mean():+.1f} [{100*lo:+.1f}, {100*hi:+.1f}]" + ("$^*$" if lo > 0 or hi < 0 else ""))
        SH = {"tiny_mlm": "Tiny +MLM", "tiny_scratch": "Tiny random", "tfidf_lr": "TF-IDF+LR"}
        rows.append([f"{SH[a]} $-$ {SH[b]}"] + cells)
        md.append([NAMES[a], NAMES[b]] + cells)
    tex_table("tab_stats.tex", ["Comparison"] + [f"$n{{=}}{n}$" for n in TRAIN_SIZES], rows, "l" + "c" * len(TRAIN_SIZES),
              "Paired bootstrap (2000 resamples of the test set, pooled over 3 seeds): accuracy difference in points with 95\\% CI. $^*$ = CI excludes 0.",
              "tab:stats")
    return "## Paired bootstrap (accuracy difference, pp, 95% CI)\n" + pd.DataFrame(md, columns=["A", "B"] + [str(n) for n in TRAIN_SIZES]).to_markdown(index=False)


# ------------------------------------------------------------------ RQ8: cross-domain transfer matrix
DOMAINS3 = ["sst2", "finance", "airline"]
DOMAIN_LABEL3 = {"sst2": "SST-2\n(movies)", "finance": "Finance\n(news)", "airline": "Airline\n(tweets)"}


def load_domain_results(domain):
    """Concatenate results.jsonl + families.jsonl rows for one domain (curve/family stages only)."""
    frames = []
    for fname in ("results.jsonl", "families.jsonl"):
        f = CORE / fname
        if f.exists():
            d = only_domain(pd.DataFrame([json.loads(l) for l in open(f)]), domain)
            frames.append(d[d.stage.isin(["curve", "family"])])
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def cross_domain_matrix():
    """3x3 heatmap + table: model trained on domain A (full data, best model per domain), tested on A/B/C.
    Uses whichever of {tiny_mlm, tiny_scratch, tfidf_lr} has full-data rows in each domain (mean over seeds)."""
    have = {d: load_domain_results(d) for d in DOMAINS3}
    if all(d.empty for d in have.values()):
        return None, None
    mat = {}
    for train_d, df in have.items():
        if df.empty:
            continue
        full_n = df.n.max()
        for model in ("tiny_mlm", "tiny_scratch", "tfidf_lr"):
            g = df[(df.model == model) & (df.n == full_n)]
            if g.empty:
                continue
            row = {train_d: {
                "acc_mean": float(g.test_acc.mean()),
                "acc_sd": float(g.test_acc.std(ddof=1)) if len(g) > 1 else 0.0,
                "f1_mean": float(g.test_f1.mean()),
                "f1_sd": float(g.test_f1.std(ddof=1)) if len(g) > 1 else 0.0,
            }}
            for test_d in DOMAINS3:
                if test_d == train_d:
                    continue
                acc_col, f1_col = f"cross_{test_d}_acc", f"cross_{test_d}_f1"
                if acc_col in g and f1_col in g:
                    row[test_d] = {
                        "acc_mean": float(g[acc_col].mean()),
                        "acc_sd": float(g[acc_col].std(ddof=1)) if len(g) > 1 else 0.0,
                        "f1_mean": float(g[f1_col].mean()),
                        "f1_sd": float(g[f1_col].std(ddof=1)) if len(g) > 1 else 0.0,
                    }
            mat.setdefault(model, {})[train_d] = row
    if not mat:
        return None, None
    fig, axes = plt.subplots(1, len(mat), figsize=(4.6 * len(mat), 3.8))
    if len(mat) == 1:
        axes = [axes]
    tex_rows = []
    for ax, (model, rows) in zip(axes, mat.items()):
        M = np.full((3, 3), np.nan)
        for i, tr in enumerate(DOMAINS3):
            for j, te in enumerate(DOMAINS3):
                if tr in rows and te in rows[tr]:
                    M[i, j] = 100 * rows[tr][te]["acc_mean"]
        im = ax.imshow(M, cmap="YlGnBu", vmin=20, vmax=95)
        ax.set_xticks(range(3)); ax.set_xticklabels([DOMAIN_LABEL3[d] for d in DOMAINS3], fontsize=8)
        ax.set_yticks(range(3)); ax.set_yticklabels([DOMAIN_LABEL3[d] for d in DOMAINS3], fontsize=8)
        ax.set_xlabel("tested on"); ax.set_ylabel("trained on")
        ax.set_title(NAMES.get(model, model), fontsize=10)
        for i in range(3):
            for j in range(3):
                if not np.isnan(M[i, j]):
                    ax.text(j, i, f"{M[i,j]:.1f}", ha="center", va="center",
                            color="white" if M[i, j] > 70 else "black", fontsize=9)
        for i, tr in enumerate(DOMAINS3):
            for j, te in enumerate(DOMAINS3):
                if tr in rows and te in rows[tr]:
                    cell = rows[tr][te]
                    tex_rows.append([NAMES.get(model, model), DOMAIN_LABEL3[tr].replace(chr(10), " "),
                                     DOMAIN_LABEL3[te].replace(chr(10), " "),
                                     f"${100*cell['acc_mean']:.1f}\\pm{100*cell['acc_sd']:.1f}$",
                                     f"${100*cell['f1_mean']:.1f}\\pm{100*cell['f1_sd']:.1f}$"])
    fig.tight_layout(); savefig(fig, "fig_cross_domain")
    tex_table("tab_cross_domain.tex", ["Model", "Trained on", "Tested on", "Accuracy (mean $\\pm$ SD, \\%)", "Macro-F1 (mean $\\pm$ SD, \\%)"], tex_rows, "llccc",
              "Cross-domain transfer: mean and standard deviation over three seeds for neural models; deterministic TF-IDF runs have zero observed seed variation. Macro-F1 complements accuracy under domain-specific class imbalance.",
              "tab:cross")
    return tex_rows, "## Cross-domain transfer matrix (mean ± SD; accuracy and Macro-F1)\n" + pd.DataFrame(tex_rows, columns=["model", "trained_on", "tested_on", "accuracy", "macro_f1"]).to_markdown(index=False)


def multidomain_learning_tables(df=None):
    """Write a reproducible Macro-F1 learning-curve table for finance and airline.

    These domains are class-imbalanced, so the accuracy-only learning curves should
    not be the only basis for comparing models. Values are mean +/- population SD
    across the three saved seeds; this is descriptive, not a significance test.
    """
    if df is None:
        df = load(None)
    c = df[(df.stage == "curve") & (df.model.isin(NAMES))].copy()
    acc_rows, f1_rows = [], []
    for domain in ("finance", "airline"):
        sizes = {"finance": [50, 100, 300, 700, 1377],
                 "airline": [100, 300, 1000, 3000, 8125]}[domain]
        for model in NAMES:
            gmodel = c[(c.domain == domain) & (c.model == model)]
            if gmodel.empty:
                continue
            acc_vals, f1_vals = [], []
            for n in sizes:
                g = gmodel[gmodel.n == n]
                acc_vals.append(ms(g.test_acc) if len(g) else "--")
                f1_vals.append(ms(g.test_f1) if len(g) else "--")
            acc_rows.append([domain.capitalize(), NAMES[model]] + acc_vals)
            f1_rows.append([domain.capitalize(), NAMES[model]] + f1_vals)
    header = ["Domain", "Model", "$n_1$", "$n_2$", "$n_3$", "$n_4$", "$n_5$ (full)"]
    if acc_rows:
        tex_table("tab_fin_air.tex", header, acc_rows, "llccccc",
                  "Accuracy learning curves on Financial PhraseBank and Twitter Airline Sentiment (mean$\\pm$SD over three seeds).",
                  "tab:fin_air")
    if f1_rows:
        tex_table("tab_fin_air_f1.tex", header, f1_rows, "llccccc",
                  "Macro-F1 learning curves on Financial PhraseBank and Twitter Airline Sentiment (mean$\\pm$SD over three seeds). These descriptive summaries complement accuracy under class imbalance; they are not inferential tests.",
                  "tab:finairf1")
    return f1_rows


def domain_ambiguity_summary():
    """Compact full-budget summary of real agreement/confidence strata for the paper."""
    rows = []
    core_models = ("tfidf_lr", "tiny_scratch", "tiny_mlm")
    for domain in ("finance", "airline"):
        f = CORE / f"ambiguity_{domain}.json"
        if not f.exists():
            continue
        r = json.loads(f.read_text())
        n = str(DOMAIN_TRAIN_SIZES[domain][-1])
        strata = r.get("strata", {}).get(n, {})
        signal = "Agreement" if domain == "finance" else "Worker confidence"
        for model in core_models:
            if model not in strata:
                continue
            v = strata[model]
            ece = v.get("ece", [None])[0]
            rows.append([domain.capitalize(), signal, NAMES.get(model, model),
                         f"{100*v['acc_strong'][0]:.1f}", f"{100*v['acc_weak'][0]:.1f}",
                         f"{v['auroc'][0]:.3f}", f"{100*ece:.1f}" if ece is not None else "--"])
    if rows:
        tex_table("tab_amb_domain_summary.tex",
                  ["Domain", "Signal", "Model", "High-signal acc. (\\%)", "Low-signal acc. (\\%)", "AUROC", "ECE (\\%)"],
                  rows, "lllcccc",
                  "Full-budget accuracy by real ambiguity-related signal: finance uses 100\\% vs 50--65\\% annotator agreement; airline uses crowd-worker confidence $\\geq0.9$ vs $<0.65$. AUROC measures whether model confidence identifies the low-signal stratum. Signal definitions differ by domain, so rows should not be compared as a common scale.",
                  "tab:ambdomains")
    return rows


def domain_ambiguity_assets(domain):
    """Per-domain ambiguity tables using that domain's REAL signal (agreement tier / confidence), reusing
    the same table/figure helpers as the sst2 analysis. Returns markdown or None if no data yet."""
    try:
        import ambiguity
        r = ambiguity.main(CORE / "preds", domain, output_dir=CORE)   # ALWAYS recompute fresh (no stale-cache short-circuit)
    except Exception as e:
        print(f"[make_assets] {domain} ambiguity skipped ({type(e).__name__}: {e})")
        return None
    if not r["strata"]:
        return None
    md = [f"## {domain}: real ambiguity signal ({r['signal']['kind']})\n"
          f"strong={r['counts']['strong']}, weak={r['counts']['weak']}, total={r['counts']['total']}"]
    rows = []
    for n, strata in r["strata"].items():
        for m, v in strata.items():
            rows.append([n, m, f"{100*v['acc_strong'][0]:.1f}", f"{100*v['acc_weak'][0]:.1f}",
                        f"{v['auroc'][0]:.3f}", f"{100*v.get('ece', [float('nan')])[0]:.1f}" if "ece" in v else "--"])
    if rows:
        tex_table(f"tab_amb_{domain}.tex", ["$n$", "Model", "Acc. strong", "Acc. weak", "AUROC", "ECE"], rows, "rlcccc",
                  f"{domain.capitalize()}: accuracy on strong/weak ({r['signal']['kind']}) sentences, ambiguity AUROC and ECE.",
                  f"tab:amb{domain}")
        md.append(pd.DataFrame(rows, columns=["n", "model", "acc_strong", "acc_weak", "auroc", "ece"]).to_markdown(index=False))
    return "\n".join(md)



def main():
    df = load(); md = []
    log = fig_pretrain(); fig_learning(df); fig_robust(df); fig_ckpt(df)
    eff = json.load(open(CORE / "efficiency.json")) if (CORE / "efficiency.json").exists() else None
    if eff:
        fig_eff(eff, df)
    md += tables(df, eff)
    try:
        md.append(bootstrap_tests(df))
    except FileNotFoundError as e:
        print(f"[make_assets] skipping bootstrap table (missing predictions: {e.filename})")
    fam = load_families(); fig_families(df, fam); fig_hf(df, eff)
    md += families_tables(df, fam); md += ambiguity_assets()
    for fn in (hf_table, lambda: hf_eff_table(eff), hf_bootstrap):
        out = fn()
        if out:
            md.append(out)
    try:
        _, cross_md = cross_domain_matrix()
        if cross_md:
            md.append(cross_md)
        f1_rows = multidomain_learning_tables(load(None))
        if f1_rows:
            md.append("## Finance/Airline in-domain Macro-F1 (mean ± SD over 3 seeds)\n" +
                       pd.DataFrame(f1_rows, columns=["domain", "model", "n1", "n2", "n3", "n4", "n5"]).to_markdown(index=False))
    except Exception as e:
        print(f"[make_assets] cross-domain matrix skipped ({type(e).__name__}: {e})")
    for d in ("finance", "airline"):
        out = domain_ambiguity_assets(d)
        if out:
            md.append(out)
    try:
        ambiguity_rows = domain_ambiguity_summary()
        if ambiguity_rows:
            md.append("## Full-budget ambiguity signals across domains\n" + pd.DataFrame(
                ambiguity_rows, columns=["domain", "signal", "model", "high_signal_acc", "low_signal_acc", "auroc", "ece"]).to_markdown(index=False))
    except Exception as e:
        print(f"[make_assets] domain ambiguity summary skipped ({type(e).__name__}: {e})")
    last = log.iloc[-1]
    md.insert(0, f"# Results summary\n\nPre-training: final held-out MLM loss {last.val_loss:.3f} (ppl {last.val_ppl:.0f}), "
                 f"masked-token acc {100*last.val_acc:.1f}% after {int(last.step)} steps, {last.elapsed_s/60:.1f} min.\n")
    (RES / "summary.md").write_text("\n\n".join(md))
    # export the raw table as CSV
    load(None).drop(columns=["history"], errors="ignore").to_csv(RES / "all_runs.csv", index=False)   # every domain
    print("\n\n".join(md))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default=None,
                    help="folder with the core experiment outputs (results.jsonl, efficiency.json, logs/pretrain.jsonl, preds/); "
                         "default: results/. Use results_sandbox_reference to reuse the reference run.")
    a = ap.parse_args()
    if a.results_dir:
        CORE = Path(a.results_dir).resolve()
    if not (CORE / "results.jsonl").exists():
        raise SystemExit(f"[make_assets] {CORE/'results.jsonl'} not found.\n"
                         "  -> run the core experiments (bash run_all.sh), or reuse the reference run:\n"
                         "     python make_assets.py --results_dir ../results_sandbox_reference")
    main()
