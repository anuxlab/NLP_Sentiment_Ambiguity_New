"""Self-supervised masked-language-model (MLM) pre-training of a tiny BERT on CPU.

    python pretrain_mlm.py --steps 5000 --lr 1e-3 --dropout 0.0 --save_steps 0,100,300,1000,2500,5000

Outputs (HF format, loadable with `BertModel.from_pretrained`):
    results/models/mlm_step{N}/   for every N in --save_steps  (N=0 -> random init)
    results/models/mlm_final/     last step
    results/models/tokenizer/     WordPiece tokenizer
    results/logs/pretrain.jsonl   training / held-out loss + masked-token accuracy
"""
import argparse, json, math, time
import numpy as np
import torch
from transformers import BertConfig, BertForMaskedLM, PreTrainedTokenizerFast

from config import ARCH, MAX_LEN, MODELS, RES, VOCAB_SIZE
from data import load_movie_reviews
from tok import train_tokenizer


def build_blocks(tokenizer, docs, block=MAX_LEN - 2):
    cls, sep = tokenizer.cls_token_id, tokenizer.sep_token_id
    ids = tokenizer(docs, add_special_tokens=False, truncation=False)["input_ids"]
    stream = []
    for x in ids:
        stream.extend(x + [sep])
    n = len(stream) // block
    arr = np.array(stream[: n * block]).reshape(n, block)
    out = np.concatenate([np.full((n, 1), cls), arr, np.full((n, 1), sep)], axis=1)
    return torch.tensor(out, dtype=torch.long)


def mask_batch(x, tok, gen, p=0.15):
    special = (x == tok.cls_token_id) | (x == tok.sep_token_id) | (x == tok.pad_token_id)
    prob = torch.full(x.shape, p); prob[special] = 0
    m = torch.bernoulli(prob, generator=gen).bool()
    r = torch.rand(x.shape, generator=gen)
    inp = x.clone()
    inp[m & (r < 0.8)] = tok.mask_token_id                       # 80 % [MASK]
    rnd = m & (r >= 0.8) & (r < 0.9)                             # 10 % random token
    inp[rnd] = torch.randint(5, len(tok), x.shape, generator=gen)[rnd]
    return inp, m                                                # 10 % unchanged


def mlm_loss(model, inp, tgt, m):
    h = model.bert(input_ids=inp).last_hidden_state              # no padding: fixed-length blocks
    logits = model.cls(h[m])                                     # vocab projection only at masked positions
    loss = torch.nn.functional.cross_entropy(logits, tgt[m])
    acc = (logits.argmax(-1) == tgt[m]).float().mean()
    return loss, acc


@torch.no_grad()
def evaluate(model, tok, blocks, bs=128):
    model.eval(); gen = torch.Generator().manual_seed(123)
    ls, ac, n = 0., 0., 0
    for i in range(0, len(blocks), bs):
        x = blocks[i:i + bs]; inp, m = mask_batch(x, tok, gen)
        l, a = mlm_loss(model, inp, x, m)
        ls += l.item() * len(x); ac += a.item() * len(x); n += len(x)
    model.train()
    return ls / n, ac / n


def save(model, tok, name):
    d = MODELS / name
    model.save_pretrained(d); tok.save_pretrained(d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--save_steps", type=str, default="0,100,300,1000,2500,5000")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dropout", type=float, default=0.0, help="dropout during pre-training (fine-tuning always uses 0.1)")
    a = ap.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)

    docs, _ = load_movie_reviews()
    tok = train_tokenizer(docs, VOCAB_SIZE, MODELS / "tokenizer")
    blocks = build_blocks(tok, docs)
    perm = torch.randperm(len(blocks), generator=torch.Generator().manual_seed(0))
    n_val = max(128, int(0.05 * len(blocks)))
    val, train = blocks[perm[:n_val]], blocks[perm[n_val:]]
    print(f"[pretrain] vocab={len(tok)}  train blocks={len(train)}  val blocks={len(val)}  "
          f"tokens/epoch={train.numel():,}")

    arch = dict(ARCH, hidden_dropout_prob=a.dropout, attention_probs_dropout_prob=a.dropout)
    cfg = BertConfig(vocab_size=len(tok), pad_token_id=tok.pad_token_id, **arch)
    model = BertForMaskedLM(cfg)
    n_params = sum(p.numel() for p in model.bert.parameters())
    print(f"[pretrain] encoder params = {n_params/1e6:.2f} M")
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01, betas=(0.9, 0.98))
    warm = int(0.06 * a.steps)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else max(0.0, (a.steps - s) / (a.steps - warm)))
    save_at = set(int(s) for s in a.save_steps.split(","))
    log = open(RES / "logs" / "pretrain.jsonl", "w")
    gen = torch.Generator().manual_seed(a.seed)

    if 0 in save_at:
        save(model, tok, "mlm_step0")
    model.train(); t0 = time.time(); step = 0; run_l, run_a = [], []
    while step < a.steps:
        order = torch.randperm(len(train), generator=gen)
        for i in range(0, len(order) - a.bs + 1, a.bs):
            x = train[order[i:i + a.bs]]
            inp, m = mask_batch(x, tok, gen)
            loss, acc = mlm_loss(model, inp, x, m)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            step += 1; run_l.append(loss.item()); run_a.append(acc.item())
            if step % 50 == 0 or step == a.steps:
                vl, va = evaluate(model, tok, val)
                rec = dict(step=step, train_loss=float(np.mean(run_l)), train_acc=float(np.mean(run_a)),
                           val_loss=vl, val_acc=va, val_ppl=math.exp(vl),
                           lr=sched.get_last_lr()[0], elapsed_s=time.time() - t0)
                log.write(json.dumps(rec) + "\n"); log.flush(); run_l, run_a = [], []
                print(f"[pretrain] step {step:5d}/{a.steps}  train_loss {rec['train_loss']:.3f}  "
                      f"val_loss {vl:.3f} (ppl {rec['val_ppl']:.0f})  val_masked_acc {va:.3f}  "
                      f"{rec['elapsed_s']/60:.1f} min", flush=True)
            if step in save_at:
                save(model, tok, f"mlm_step{step}")
            if step >= a.steps:
                break
    save(model, tok, "mlm_final")
    print("[pretrain] done.")


if __name__ == "__main__":
    main()
