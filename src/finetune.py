"""Fine-tuning + evaluation utilities shared by every experiment.

Any encoder can be plugged in:
    init="scratch"  random weights, same architecture as our tiny BERT
    init="mlm"      our MLM-pre-trained checkpoint (local dir)
    init="hf"       any HuggingFace checkpoint (hub name or local dir), e.g. prajjwal1/bert-tiny
"""
import copy, time
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from transformers import AutoModel, AutoTokenizer, BertConfig, BertModel

from config import MAX_LEN
import transformers
transformers.logging.set_verbosity_error()

torch.set_num_threads(max(1, torch.get_num_threads()))


DROP = dict(hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1)   # fine-tuning dropout


class Classifier(nn.Module):
    """Encoder + masked mean-pooling + linear head (works for any HF encoder)."""

    def __init__(self, encoder, n_classes=2, dropout=0.1):
        super().__init__()
        self.enc = encoder
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(encoder.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        h = self.enc(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        m = attention_mask.unsqueeze(-1).to(h.dtype)
        return self.head(self.drop((h * m).sum(1) / m.sum(1).clamp(min=1)))


def build_model(init, source, seed):
    """source: checkpoint dir (scratch/mlm) or hub name/dir (hf)."""
    torch.manual_seed(seed)
    if init == "scratch":
        enc = BertModel(BertConfig.from_pretrained(source, **DROP), add_pooling_layer=False)
    elif init == "mlm":
        enc = BertModel.from_pretrained(source, add_pooling_layer=False, **DROP)
    elif init == "hf":
        enc = AutoModel.from_pretrained(source)
    else:
        raise ValueError(init)
    return Classifier(enc)


def load_tokenizer(source):
    return AutoTokenizer.from_pretrained(source)


def encode(tok, texts):
    return tok(list(texts), truncation=True, max_length=MAX_LEN, padding=False)["input_ids"]


def collate(ids_list, pad_id):
    L = max(len(x) for x in ids_list)
    ids = torch.full((len(ids_list), L), pad_id, dtype=torch.long)
    att = torch.zeros((len(ids_list), L), dtype=torch.long)
    for i, x in enumerate(ids_list):
        ids[i, :len(x)] = torch.tensor(x); att[i, :len(x)] = 1
    return ids, att


@torch.no_grad()
def predict_proba(model, tok, texts, bs=128):
    model.eval()
    ids = encode(tok, texts)
    order = np.argsort([len(x) for x in ids])                  # length-sorted batches = less padding
    out = np.zeros((len(ids), 2), dtype=np.float32)
    for i in range(0, len(ids), bs):
        b = order[i:i + bs]
        x, a = collate([ids[j] for j in b], tok.pad_token_id)
        out[b] = torch.softmax(model(x, a), -1).numpy()
    return out


def metrics(y, p):
    pred = p.argmax(1)
    return dict(acc=float((pred == y).mean()), f1=float(f1_score(y, pred, average="macro")))


def fit(model, tok, tr_texts, tr_y, dev_texts, dev_y, lr, seed, bs=32, max_epochs=None):
    """AdamW + linear warm-up/decay; keeps the epoch with best dev accuracy."""
    n = len(tr_texts)
    steps_per_epoch = int(np.ceil(n / bs))
    epochs = max_epochs or int(np.clip(np.ceil(600 / steps_per_epoch), 5, 30))
    ids = encode(tok, tr_texts); y = torch.tensor(tr_y)
    total = epochs * steps_per_epoch; warm = max(1, int(0.1 * total))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else max(0.0, (total - s) / (total - warm)))
    rng = np.random.RandomState(seed)
    best, best_state, best_ep, hist = -1, None, -1, []
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        perm = rng.permutation(n); tl = 0.0
        for i in range(0, n, bs):
            b = perm[i:i + bs]
            x, a = collate([ids[j] for j in b], tok.pad_token_id)
            loss = nn.functional.cross_entropy(model(x, a), y[b])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            tl += loss.item() * len(b)
        dev_acc = metrics(dev_y, predict_proba(model, tok, dev_texts))["acc"]
        hist.append(dict(epoch=ep, train_loss=tl / n, dev_acc=dev_acc))
        if dev_acc > best:
            best, best_ep, best_state = dev_acc, ep, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return dict(dev_acc=best, best_epoch=best_ep, epochs=epochs, train_time_s=time.time() - t0, history=hist)
