"""Global configuration shared by all scripts."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RES = ROOT / "results"
MODELS = RES / "models"
PREDS = RES / "preds"
FIGS = RES / "figures"
for _p in (DATA, RES, MODELS, PREDS, FIGS, RES / "logs"):
    _p.mkdir(parents=True, exist_ok=True)

MAX_LEN = 64            # max wordpieces per input (also the MLM block size)
VOCAB_SIZE = 6000       # WordPiece vocabulary trained on the unlabeled corpus

# Tiny BERT architecture (~2.2 M parameters incl. embeddings)
ARCH = dict(hidden_size=128, num_hidden_layers=2, num_attention_heads=2,
            intermediate_size=512, max_position_embeddings=MAX_LEN,
            hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1)

TRAIN_SIZES = [100, 300, 1000, 3000, 6920]   # 6920 = full SST-2 train split

# Multi-domain registry (RQ8: cross-domain generalisation). Each domain has its own budgets, scaled to its
# own training-pool size (train split sizes below are approximate; exact counts printed by data.py loaders).
# Legacy hard-label cross-domain runner. Finance/Airline remain readable for reproducing
# archived exploratory runs, but are excluded from the primary distribution-aware study.
DOMAINS = ["sst2"]

# Primary multi-domain distribution-aware extension. These datasets expose individual
# human labels; see annotated_domains.py and run_annotated_domain_experiments.py.
DISTRIBUTIONAL_DOMAINS = ["mvsc", "sentimp_en"]
DISTRIBUTIONAL_DOMAIN_TRAIN_SIZES = {
    "mvsc": [50, 100, 300, 700, 1500],
    "sentimp_en": [25, 50, 100, 200, 300],
}
DOMAIN_TRAIN_SIZES = {
    "sst2": TRAIN_SIZES,                      # 6,920 train sentences (movie reviews)
    "finance": [50, 100, 300, 700, 1377],      # 1,377 train sentences (finance news, real agreement tiers)
    "airline": [100, 300, 1000, 3000, 8125],   # 8,125 train tweets (social media, real confidence scores)
}
DOMAIN_LABEL = {"sst2": "SST-2 (movie reviews)", "finance": "Financial PhraseBank (finance news)", "airline": "Twitter Airline (social media)"}
SEEDS = [0, 1, 2]
