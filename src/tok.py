"""Train a small lower-cased WordPiece tokenizer and save it in HF format."""
from tokenizers import Tokenizer, models, trainers, normalizers, pre_tokenizers, processors, decoders
from transformers import PreTrainedTokenizerFast

SPECIALS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]


def train_tokenizer(texts, vocab_size, out_dir):
    tok = Tokenizer(models.WordPiece(unk_token="[UNK]"))
    tok.normalizer = normalizers.BertNormalizer(lowercase=True)
    tok.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    trainer = trainers.WordPieceTrainer(vocab_size=vocab_size, special_tokens=SPECIALS, min_frequency=2)
    tok.train_from_iterator(texts, trainer)
    tok.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]", pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[("[CLS]", tok.token_to_id("[CLS]")), ("[SEP]", tok.token_to_id("[SEP]"))])
    tok.decoder = decoders.WordPiece()
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="[UNK]", pad_token="[PAD]",
                                   cls_token="[CLS]", sep_token="[SEP]", mask_token="[MASK]",
                                   model_max_length=512)
    fast.save_pretrained(out_dir)
    return fast
