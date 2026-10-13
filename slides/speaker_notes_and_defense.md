# Speaker notes and defense guide

**Talk:** 10 minutes (12 slides, ~50s each) + 5 minutes Q&A. Backup slides B1-B4 on demand only.

## Timing plan

| Slide | Time |
|---|---|
| Slide 1 - Title | 0:00-0:45 |
| Slide 2 - Problem | 0:45-1:45 |
| Slide 3 - Approach | 1:45-2:45 |
| Slide 4 - Plateau | 2:45-3:15 |
| Slide 5 - Verdict | 3:15-4:00 |
| Slide 6 - Families | 4:00-5:00 |
| Slide 7 - Label-equivalence | 5:00-5:30 |
| Slide 8 - Decoupling | 5:30-6:15 |
| Slide 9 - Ambiguity | 6:15-7:15 |
| Slide 10 - Robustness | 7:15-8:00 |
| Slide 11 - Twist | 8:00-8:45 |
| Slide 11b - Cross-domain | 8:45-9:30 |

## Speaker script

### Slide 1 - Title  (0:00-0:45)

Good morning. I compare ways to use a small unlabeled corpus: train static embeddings or a tiny contextual model with MLM. I also include VADER as a separate external sentiment lexicon that uses no task labels and does not consume that corpus.
The CPU-only pipeline compares those approaches with TF-IDF, a tiny BERT with and without pre-training, and public checkpoints.
Beyond accuracy, I also look at robustness and - the twist - where models succeed or fail under ambiguity. Ten minutes, then five for questions.

### Slide 2 - Problem  (0:45-1:45)

Eight questions share a set of evaluation splits and test procedures. Model-specific tokenizers, features, and optimization settings differ where necessary; VADER is an external lexicon, while fastText and MLM use the same unlabeled corpus. RQ8 adds finance and airline transfer to the original SST-2 experiments.

### Slide 3 - Approach  (1:45-2:45)

Six families, one protocol. A VADER sentiment lexicon that never sees a training label. TF-IDF with word or character n-grams. word2vec and fastText - static embeddings trained on the exact same 1.2 million unlabeled words as our MLM pre-training. Our tiny BERT, with and without that pre-training. And public BERT-Tiny and DistilBERT.
Everything is evaluated on clean data, typos, word-drop, out-of-domain IMDb reviews, and - new for this version - ambiguity strata built from fine-grained SST-5 labels, confidence calibration, and simple linguistic markers like contrast words.
The tiny BERT has 1.17 million parameters and pre-trains in 13 minutes on one CPU.

### Slide 4 - Plateau  (2:45-3:15)

Quick sanity check before results. For the first 1,500 of 5,000 pre-training steps, the loss sits near the unigram entropy - the model is only learning word frequencies. It breaks out after that, ending at perplexity 77 and 29 percent masked-token accuracy. So pre-training is doing something real; the question is whether it helps.

### Slide 5 - Verdict  (3:15-4:00)

Here's the headline. From 300 labels on, the pre-trained model falls behind its own untrained twin - by 0.3 to 4.5 points, and the confidence intervals exclude zero at three of five budgets. TF-IDF then overtakes both. At laptop scale, this pre-training recipe did not pay off in the usual sense. But that's not the whole story - the next two slides are why.

### Slide 6 - Families  (4:00-5:00)

Among methods trained on the shared unlabeled corpus, fastText is the best trainable option at 100-300 labels and still beats MLM at 1,000 labels (70.2 versus 65.7 percent). It plateaus around 71 percent while TF-IDF and the transformers keep climbing. VADER is separate: it is an external lexicon using no task labels and does not consume this corpus; its 69.4 percent is equivalent to about 840 TF-IDF labels.
And character n-grams give the best typo robustness of any classical model.

### Slide 7 - Label-equivalence  (5:00-5:30)

To make these gaps concrete, I read every model's accuracy off the TF-IDF curve and asked: how many TF-IDF labels is this worth? Our pre-trained model at 1,000 labels is worth about 450 - less than half its own labels. DistilBERT at 1,000 labels beats TF-IDF with all 6,920 - almost a sevenfold saving. This is consistent with benefits from larger-scale pre-training, but the public checkpoint also differs in corpus, vocabulary and training recipe, so scale is not isolated.

### Slide 8 - Decoupling  (5:30-6:15)

Why the negative result? I tested each pre-training checkpoint two ways: fine-tune it, or freeze it and train only a linear probe. They disagree. The probe rises from 56.4 to 59.4 percent - MLM does add sentiment-relevant information. But fine-tuned accuracy falls, from 69.7 to 65.7. Our pre-trained weights are a better representation but a worse starting point for fine-tuning. The next slide shows exactly where that plays out.

### Slide 9 - Ambiguity  (6:15-7:15)

Here's the twist. I split test sentences by polarity strength, using fine-grained SST-5 labels: strong, clearly positive or negative, versus weak, mild and closer to neutral. At 1,000 labels, the estimated deficit is 4.7 points on STRONG-polarity sentences and 1.3 on WEAK-polarity sentences. The weak-stratum interval includes zero, which is inconclusive, not evidence that the effect disappears. The interaction is significant at this budget. Strong/weak polarity is an intensity proxy, not direct annotator disagreement.
And across every model I tried, confidence barely tracks ambiguity - the best AUROC for flagging a weak-polarity sentence from low confidence is 0.62, barely above chance. Worse, transformers trained on 1,000 labels are badly overconfident - calibration error of 23 to 26 percent, versus 3 to 6 for TF-IDF. You cannot trust a small transformer's confidence to tell you when it's unsure.

### Slide 10 - Robustness  (7:15-8:00)

Typos hurt every transformer more than TF-IDF - likely because WordPiece fragments misspelled words. And out-of-domain, on IMDb reviews, pre-training did not help our model, and no public checkpoint beat TF-IDF either.

### Slide 11 - Twist  (8:00-8:45)

Public BERT-Tiny shares our depth and width but differs in pre-training corpus, vocabulary, and training recipe; it scores 9.3 points above our model at 1,000 labels. This is consistent with benefits from larger-scale pre-training, but it does not isolate scale as the cause.

### Slide 11b - Cross-domain  (8:45-9:30)

Does any of this generalise beyond movies? I repeated the two sharpest findings on two more domains with REAL ambiguity signals - finance news with actual annotator agreement tiers, and airline tweets with actual crowd-worker confidence scores.
At n=100 on airline, fastText has a 3.2-point higher accuracy estimate than +MLM; +MLM trails random initialization by about 1.8 points. The archive does not currently reproduce the paired confidence intervals previously reported for these domain comparisons, so treat these as descriptive until the new bootstrap script is run.
Full-data point estimates are close in finance and airline, but the current artifacts do not support a reproducible claim of statistical equivalence. Cross-domain transfer is directional: scratch Tiny-BERT drops 4.7 points SST-2-to-airline and 35.1 points airline-to-SST-2; TF-IDF finance-to-airline drops 60.5 points. Interpret accuracy alongside Macro-F1 because the class distributions differ sharply.

### Slide 12 - Takeaways  (8:45-10:00)

Three findings. One: the best trainable use of the shared unlabeled corpus depends on label budget - fastText early, TF-IDF later; VADER is a separate external baseline. Two: at n=1000 the pre-training deficit is larger on strong-polarity examples, while the weak-stratum estimate is inconclusive. Three: confidence is a weak detector of this polarity-strength proxy, and small transformers are overconfident.
Field guide: a few hundred labels, fastText is a strong trainable option; with more labels, TF-IDF is competitive; public checkpoints perform well but the scale effect is not isolated. The three-domain base pipeline runs with one command, while public-checkpoint benchmarks are optional. Thank you - happy to take questions.

### Backup B1 - Cost  (Q&A)

Use if asked about efficiency. INT8 dynamic quantization cuts the tiny model's file by 25 percent with unchanged accuracy. TF-IDF is smaller, faster and more accurate than any transformer here. The fastText row is an estimate, not a separate measurement, and I say so if asked.

### Backup B2 - Statistics  (Q&A)

The interaction test - is the strong-polarity gap bigger than the weak-polarity gap - has a 95 percent interval of -3.5 to -0.3 at n=1000. The weak-stratum interval itself includes zero; polarity strength is not direct annotator disagreement.

### Backup B3 - Limitations  (Q&A)

Be upfront: the ambiguity signal measures polarity strength, not real annotator disagreement, and covers most of the dataset as 'weak', so it's a coarse proxy. Public checkpoints used untuned learning rates and their predictions weren't stored, so no paired tests exist for them yet. Everything else is standard caveats: three seeds, one task, one language.

### Backup B4 - Reproducibility  (Q&A)

The default command runs the three-domain base study. The public-checkpoint benchmark remains optional because it needs model downloads and additional CPU time. Saved predictions support regeneration of the reported tables and the separate multi-domain bootstrap.

## Q&A defense bank (5 minutes)

**Q1. Is the negative result just a bug?**

No: MLM loss falls from 8.2 to 4.35 (works mechanically), the frozen probe improves with more pre-training (checkpoints really differ), and the same fine-tuning code gives strong results for public BERT-Tiny and DistilBERT. The claim is limited to the tested corpus, training schedule and fine-tuning recipe, not pre-training in general.

**Q2. What is genuinely new here, since there's no new model?**

The contribution is the protocol and the analysis: one evaluation harness spanning a lexicon, static embeddings, TF-IDF, from-scratch and pre-trained transformers, and public checkpoints; the probe/fine-tune decoupling; and the ambiguity-stratified analysis showing WHERE pre-training's damage concentrates. It is an empirical study, not a new method, and I say so.

**Q3. Why does fastText do well early and then plateau?**

Averaged static embeddings capture broad topical/sentiment signal with very few labels since the embedding space already groups similar words. But averaging loses word order and can't model negation or contrast well, so it can't keep improving as more labeled examples let sparse or contextual models fit finer decision boundaries. I did not verify this mechanism directly.

**Q4. Is the ambiguity split (SST-5 strength) really 'ambiguity'?**

It's a graded-intensity proxy, not annotator disagreement, and I'm explicit about that. It's validated indirectly: sentences few models get right are disproportionately 'weak' polarity (79.6% vs 62.8% base rate), so it correlates with what's actually hard, but it is not the strongest possible ambiguity signal.

**Q5. Why is the interaction significant at n=1000 but not n=6920?**

At n=1000 the strong-sentence gap is -4.7 and the weak-sentence gap is -1.3, a large and well-estimated difference. At n=6920 both models are more accurate and the gaps shrink (-2.6 and -0.1), so the difference is smaller and the confidence interval, while still negative on average, includes zero. More seeds would sharpen this.

**Q6. Are three seeds enough?**

They give noisy standard deviations, which I flag as a limitation. The paired bootstrap covers test-set sampling variation independently. The main directional findings - TF-IDF ahead from 1,000 labels, pre-training's deficit on strong sentences - are consistent across multiple budgets and seeds, not resting on one comparison.

**Q7. Why does the transformer do relatively well on 'contrast' sentences (but/although)?**

Contextual self-attention can in principle combine information across a sentence, so it may partially resolve a contrastive clause, whereas bag-of-words methods (TF-IDF, fastText) just average or sum features regardless of structure. The gap between the transformer and TF-IDF nearly disappears on contrast sentences (-0.1 vs -3.5 elsewhere). This is consistent with, not proof of, compositional processing; I did not run a targeted structural test.

**Q8. Why are the transformers so overconfident at n=1000?**

With few examples, cross-entropy training can push output probabilities toward the extremes on the training set, and this miscalibration transfers to test time; it is a widely observed phenomenon in small-data fine-tuning, not unique to this study. TF-IDF's regularised linear scores stay closer to calibrated by construction.

**Q9. Is the public-checkpoint comparison fair?**

Not perfectly: they used fixed learning rates (3e-4, 5e-5) while ours had a small tuned grid, which if anything understates their advantage. They also differ in vocabulary size from ours, which I can't separate from pre-training scale. I say this explicitly.

**Q10. What would change your conclusion?**

If a better-tuned or larger in-domain pre-training run closed the gap to random initialization, that would narrow the negative result to this tested setup; the current public-checkpoint comparison alone cannot identify scale as the cause. If the ambiguity/strength split did not replicate with a real disagreement-based signal, I would weaken that claim specifically.

**Q11. Why mean pooling and not the [CLS] token?**

MLM never trains the [CLS] token or a pooler head, so a randomly initialised pooler would add noise. Mean pooling works uniformly for random-init, pre-trained, and public models.

**Q12. What's your practical recommendation?**

With a CPU, unlabeled in-domain text, and a few hundred labels: fastText. A few thousand: TF-IDF, word or character n-grams. If you have a public checkpoint: use it, and prefer DistilBERT-scale if you can afford the fine-tuning time. Never trust a small transformer's raw confidence to flag ambiguous inputs.

**Q13. How reproducible is this?**

One command (bash run_all.sh) runs the three-domain base study; the optional public-checkpoint benchmark needs separate model downloads. Saved predictions and make_assets.py regenerate the main figures and tables. Different hardware shifts numbers by a fraction of a point; all numbers here are from one machine.

**Q14. What's the single most interesting number?**

The interaction effect: -3.5 points [-6.7,-0.3] at n=1000, meaning pre-training's damage is concentrated on sentences that were easy to classify anyway, not spread evenly. That's the ambiguity twist in one number.

**Q15. Could this be published somewhere?**

As a controlled empirical study with a negative result and an ambiguity-stratified analysis, yes - to a workshop or student-research track that accepts negative results and reproducibility studies, not yet as a full-length top-venue paper without reproducible multi-domain inference, stronger tuning, and broader public-checkpoint comparisons.

## Key numbers

- Model: 1,173,248 params; corpus 1.19M words; 5,000 steps, 13 min; final ppl 77, masked acc 29.1%.
- SST-2 accuracy (n=1000/6920): TF-IDF LR 70.5/81.3; random init 68.2/78.5; +MLM 65.7/77.5; fastText 70.2/71.2; VADER 69.4 (any n); BERT-Tiny 75.0/81.7; DistilBERT 86.5/90.4.
- MLM - random init: -0.3, -4.5*, -2.5*, -2.3*, -1.0 (n=100...6920).
- Ambiguity (n=1000): strong -4.7* [-7.2,-2.2]; weak -1.3 [-3.4,+0.8]; interaction -3.5* [-6.7,-0.3].
- Ambiguity AUROC (low conf -> weak polarity): 0.545-0.622, best = char TF-IDF at n=6920.
- ECE at n=1000: TF-IDF 3.1-6.4%; transformers 22.9-25.6%.
- Label-equivalence: VADER ~840; +MLM ours ~450 (n=1000); DistilBERT >6,920 (n=1000).
