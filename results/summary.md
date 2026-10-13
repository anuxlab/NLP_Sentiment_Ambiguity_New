# Results summary

Pre-training: final held-out MLM loss 4.349 (ppl 77), masked-token acc 29.1% after 5000 steps, 13.1 min.


## SST-2 test accuracy (%)
| model                         | 100          | 300          | 1000         | 3000         | 6920         |
|:------------------------------|:-------------|:-------------|:-------------|:-------------|:-------------|
| TF-IDF + LogReg               | 57.9$\pm$1.7 | 63.2$\pm$1.7 | 70.5$\pm$0.7 | 77.5$\pm$1.0 | 81.3$\pm$0.0 |
| TF-IDF + NaiveBayes           | 58.6$\pm$1.6 | 64.5$\pm$0.4 | 71.6$\pm$0.7 | 78.6$\pm$0.8 | 81.9$\pm$0.0 |
| Tiny-BERT (random init)       | 57.0$\pm$0.7 | 63.2$\pm$0.5 | 68.2$\pm$1.3 | 75.2$\pm$0.3 | 78.5$\pm$0.9 |
| Tiny-BERT (+MLM pre-training) | 56.8$\pm$0.7 | 58.7$\pm$3.8 | 65.7$\pm$0.9 | 72.9$\pm$0.1 | 77.5$\pm$1.1 |

## Robustness, n=1000
| model                         | clean        | typos 10%    | typos 30%    | word-drop 20%   | word-drop 40%   | OOD IMDb     |
|:------------------------------|:-------------|:-------------|:-------------|:----------------|:----------------|:-------------|
| TF-IDF + LogReg               | 70.5$\pm$0.7 | 69.7$\pm$0.4 | 68.4$\pm$1.5 | 69.3$\pm$1.0    | 67.2$\pm$1.1    | 67.9$\pm$1.3 |
| TF-IDF + NaiveBayes           | 71.6$\pm$0.7 | 70.7$\pm$1.3 | 69.1$\pm$0.1 | 70.7$\pm$1.2    | 68.7$\pm$0.5    | 69.7$\pm$2.0 |
| Tiny-BERT (random init)       | 68.2$\pm$1.3 | 67.0$\pm$1.0 | 64.6$\pm$0.3 | 67.2$\pm$1.5    | 64.8$\pm$1.7    | 58.6$\pm$4.2 |
| Tiny-BERT (+MLM pre-training) | 65.7$\pm$0.9 | 64.0$\pm$1.2 | 60.9$\pm$1.2 | 64.7$\pm$0.9    | 62.1$\pm$1.0    | 55.5$\pm$0.7 |

## Robustness, n=6920
| model                         | clean        | typos 10%    | typos 30%    | word-drop 20%   | word-drop 40%   | OOD IMDb     |
|:------------------------------|:-------------|:-------------|:-------------|:----------------|:----------------|:-------------|
| TF-IDF + LogReg               | 81.3$\pm$0.0 | 80.0$\pm$0.0 | 77.5$\pm$0.0 | 77.9$\pm$0.0    | 75.7$\pm$0.0    | 72.5$\pm$0.0 |
| TF-IDF + NaiveBayes           | 81.9$\pm$0.0 | 80.9$\pm$0.0 | 79.1$\pm$0.0 | 79.0$\pm$0.0    | 75.6$\pm$0.0    | 75.5$\pm$0.0 |
| Tiny-BERT (random init)       | 78.5$\pm$0.9 | 76.5$\pm$1.0 | 73.6$\pm$0.8 | 77.0$\pm$0.4    | 72.9$\pm$0.4    | 68.1$\pm$0.7 |
| Tiny-BERT (+MLM pre-training) | 77.5$\pm$1.1 | 75.2$\pm$1.7 | 70.6$\pm$1.9 | 76.8$\pm$1.3    | 71.6$\pm$0.7    | 63.7$\pm$1.4 |

## Accuracy change vs clean (pp), full data
| model                         |   typos 10% |   typos 30% |   word-drop 20% |   word-drop 40% |
|:------------------------------|------------:|------------:|----------------:|----------------:|
| TF-IDF + LogReg               |        -1.4 |        -3.8 |            -3.4 |            -5.6 |
| TF-IDF + NaiveBayes           |        -1   |        -2.9 |            -2.9 |            -6.3 |
| Tiny-BERT (random init)       |        -2   |        -4.9 |            -1.5 |            -5.5 |
| Tiny-BERT (+MLM pre-training) |        -2.3 |        -6.9 |            -0.7 |            -5.9 |

## LR pilot (dev acc %)
| model                         |    n |   0.0002 |   0.0005 |   0.001 |   0.002 |   0.004 |
|:------------------------------|-----:|---------:|---------:|--------:|--------:|--------:|
| Tiny-BERT (random init)       |  300 |     63.3 |     63.5 |    63.3 |    62.3 |    63   |
| Tiny-BERT (random init)       | 3000 |     76.3 |     74.7 |    74.3 |    72.8 |    70.5 |
| Tiny-BERT (+MLM pre-training) |  300 |     63   |     62   |    61.6 |    62   |    59.2 |
| Tiny-BERT (+MLM pre-training) | 3000 |     74.2 |     74.8 |    74.8 |    71.7 |    62.6 |

## Pre-training steps ablation
|   steps | probe        | finetune     | imdb         |
|--------:|:-------------|:-------------|:-------------|
|       0 | 56.4$\pm$0.7 | 69.7$\pm$0.7 | 58.5$\pm$4.5 |
|     100 | 56.9$\pm$1.2 | 69.3$\pm$0.7 | 56.6$\pm$4.8 |
|     300 | 56.0$\pm$0.9 | 67.6$\pm$1.3 | 56.8$\pm$4.7 |
|    1000 | 58.4$\pm$0.2 | 64.2$\pm$2.5 | 59.0$\pm$2.9 |
|    2500 | 59.3$\pm$1.3 | 66.2$\pm$1.1 | 59.7$\pm$3.8 |
|    5000 | 59.4$\pm$1.1 | 65.7$\pm$0.9 | 55.5$\pm$0.7 |

## Efficiency
| model                     |   params M |   size MB |   lat ms |   p95 ms |   sent/s |   acc |   acc typo30 |
|:--------------------------|-----------:|----------:|---------:|---------:|---------:|------:|-------------:|
| Tiny-BERT scratch, FP32   |       1.17 |      4.71 |     1.03 |     1.18 |     4914 |  78.3 |         73.1 |
| Tiny-BERT +MLM, FP32      |       1.17 |      4.71 |     1.02 |     1.18 |     5073 |  76.1 |         68   |
| Tiny-BERT +MLM, INT8 dyn. |       1.17 |      3.54 |     1.71 |     1.99 |     3196 |  75.9 |         68   |
| TF-IDF + LogReg           |       0.08 |      2.72 |     0.31 |     0.34 |    48388 |  81.3 |         77.5 |

## Model families
| model                               | n=1000       | n=6920       | typo30       | drop20       | OOD          |
|:------------------------------------|:-------------|:-------------|:-------------|:-------------|:-------------|
| VADER lexicon (no task labels)      | 69.4$\pm$0.0 | 69.4$\pm$0.0 | 66.4$\pm$0.0 | 67.2$\pm$0.0 | 59.8$\pm$0.0 |
| TF-IDF + LogReg                     | 70.5$\pm$0.7 | 81.3$\pm$0.0 | 77.5$\pm$0.0 | 77.9$\pm$0.0 | 72.5$\pm$0.0 |
| TF-IDF + NaiveBayes                 | 71.6$\pm$0.7 | 81.9$\pm$0.0 | 79.1$\pm$0.0 | 79.0$\pm$0.0 | 75.5$\pm$0.0 |
| TF-IDF word + SVM                   | 70.5$\pm$0.7 | 80.5$\pm$0.0 | 77.2$\pm$0.0 | 77.2$\pm$0.0 | 73.2$\pm$0.0 |
| TF-IDF char n-gram + LogReg         | 71.9$\pm$0.8 | 81.5$\pm$0.0 | 78.7$\pm$0.0 | 78.4$\pm$0.0 | 70.8$\pm$0.0 |
| word2vec (same 1.2M words) + LogReg | 67.2$\pm$0.3 | 68.6$\pm$0.0 | 67.8$\pm$0.0 | 68.7$\pm$0.0 | 67.0$\pm$0.0 |
| fastText (same 1.2M words) + LogReg | 70.2$\pm$0.6 | 71.2$\pm$0.0 | 69.6$\pm$0.0 | 69.6$\pm$0.0 | 66.2$\pm$0.0 |
| Tiny-BERT (random init)             | 68.2$\pm$1.3 | 78.5$\pm$0.9 | 73.6$\pm$0.8 | 77.0$\pm$0.4 | 68.1$\pm$0.7 |
| Tiny-BERT (+MLM pre-training)       | 65.7$\pm$0.9 | 77.5$\pm$1.1 | 70.6$\pm$1.9 | 76.8$\pm$1.3 | 63.7$\pm$1.4 |

## Label-equivalence (TF-IDF+LR labels)
| model                               | n=1000   | n=6920   |
|:------------------------------------|:---------|:---------|
| VADER lexicon (no task labels)      | 840      | --       |
| TF-IDF + NaiveBayes                 | 1,190    | $>$6,920 |
| TF-IDF word + SVM                   | 1,010    | 5,720    |
| TF-IDF char n-gram + LogReg         | 1,260    | $>$6,920 |
| word2vec (same 1.2M words) + LogReg | 590      | 740      |
| fastText (same 1.2M words) + LogReg | 960      | 1,120    |
| Tiny-BERT (random init)             | 690      | 3,730    |
| Tiny-BERT (+MLM pre-training)       | 450      | 3,000    |
| bert\_uncased\_L-2\_H-128\_A-2      | 2,040    | $>$6,920 |
| distilbert-base-uncased             | $>$6,920 | $>$6,920 |

## Ambiguity strata, n=1000
| model                               |   acc strong |   acc weak |   gap |   AUROC | ECE   |   acc@50% |
|:------------------------------------|-------------:|-----------:|------:|--------:|:------|----------:|
| TF-IDF + LogReg                     |         77.3 |       66.4 |  10.9 |   0.582 | 3.1   |      82.2 |
| TF-IDF char n-gram + LogReg         |         79.8 |       67.3 |  12.5 |   0.579 | 6.4   |      83.4 |
| TF-IDF word + SVM                   |         77.3 |       66.6 |  10.7 |   0.581 | --    |      82.2 |
| VADER lexicon (no task labels)      |         77.4 |       64.7 |  12.8 |   0.559 | --    |      78.5 |
| fastText (same 1.2M words) + LogReg |         75   |       67.3 |   7.6 |   0.552 | 5.5   |      81   |
| word2vec (same 1.2M words) + LogReg |         72.2 |       64.2 |   8   |   0.545 | 4.8   |      79.2 |
| Tiny-BERT (random init)             |         73.6 |       65   |   8.7 |   0.561 | 25.6  |      78.8 |
| Tiny-BERT (+MLM pre-training)       |         68.9 |       63.7 |   5.2 |   0.545 | 22.9  |      74.8 |

## Ambiguity strata, n=6920
| model                               |   acc strong |   acc weak |   gap |   AUROC | ECE   |   acc@50% |
|:------------------------------------|-------------:|-----------:|------:|--------:|:------|----------:|
| TF-IDF + LogReg                     |         88.1 |       77.3 |  10.7 |   0.617 | 4.4   |      94   |
| TF-IDF char n-gram + LogReg         |         88.9 |       77.2 |  11.8 |   0.622 | 2.8   |      92.3 |
| TF-IDF word + SVM                   |         87.8 |       76.1 |  11.6 |   0.611 | --    |      93.6 |
| VADER lexicon (no task labels)      |         77.4 |       64.7 |  12.8 |   0.559 | --    |      78.5 |
| fastText (same 1.2M words) + LogReg |         77.4 |       67.5 |  10   |   0.553 | 3.2   |      84.2 |
| word2vec (same 1.2M words) + LogReg |         75.5 |       64.5 |  11   |   0.554 | 3.7   |      81.9 |
| Tiny-BERT (random init)             |         85.7 |       74.2 |  11.5 |   0.604 | 8.2   |      91.6 |
| Tiny-BERT (+MLM pre-training)       |         83.1 |       74.1 |   9   |   0.604 | 7.6   |      89.3 |

## Ambiguity bootstrap
|    n | cmp                       | strong              | weak               | strong-weak        | contrast             | no contrast        |
|-----:|:--------------------------|:--------------------|:-------------------|:-------------------|:---------------------|:-------------------|
| 1000 | +MLM $-$ random init      | -4.7 [-7.2, -2.2]*  | -1.3 [-3.4, +0.8]  | -3.5 [-6.7, -0.3]* | -4.1 [-8.1, +0.2]    | -2.2 [-4.0, -0.4]* |
| 1000 | +MLM $-$ TF-IDF LR        | -8.4 [-11.3, -5.6]* | -2.7 [-5.0, -0.4]* | -5.7 [-9.3, -2.1]* | -4.9 [-9.2, -0.3]*   | -4.8 [-6.8, -2.8]* |
| 1000 | random init $-$ TF-IDF LR | -3.6 [-5.8, -1.3]*  | -1.4 [-3.4, +0.4]  | -2.2 [-5.1, +0.6]  | -0.9 [-4.1, +2.5]    | -2.6 [-4.1, -1.0]* |
| 1000 | random init $-$ fastText  | -1.3 [-4.3, +1.8]   | -2.4 [-4.8, +0.1]  | +1.0 [-2.8, +4.8]  | +2.7 [-1.7, +7.0]    | -3.0 [-5.1, -0.9]* |
| 1000 | +MLM $-$ fastText         | -6.0 [-9.1, -2.9]*  | -3.6 [-6.3, -0.9]* | -2.4 [-6.3, +1.5]  | -1.4 [-6.2, +3.8]    | -5.2 [-7.4, -2.9]* |
| 6920 | +MLM $-$ random init      | -2.6 [-4.8, -0.2]*  | -0.1 [-2.2, +2.0]  | -2.5 [-5.6, +0.3]  | -1.2 [-4.8, +2.7]    | -1.0 [-2.6, +0.6]  |
| 6920 | +MLM $-$ TF-IDF LR        | -4.9 [-7.6, -2.4]*  | -3.2 [-5.8, -0.6]* | -1.7 [-5.4, +2.0]  | -1.3 [-6.0, +3.7]    | -4.4 [-6.4, -2.4]* |
| 6920 | random init $-$ TF-IDF LR | -2.4 [-4.7, -0.0]*  | -3.1 [-5.6, -0.8]* | +0.8 [-2.5, +4.0]  | -0.1 [-4.3, +3.9]    | -3.5 [-5.4, -1.6]* |
| 6920 | random init $-$ fastText  | +8.3 [+5.1, +11.6]* | +6.7 [+4.0, +9.7]* | +1.5 [-2.6, +5.7]  | +10.3 [+5.3, +15.0]* | +6.6 [+4.2, +9.0]* |
| 6920 | +MLM $-$ fastText         | +5.7 [+2.4, +9.0]*  | +6.6 [+3.9, +9.5]* | -0.9 [-5.3, +3.3]  | +9.1 [+3.8, +14.3]*  | +5.7 [+3.3, +8.0]* |

## Consensus vs polarity strength
|    n | group                         |   count | weak share   |
|-----:|:------------------------------|--------:|:-------------|
| 1000 | hard ($<$25% of runs correct) |      98 | 79.6%        |
| 1000 | middle                        |     846 | 68.4%        |
| 1000 | easy ($geq$75% correct)       |     877 | 55.4%        |
| 1000 | all sentences (base rate)     |    1821 | 62.8%        |
| 6920 | hard ($<$25% of runs correct) |     124 | 81.5%        |
| 6920 | middle                        |     490 | 73.1%        |
| 6920 | easy ($geq$75% correct)       |    1207 | 56.7%        |
| 6920 | all sentences (base rate)     |    1821 | 62.8%        |

## Public HF checkpoints
| model                      |   params M |    n | clean        | typo30       | drop20       | imdb         |
|:---------------------------|-----------:|-----:|:-------------|:-------------|:-------------|:-------------|
| bert_uncased_L-2_H-128_A-2 |        4.4 | 1000 | 75.0$\pm$2.0 | 70.3$\pm$0.6 | 72.4$\pm$1.9 | 68.3$\pm$0.6 |
| bert_uncased_L-2_H-128_A-2 |        4.4 | 6920 | 81.7$\pm$0.0 | 74.6$\pm$0.5 | 78.9$\pm$0.2 | 69.1$\pm$1.4 |
| distilbert-base-uncased    |       66.4 | 1000 | 86.5$\pm$0.9 | 80.5$\pm$0.6 | 82.4$\pm$1.6 | 69.0$\pm$6.3 |
| distilbert-base-uncased    |       66.4 | 6920 | 90.4$\pm$0.2 | 83.9$\pm$0.3 | 85.5$\pm$0.1 | 72.8$\pm$1.6 |

## Cross-domain transfer matrix
| model                         | trained_on       | tested_on        |   acc |
|:------------------------------|:-----------------|:-----------------|------:|
| Tiny-BERT (+MLM pre-training) | SST-2 (movies)   | SST-2 (movies)   |  77.5 |
| Tiny-BERT (+MLM pre-training) | SST-2 (movies)   | Finance (news)   |  51.8 |
| Tiny-BERT (+MLM pre-training) | SST-2 (movies)   | Airline (tweets) |  68.3 |
| Tiny-BERT (+MLM pre-training) | Finance (news)   | SST-2 (movies)   |  54.1 |
| Tiny-BERT (+MLM pre-training) | Finance (news)   | Finance (news)   |  86.5 |
| Tiny-BERT (+MLM pre-training) | Finance (news)   | Airline (tweets) |  47.2 |
| Tiny-BERT (+MLM pre-training) | Airline (tweets) | SST-2 (movies)   |  55.6 |
| Tiny-BERT (+MLM pre-training) | Airline (tweets) | Finance (news)   |  33.4 |
| Tiny-BERT (+MLM pre-training) | Airline (tweets) | Airline (tweets) |  91.5 |
| Tiny-BERT (random init)       | SST-2 (movies)   | SST-2 (movies)   |  78.5 |
| Tiny-BERT (random init)       | SST-2 (movies)   | Finance (news)   |  52.3 |
| Tiny-BERT (random init)       | SST-2 (movies)   | Airline (tweets) |  73.8 |
| Tiny-BERT (random init)       | Finance (news)   | SST-2 (movies)   |  53   |
| Tiny-BERT (random init)       | Finance (news)   | Finance (news)   |  87.2 |
| Tiny-BERT (random init)       | Finance (news)   | Airline (tweets) |  35.6 |
| Tiny-BERT (random init)       | Airline (tweets) | SST-2 (movies)   |  56.5 |
| Tiny-BERT (random init)       | Airline (tweets) | Finance (news)   |  35.4 |
| Tiny-BERT (random init)       | Airline (tweets) | Airline (tweets) |  91.6 |
| TF-IDF + LogReg               | SST-2 (movies)   | SST-2 (movies)   |  81.3 |
| TF-IDF + LogReg               | SST-2 (movies)   | Finance (news)   |  62.7 |
| TF-IDF + LogReg               | SST-2 (movies)   | Airline (tweets) |  75.3 |
| TF-IDF + LogReg               | Finance (news)   | SST-2 (movies)   |  52.2 |
| TF-IDF + LogReg               | Finance (news)   | Finance (news)   |  87.3 |
| TF-IDF + LogReg               | Finance (news)   | Airline (tweets) |  26.8 |
| TF-IDF + LogReg               | Airline (tweets) | SST-2 (movies)   |  53   |
| TF-IDF + LogReg               | Airline (tweets) | Finance (news)   |  32   |
| TF-IDF + LogReg               | Airline (tweets) | Airline (tweets) |  92   |

## finance: real ambiguity signal (real_annotator_agreement_tier)
strong=175, weak=57, total=394
|    n | model           |   acc_strong |   acc_weak |   auroc | ece   |
|-----:|:----------------|-------------:|-----------:|--------:|:------|
|  300 | tfidf_lr        |         80   |       74.9 |   0.457 | 7.4   |
|  300 | tfidf_nb        |         79.8 |       77.8 |   0.466 | 6.4   |
|  300 | tiny_scratch    |         81.5 |       80.7 |   0.439 | 17.6  |
|  300 | tiny_mlm        |         76.2 |       76   |   0.41  | 19.6  |
|  300 | vader_lexicon   |         67.4 |       52.6 |   0.613 | --    |
|  300 | tfidf_svm       |         80.8 |       74.9 |   0.466 | --    |
|  300 | tfidf_char_lr   |         81.7 |       78.9 |   0.405 | 7.4   |
|  300 | w2v_avg_lr      |         63.6 |       69   |   0.368 | 6.2   |
|  300 | fasttext_avg_lr |         63.6 |       69.6 |   0.307 | 6.8   |
| 1377 | tfidf_lr        |         90.9 |       84.2 |   0.584 | 4.7   |
| 1377 | tfidf_nb        |         84.6 |       78.9 |   0.471 | 7.5   |
| 1377 | tiny_scratch    |         90.7 |       81.9 |   0.513 | 10.8  |
| 1377 | tiny_mlm        |         89.7 |       83.6 |   0.454 | 10.3  |
| 1377 | vader_lexicon   |         67.4 |       52.6 |   0.613 | --    |
| 1377 | tfidf_svm       |         91.4 |       84.2 |   0.585 | --    |
| 1377 | tfidf_char_lr   |         89.7 |       84.2 |   0.503 | 6.5   |
| 1377 | w2v_avg_lr      |         68   |       71.9 |   0.325 | 5.4   |
| 1377 | fasttext_avg_lr |         68   |       68.4 |   0.274 | 7.6   |

## airline: real ambiguity signal (real_crowdworker_confidence)
strong=1779, weak=138, total=2407
|    n | model           |   acc_strong |   acc_weak |   auroc | ece   |
|-----:|:----------------|-------------:|-----------:|--------:|:------|
| 1000 | tfidf_lr        |         92.6 |       75.8 |   0.653 | 4.1   |
| 1000 | tfidf_nb        |         90.8 |       73.9 |   0.653 | 6.9   |
| 1000 | tiny_scratch    |         92.4 |       77.8 |   0.637 | 8.4   |
| 1000 | tiny_mlm        |         91.8 |       74.2 |   0.622 | 11.3  |
| 1000 | vader_lexicon   |         66.5 |       58.7 |   0.505 | --    |
| 1000 | tfidf_svm       |         92.9 |       75.8 |   0.653 | --    |
| 1000 | tfidf_char_lr   |         93.5 |       80   |   0.641 | 2.4   |
| 1000 | w2v_avg_lr      |         90.6 |       74.6 |   0.61  | 2.9   |
| 1000 | fasttext_avg_lr |         91.6 |       75.4 |   0.601 | 2.6   |
| 8125 | tfidf_lr        |         96   |       77.5 |   0.685 | 2.6   |
| 8125 | tfidf_nb        |         93.4 |       76.1 |   0.678 | 5.4   |
| 8125 | tiny_scratch    |         95.5 |       79   |   0.67  | 5.2   |
| 8125 | tiny_mlm        |         95.3 |       79   |   0.659 | 4.4   |
| 8125 | vader_lexicon   |         66.5 |       58.7 |   0.505 | --    |
| 8125 | tfidf_svm       |         95.9 |       78.3 |   0.686 | --    |
| 8125 | tfidf_char_lr   |         96.3 |       79   |   0.67  | 1.3   |
| 8125 | w2v_avg_lr      |         92.1 |       75.4 |   0.614 | 1.4   |
| 8125 | fasttext_avg_lr |         92.5 |       74.6 |   0.611 | 1.1   |