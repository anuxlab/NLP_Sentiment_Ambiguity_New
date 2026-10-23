# DynaSent pilot results

- Round: `round1`
- Backend: `tfidf`
- Budget: `100`
- Evaluation rows: `3600`
- Seeds: `[0, 1, 2]`

> Pilot results are exploratory, not final research claims. The test subset is fixed by seed 20261014; use the full test split and multiple seeds for final evaluation.

## Metrics

| Condition | Seed | Train rows | Null-gold train rows | Train seconds | JS divergence ↓ | Soft Brier ↓ | Accuracy | Macro-F1 | ECE ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| majority_ce | 0 | 100 | 0 | 1.68 | 0.3848 | 0.5068 | 0.34694444444444444 | 0.164060016999837 | 0.08415654001136622 |
| human_soft_targets | 0 | 100 | 0 | 0.11 | 0.3837 | 0.4992 | 0.36277777777777775 | 0.19474947782068908 | 0.04232270595100191 |
| label_smoothing | 0 | 100 | 0 | 0.08 | 0.3851 | 0.5057 | 0.3491666666666667 | 0.16736612246705307 | 0.07717145037319925 |
| majority_ce | 1 | 100 | 0 | 0.09 | 0.3941 | 0.5293 | 0.3386111111111111 | 0.14227910001229763 | 0.12963321978019343 |
| human_soft_targets | 1 | 100 | 0 | 0.08 | 0.3878 | 0.5104 | 0.3458333333333333 | 0.1618207476745432 | 0.08419506587915952 |
| label_smoothing | 1 | 100 | 0 | 0.19 | 0.3939 | 0.5267 | 0.3411111111111111 | 0.14669522869470997 | 0.12010105400449698 |
| majority_ce | 2 | 100 | 0 | 0.13 | 0.3890 | 0.5036 | 0.37027777777777776 | 0.2121123292623597 | 0.0473939524259832 |
| human_soft_targets | 2 | 100 | 0 | 0.10 | 0.3907 | 0.5027 | 0.37277777777777776 | 0.21797255481067507 | 0.03892822533845903 |
| label_smoothing | 2 | 100 | 0 | 0.09 | 0.3896 | 0.5032 | 0.37333333333333335 | 0.2166264402950867 | 0.04019187333683173 |
