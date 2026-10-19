import zipfile, json, numpy as np, pandas as pd, math
from pathlib import Path
from collections import defaultdict
from sklearn.metrics import f1_score

src='/mnt/data/dynasent-multiseed-38672586726.zip'
out=Path('/mnt/data/frozen_analysis_phase'); out.mkdir(exist_ok=True)
with zipfile.ZipFile(src) as z:
    records=[]
    for round_name in ['round1','round2']:
        for line in z.read(f'{round_name}/predictions.jsonl').decode().splitlines():
            records.append(json.loads(line))

df=pd.DataFrame(records)
conditions=['majority_ce','human_soft_targets','label_smoothing','human_soft_targets_null_augmented']
labels=['negative','neutral','positive','mixed']; lab2i={x:i for i,x in enumerate(labels)}
# Validate expected dimensions and alignment
assert set(df['round'])=={'round1','round2'}
assert set(df['condition'])==set(conditions)
assert set(df['seed'])=={0,1,2,3,4}
for rnd,n in [('round1',3600),('round2',720)]:
    for seed in range(5):
        for cond in conditions:
            x=df[(df['round']==rnd)&(df['seed']==seed)&(df['condition']==cond)]
            assert len(x)==n,(rnd,seed,cond,len(x))
            assert x.text_id.nunique()==n
# build arrays indexed round, seed, condition with sorted text_id order and assert alignment
arrays={}
for rnd in ['round1','round2']:
    base=df[(df['round']==rnd)&(df.seed==0)&(df.condition=='majority_ce')].sort_values('text_id')
    ids=base.text_id.to_numpy()
    human=np.array(base.human_distribution.tolist(),float)
    gold=base.gold_label.to_numpy()
    y=np.array([lab2i[x] for x in gold],int)
    entropy=-(np.where(human>0,human*np.log2(np.where(human>0,human,1)),0)).sum(axis=1)
    tiers=np.where(entropy<1e-12,'unanimous','vote_disagreement')
    arrays[rnd]={'ids':ids,'human':human,'y':y,'entropy':entropy,'tiers':tiers,'pred':{},'js':{}}
    for seed in range(5):
        for cond in conditions:
            x=df[(df['round']==rnd)&(df.seed==seed)&(df.condition==cond)].sort_values('text_id')
            assert np.array_equal(ids,x.text_id.to_numpy()), f'IDs mismatch {rnd} {seed} {cond}'
            arrays[rnd]['pred'][(seed,cond)]=np.array([lab2i[a] for a in x.predicted_label],int)
            arrays[rnd]['js'][(seed,cond)]=x.js_divergence_bits.to_numpy(float)

# metrics functions
def macro_f1(y, pred):
    # NumPy-only macro-F1, fixed four-class label set.
    cm=np.bincount(y*4+pred,minlength=16).reshape(4,4)
    tp=np.diag(cm).astype(float); fp=cm.sum(axis=0)-tp; fn=cm.sum(axis=1)-tp
    den=2*tp+fp+fn
    f=np.divide(2*tp,den,out=np.zeros_like(tp),where=den>0)
    return float(f.mean())
def acc(y,pred): return float(np.mean(y==pred))

def percentile_ci(a): return np.quantile(a,[.025,.975])

rng=np.random.default_rng(20261015)
B=2000
bootstrap_rows=[]
point_rows=[]
# overall and pre-specified disagreement tiers; primary overall JS for human_soft_targets vs CE per round
for rnd in ['round1','round2']:
    a=arrays[rnd]; n=len(a['y'])
    strata=['all','unanimous','vote_disagreement']
    for tier in strata:
        mask=np.ones(n,bool) if tier=='all' else a['tiers']==tier
        idx0=np.flatnonzero(mask); nn=len(idx0)
        y=a['y'][idx0]
        for cand in ['human_soft_targets','label_smoothing','human_soft_targets_null_augmented']:
            # per seed point differences oriented as raw candidate-baseline for JS; macroF1/accuracy raw
            point_by_metric={m:[] for m in ['js_divergence','macro_f1','accuracy']}
            seed_diff={m:[] for m in point_by_metric}
            # paired bootstrap replicate by resampling same test examples within the tier, across all seeds/conditions
            reps={m:np.empty(B,float) for m in point_by_metric}
            for b in range(B):
                sample=rng.choice(idx0,size=nn,replace=True)
                ys=a['y'][sample]
                vals={m:[] for m in point_by_metric}
                for seed in range(5):
                    pc=a['pred'][(seed,cand)][sample]; pb=a['pred'][(seed,'majority_ce')][sample]
                    jc=a['js'][(seed,cand)][sample]; jb=a['js'][(seed,'majority_ce')][sample]
                    vals['js_divergence'].append(float(jc.mean()-jb.mean()))
                    vals['macro_f1'].append(macro_f1(ys,pc)-macro_f1(ys,pb))
                    vals['accuracy'].append(acc(ys,pc)-acc(ys,pb))
                for m in vals: reps[m][b]=float(np.mean(vals[m]))
            # point estimates, based on full selected subset average seed-level differences
            for seed in range(5):
                pc=a['pred'][(seed,cand)][idx0]; pb=a['pred'][(seed,'majority_ce')][idx0]
                jc=a['js'][(seed,cand)][idx0]; jb=a['js'][(seed,'majority_ce')][idx0]
                seed_diff['js_divergence'].append(float(jc.mean()-jb.mean()))
                seed_diff['macro_f1'].append(macro_f1(y,pc)-macro_f1(y,pb))
                seed_diff['accuracy'].append(acc(y,pc)-acc(y,pb))
            for m in point_by_metric:
                point=float(np.mean(seed_diff[m])); lo,hi=percentile_ci(reps[m])
                # two-sided bootstrap tail probability relative to zero; simple unadjusted exploratory estimate
                p=min(1.0,2*min((np.sum(reps[m]<=0)+1)/(B+1),(np.sum(reps[m]>=0)+1)/(B+1)))
                point_rows.append({'round':rnd,'tier':tier,'candidate':cand,'baseline':'majority_ce','metric':m,'n_examples':nn,'n_seeds':5,'mean_difference_candidate_minus_baseline':point,'seed_sd_of_differences':float(np.std(seed_diff[m],ddof=1)),'bootstrap_ci95_low':float(lo),'bootstrap_ci95_high':float(hi),'bootstrap_p_unadjusted':float(p),'bootstrap_replicates':B,'seed_fixed':True})

res=pd.DataFrame(point_rows)
# primary family: human soft-target vs CE, overall JS, two rounds; Holm adjustment for the two bootstrap p estimates
sel=(res.tier=='all')&(res.candidate=='human_soft_targets')&(res.metric=='js_divergence')
primary=res[sel].copy()
pvals=primary.bootstrap_p_unadjusted.to_numpy(); order=np.argsort(pvals); adj=np.empty_like(pvals); running=0
m=len(pvals)
for rank,idx in enumerate(order):
    val=(m-rank)*pvals[idx]; running=max(running,val); adj[idx]=min(1,running)
for ix,val in zip(primary.index,adj): res.loc[ix,'holm_adjusted_p_primary_family']=float(val)
res.to_csv(out/'paired_bootstrap_all_tiers.csv',index=False)

# Disagreement tier descriptive means per method (across examples and seeds), accuracy and JS, entropy counts
summary=[]
for rnd in ['round1','round2']:
 a=arrays[rnd]
 for tier in ['unanimous','vote_disagreement']:
  mask=a['tiers']==tier
  for cond in conditions:
   js=[]; accs=[]; f1s=[]
   for seed in range(5):
    js.append(float(a['js'][(seed,cond)][mask].mean()))
    accs.append(acc(a['y'][mask],a['pred'][(seed,cond)][mask]))
    f1s.append(macro_f1(a['y'][mask],a['pred'][(seed,cond)][mask]))
   summary.append({'round':rnd,'tier':tier,'condition':cond,'n_examples_per_seed':int(mask.sum()),'mean_human_entropy_bits':float(a['entropy'][mask].mean()),'mean_js_across_seeds':float(np.mean(js)),'sd_js_across_seeds':float(np.std(js,ddof=1)),'mean_macro_f1_across_seeds':float(np.mean(f1s)),'sd_macro_f1_across_seeds':float(np.std(f1s,ddof=1)),'mean_accuracy_across_seeds':float(np.mean(accs)),'sd_accuracy_across_seeds':float(np.std(accs,ddof=1))})
pd.DataFrame(summary).to_csv(out/'disagreement_tier_metrics.csv',index=False)

# Plan frozen for next experiments, explicit current-analysis distinction
plan='''# Frozen analysis plan for the next experiment cycle

**Status:** This protocol is frozen prospectively for the next experiment cycle. The current five-seed results have already been inspected; analyses of those results are exploratory, not preregistered or confirmatory.

## Research question
Under a scarce label budget, does training against human annotator vote distributions improve alignment between model output distributions and held-out human vote distributions, compared with majority-label cross-entropy? Separately, does adding eligible null-majority training examples help?

## Primary outcome and contrast
- Primary metric: mean per-example Jensen–Shannon divergence (base 2, in bits) between predicted class probabilities and held-out human vote distributions; lower is better.
- Primary contrast: `human_soft_targets` minus `majority_ce`, evaluated separately for DynaSent Round 1 and Round 2.
- Report both round-specific effect estimates and 95% paired test-example bootstrap confidence intervals. Adjust the two primary round-level tests with Holm's method.
- Bootstrap holds the five trained seeds fixed and resamples test examples with replacement. It quantifies test-example sampling uncertainty conditional on these models; it does not replace variability across independently trained seeds.

## Secondary and exploratory outcomes
- Secondary outcome: macro-F1 on examples with a valid majority gold label; accuracy reported alongside it.
- Secondary baseline: `label_smoothing` versus `majority_ce`, reported separately by round.
- Separate exploratory condition: `human_soft_targets_null_augmented` versus `majority_ce`. It uses a different training-data composition (50 null-majority + 50 non-null-majority examples at budget 100), so it is not interpreted as a matched-data causal comparison.
- Secondary subgroup analysis: unanimous vs non-unanimous human vote distributions, using entropy derived from human votes only. These are the only two observed tiers in the current dataset view (unanimous entropy 0; non-unanimous observed distribution 4:1, entropy 0.7219 bits). Do not invent empty moderate/high bins.
- Calibration metrics (soft Brier and ECE) are descriptive secondary metrics. ECE is bin-sensitive and will not be the primary outcome.

## Multiplicity and reporting
- Holm adjustment across the two primary JS-divergence round comparisons.
- Treat all other metrics, conditions, tiers, and interaction-like comparisons as secondary/exploratory; if inferential claims are made across a family of them, apply a declared multiplicity correction and label the family.
- Always show Round 1 and Round 2 separately; do not pool them into a single headline result.
- Report per-seed means/SDs alongside bootstrap intervals; do not conflate seed variability with test-example uncertainty.

## Data and reproducibility safeguards
- Keep train overlap/dedup filtering limited to the effective training view; preserve dev/test splits.
- Freeze label order as negative, neutral, positive, mixed.
- Never infer human disagreement from model confidence; derive tiers from annotator vote distributions.
- Do not tune hyperparameters, thresholds, or null fraction against the final test set.
- Preserve text IDs, per-example predictions, per-seed metrics, and training manifests.
- Before any confirmatory claim, verify that audit metadata lists all four conditions; the current artifact's `conditions` array omits the null-augmented condition even though metrics/manifests include it.

## Next experiment cycle
1. Fix condition metadata and add a test asserting that the declared conditions equal the observed metric/manifest conditions.
2. Re-run CI and the same five-seed experiment to confirm the fix is metadata-only and outputs remain consistent.
3. Analyze the frozen primary and secondary comparisons using the outputs above.
4. Only after that, extend to a small Transformer; preserve the same split, label budgets, metrics, seeds where feasible, and evaluation protocol.
'''
(out/'FROZEN_ANALYSIS_PLAN.md').write_text(plan)

# concise findings report
prim=res[(res.tier=='all') & (res.metric=='js_divergence') & (res.candidate.isin(['human_soft_targets','human_soft_targets_null_augmented']))]
lines=['# Paired uncertainty and disagreement analysis','', 'Bootstrap: 2,000 paired test-example resamples per round/tier; the same resampled example indices are used across all five seeds and candidate/baseline pairs. Estimates are conditional on the five trained seeds.','', '## Primary contrast: human soft targets vs majority CE (JS divergence)','', '| Round | Mean difference (candidate − baseline) | 95% paired bootstrap CI | Holm-adjusted p (two-round primary family) |','|---|---:|---:|---:|']
for _,r in res[(res.tier=='all')&(res.candidate=='human_soft_targets')&(res.metric=='js_divergence')].iterrows():
 lines.append(f"| {r['round']} | {r['mean_difference_candidate_minus_baseline']:.5f} | [{r['bootstrap_ci95_low']:.5f}, {r['bootstrap_ci95_high']:.5f}] | {r.get('holm_adjusted_p_primary_family',float('nan')):.4f} |")
lines += ['', 'Negative JS differences favour the candidate. The intervals are conditional on fixed trained models and should not be presented as seed-level uncertainty. Bootstrap p-values are approximate and are supplementary to the intervals.', '', '## Disagreement tiers', '', 'The current vote distributions contain two observed tiers only: unanimous and non-unanimous. The subgroup metrics are descriptive; tier sizes differ by round. See `disagreement_tier_metrics.csv` and `paired_bootstrap_all_tiers.csv`.', '', '## Caveat', '', 'Current-run analyses are exploratory because the outputs were already inspected. Use `FROZEN_ANALYSIS_PLAN.md` prospectively for the next experiment cycle. The audit condition-list mismatch remains a reproducibility issue to fix.']
(out/'PAIRED_UNCERTAINTY_REPORT.md').write_text('\n'.join(lines)+'\n')

# package
with zipfile.ZipFile(out/'analysis_bundle.zip','w',zipfile.ZIP_DEFLATED) as z:
 for f in out.iterdir():
  if f.is_file() and f.name!='analysis_bundle.zip': z.write(f,arcname=f.name)
print('records',len(df),'analysis_rows',len(res))
print((out/'PAIRED_UNCERTAINTY_REPORT.md').read_text())
print('\nDisagreement tier summary:')
print(pd.DataFrame(summary).to_string(index=False))
