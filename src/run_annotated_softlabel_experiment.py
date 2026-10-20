"""Run matched and ambiguity-inclusive TF-IDF experiments on vote-distribution datasets.

Input JSONL schema (one example per line):
  {"text": "...", "split": "train|dev|test", "votes": {"negative": 2, "neutral": 1, "positive": 0}}
Optional `gold_label` can be provided; otherwise unique majority is inferred. Vote keys
must use the ordered classes supplied with --labels. Ties are retained for soft training
and distributional evaluation, but omitted from hard-label metrics.
"""
from __future__ import annotations
import argparse, json, random, time
from pathlib import Path
import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression


def parse_rows(path, labels):
    rows=[]
    for i,line in enumerate(Path(path).read_text(encoding='utf-8').splitlines(),1):
        if not line.strip(): continue
        r=json.loads(line); text=str(r.get('text','')).strip(); split=r.get('split')
        if not text or split not in {'train','dev','test'}: raise ValueError(f'line {i}: text and split=train|dev|test required')
        votes=r.get('votes')
        if votes is None and 'target_probs' in r:
            probs=np.asarray(r['target_probs'],dtype=float)
        elif isinstance(votes,dict):
            probs=np.asarray([float(votes.get(label,0)) for label in labels],dtype=float)
        else: raise ValueError(f'line {i}: provide votes mapping or target_probs')
        if probs.shape != (len(labels),) or not np.isfinite(probs).all() or (probs<0).any() or probs.sum()<=0:
            raise ValueError(f'line {i}: invalid vote distribution')
        probs=probs/probs.sum(); maxp=probs.max(); winners=np.flatnonzero(np.isclose(probs,maxp))
        gold=r.get('gold_label')
        # A tie in observed vote counts must remain unlabeled for hard-label metrics,
        # even if an upstream file accidentally supplies an arbitrary hard label.
        if len(winners)>1: gold=None
        elif gold not in labels: gold=labels[int(winners[0])]
        rows.append({'text':text,'split':split,'target_probs':probs,'gold_label':gold,'is_tie':len(winners)>1})
    for s in ('train','dev','test'):
        if not any(r['split']==s for r in rows): raise ValueError(f'missing split: {s}')
    sets={s:{r['text'].casefold().strip() for r in rows if r['split']==s} for s in ('train','dev','test')}
    if (sets['train']&sets['dev']) or (sets['train']&sets['test']) or (sets['dev']&sets['test']): raise ValueError('normalized text overlap across splits')
    return rows


def jsd(p,q):
    p=np.clip(p,1e-12,1); q=np.clip(q,1e-12,1); p=p/p.sum(axis=1,keepdims=True); q=q/q.sum(axis=1,keepdims=True); m=(p+q)/2
    return .5*np.sum(p*np.log2(p/m),axis=1)+.5*np.sum(q*np.log2(q/m),axis=1)


def fit(train, test, labels, condition, seed, smoothing=.1):
    vec=TfidfVectorizer(ngram_range=(1,2),min_df=1,max_features=30000,sublinear_tf=True)
    X=vec.fit_transform([r['text'] for r in train]); Xt=vec.transform([r['text'] for r in test])
    y=np.asarray([r['target_probs'] for r in train]); hard=np.asarray([labels.index(r['gold_label']) if r['gold_label'] in labels else -1 for r in train])
    if condition in ('majority_ce','label_smoothing'):
        keep=hard>=0; X=X[keep]; hard=hard[keep]
        # LogisticRegression has no native label smoothing; approximate smoothing by mixing
        # one-hot labels with uniform labels in a deterministic SGD soft-target model below.
        if condition=='majority_ce':
            model=LogisticRegression(max_iter=300,random_state=seed,C=1.0).fit(X,hard)
            pred=model.predict_proba(Xt)
        else:
            pred=_fit_soft_linear(X, y[keep], Xt, seed, smoothing)
    else:
        pred=_fit_soft_linear(X,y,Xt,seed,0.0)
    out=np.zeros((len(test),len(labels)))
    # Align sklearn classes to full label order.
    if condition=='majority_ce':
        for j,c in enumerate(model.classes_): out[:,int(c)]=pred[:,j]
        pred=out
    return pred


def _fit_soft_linear(X, targets, Xt, seed, smoothing):
    # Multinomial logistic model optimized directly against soft targets.
    torch.manual_seed(seed); torch.use_deterministic_algorithms(True, warn_only=True)
    x=torch.tensor(X.toarray(),dtype=torch.float32); xv=torch.tensor(Xt.toarray(),dtype=torch.float32)
    y=torch.tensor(targets,dtype=torch.float32)
    if smoothing: y=(1-smoothing)*y+smoothing/y.shape[1]
    W=torch.nn.Linear(x.shape[1],y.shape[1]); opt=torch.optim.AdamW(W.parameters(),lr=.04,weight_decay=.01)
    for _ in range(100):
        opt.zero_grad(); loss=-(y*torch.log_softmax(W(x),dim=1)).sum(1).mean(); loss.backward(); opt.step()
    with torch.no_grad(): return torch.softmax(W(xv),dim=1).numpy()


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--input',required=True); p.add_argument('--out-dir',required=True); p.add_argument('--labels',nargs='+',default=['negative','neutral','positive']); p.add_argument('--budgets',nargs='+',type=int,default=[25,50,100,200,300]); p.add_argument('--seeds',nargs='+',type=int,default=[0,1,2,3,4]); p.add_argument('--label-smoothing',type=float,default=.1); a=p.parse_args()
    rows=parse_rows(a.input,a.labels); train=[r for r in rows if r['split']=='train']; test=[r for r in rows if r['split']=='test']; out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True)
    records=[]; predictions=[]
    for budget in a.budgets:
        for seed in a.seeds:
            rng=np.random.RandomState(seed); non_tied=[r for r in train if r['gold_label'] in a.labels];
            if len(non_tied)<budget: raise ValueError(f'budget {budget} exceeds non-tied train rows ({len(non_tied)})')
            matched=[non_tied[i] for i in rng.choice(len(non_tied),budget,replace=False)]
            tie_rows=[r for r in train if r['gold_label'] not in a.labels]
            soft_pool=matched.copy(); extras=max(0,budget//3)
            if extras and tie_rows: soft_pool += [tie_rows[i] for i in rng.choice(len(tie_rows),min(extras,len(tie_rows)),replace=False)]
            # Soft-inclusive condition uses the same non-tied examples plus up to budget/3 tied rows;
            # report actual N and tie count rather than calling it a matched comparison.
            conditions=[('majority_ce',matched),('human_soft_targets',matched),('label_smoothing',matched),('human_soft_targets_tie_inclusive',soft_pool)]
            for name,subset in conditions:
                start=time.time(); pred=fit(subset,test,a.labels,name,seed,a.label_smoothing); elapsed=time.time()-start
                targets=np.stack([r['target_probs'] for r in test]); js=jsd(pred,targets); brier=np.mean(np.sum((pred-targets)**2,axis=1))
                hard_idx=[i for i,r in enumerate(test) if r['gold_label'] in a.labels]; ytrue=[a.labels.index(test[i]['gold_label']) for i in hard_idx]; yhat=pred[hard_idx].argmax(1)
                rec={'dataset':Path(a.input).stem,'seed':seed,'budget_requested':budget,'condition':name,'n_train':len(subset),'n_tied_train':sum(r['gold_label'] not in a.labels for r in subset),'n_test':len(test),'n_tied_test':sum(r['gold_label'] not in a.labels for r in test),'js_divergence_bits':float(js.mean()),'soft_brier_score':float(brier),'accuracy':float(accuracy_score(ytrue,yhat)) if hard_idx else None,'macro_f1':float(f1_score(ytrue,yhat,average='macro',zero_division=0)) if hard_idx else None,'train_time_s':elapsed}
                records.append(rec)
                for i,(r,pr) in enumerate(zip(test,pred)): predictions.append({'seed':seed,'budget_requested':budget,'condition':name,'text':r['text'],'target_probs':r['target_probs'].tolist(),'predicted_probs':pr.tolist(),'js_divergence_bits':float(js[i])})
                print(json.dumps(rec),flush=True)
    (out/'metrics.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records),encoding='utf-8')
    (out/'predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in predictions),encoding='utf-8')
    audit={'input':str(a.input),'labels':a.labels,'split_sizes':{s:sum(r['split']==s for r in rows) for s in ('train','dev','test')},'tied_rows_by_split':{s:sum(r['split']==s and r['gold_label'] not in a.labels for r in rows) for s in ('train','dev','test')},'normalized_split_overlap':'none','conditions':[r[0] for r in conditions],'tie_inclusive_note':'The tie-inclusive soft-target condition includes matched non-tied examples plus up to floor(budget/3) tied examples; this changes training composition and is reported separately, not as a matched comparison.'}
    (out/'audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
if __name__=='__main__': main()
