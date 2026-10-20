#!/usr/bin/env python3
"""Strict artifact verifier for DynaSent multi-seed runs.

Checks the selected backend only, both rounds, the expected seed x condition matrix,
config/metrics provenance, prediction coverage, distribution validity, and the
per-group prediction counts. TF-IDF and Tiny-BERT must be stored under separate roots.
"""
from __future__ import annotations
import argparse, csv, json, math, sys
from collections import Counter, defaultdict
from pathlib import Path

ROUNDS = ("round1", "round2")
CONDITIONS = ("majority_ce", "human_soft_targets", "label_smoothing", "human_soft_targets_null_augmented")
LABELS = ("negative", "neutral", "positive", "mixed")

class VerificationError(RuntimeError): pass

def read_json(path: Path):
    if not path.is_file(): raise VerificationError(f"missing required file: {path}")
    try: return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc: raise VerificationError(f"invalid JSON in {path}: {exc}") from exc

def read_jsonl(path: Path):
    if not path.is_file(): raise VerificationError(f"missing required file: {path}")
    rows=[]
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip(): continue
        try: rows.append(json.loads(line))
        except Exception as exc: raise VerificationError(f"invalid JSONL at {path}:{line_no}: {exc}") from exc
    return rows

def close(a, b, tol=1e-5):
    try: return math.isfinite(float(a)) and math.isfinite(float(b)) and abs(float(a)-float(b)) <= tol
    except (TypeError, ValueError): return False

def valid_dist(v):
    return isinstance(v, list) and len(v)==4 and all(isinstance(x,(int,float)) and math.isfinite(x) and x >= -1e-8 for x in v) and abs(sum(v)-1.0) <= 1e-4

def verify(results_root: Path, backend: str, seeds: list[int], rounds: list[str], require_null_augmented=True):
    expected_conditions = list(CONDITIONS if require_null_augmented else CONDITIONS[:3])
    metric_rows=[]; prediction_rows=[]; configs=[]; checks=[]; errors=[]
    for rnd in rounds:
        d=results_root/rnd
        config=read_json(d/'config_and_audit.json')
        metrics_json=read_json(d/'metrics.json')
        if not isinstance(metrics_json,list): raise VerificationError(f"{d/'metrics.json'} must contain a JSON list")
        csv_path=d/'metrics.csv'
        if not csv_path.is_file(): raise VerificationError(f"missing required file: {csv_path}")
        with csv_path.open(newline='', encoding='utf-8') as f: csv_rows=list(csv.DictReader(f))
        if len(csv_rows)!=len(metrics_json): raise VerificationError(f"{rnd}: metrics.csv ({len(csv_rows)}) and metrics.json ({len(metrics_json)}) row counts differ")
        args=config.get('args',{})
        if args.get('round') != rnd: raise VerificationError(f"{rnd}: config round mismatch: {args.get('round')!r}")
        if args.get('backend') != backend: raise VerificationError(f"{rnd}: expected backend {backend!r}, config says {args.get('backend')!r}")
        audit=config.get('audit',{})
        effective_lr=audit.get('effective_learning_rate')
        if not isinstance(effective_lr, dict):
            raise VerificationError(f"{rnd}: missing audit.effective_learning_rate metadata")
        expected_lr_key = 'tiny_bert_lr' if backend == 'tiny-bert' else 'lr'
        if effective_lr.get('parameter') != expected_lr_key:
            raise VerificationError(f"{rnd}: effective learning-rate parameter must be {expected_lr_key!r}")
        configured_lr=args.get(expected_lr_key)
        if not isinstance(configured_lr,(int,float)) or not math.isfinite(configured_lr) or configured_lr <= 0:
            raise VerificationError(f"{rnd}: missing or invalid args.{expected_lr_key}")
        if not close(effective_lr.get('value'), configured_lr, tol=1e-12):
            raise VerificationError(f"{rnd}: effective learning-rate value conflicts with args.{expected_lr_key}")
        if backend == 'tiny-bert' and 'lr' in args:
            raise VerificationError(f"{rnd}: Tiny-BERT config must not include unused generic args.lr")
        if backend == 'tfidf' and 'tiny_bert_lr' in args:
            raise VerificationError(f"{rnd}: TF-IDF config must not include unused args.tiny_bert_lr")
        config_seeds=sorted(int(s) for s in args.get('seeds',[]))
        if config_seeds != sorted(seeds): raise VerificationError(f"{rnd}: configured seeds {config_seeds} != requested {sorted(seeds)}")
        configured_conditions=config.get('audit',{}).get('conditions')
        if configured_conditions is not None and not set(expected_conditions).issubset(set(configured_conditions)):
            raise VerificationError(f"{rnd}: config audit does not declare expected conditions {expected_conditions}")
        for row in metrics_json:
            if row.get('round') != rnd or row.get('backend') != backend:
                raise VerificationError(f"{rnd}: metrics row has wrong round/backend: {row.get('round')}/{row.get('backend')}")
            if backend == 'tiny-bert' and not close(row.get('learning_rate'), configured_lr, tol=1e-12):
                raise VerificationError(f"{rnd}: metric learning_rate missing or conflicts with effective Tiny-BERT learning rate")
            if int(row.get('seed',-999)) not in seeds: raise VerificationError(f"{rnd}: unexpected seed {row.get('seed')}")
            if row.get('condition') not in expected_conditions: raise VerificationError(f"{rnd}: unexpected condition {row.get('condition')}")
        # CSV and JSON representations must agree on their identity matrix.
        def ident(r): return (r.get('round'),r.get('backend'),str(r.get('seed')),r.get('condition'))
        if Counter(ident(x) for x in csv_rows) != Counter(ident(x) for x in metrics_json):
            raise VerificationError(f"{rnd}: metrics.csv and metrics.json identity rows differ")
        metric_rows.extend(metrics_json); configs.append({'round':rnd,'backend':backend,'seeds':config_seeds,'budget':args.get('budget'),'epochs':args.get('epochs'),'eval_size':args.get('eval_size'),'effective_learning_rate':effective_lr})
        preds=read_jsonl(d/'predictions.jsonl')
        for p in preds:
            if p.get('round') != rnd or p.get('backend') != backend: raise VerificationError(f"{rnd}: prediction has wrong round/backend")
            if int(p.get('seed',-999)) not in seeds: raise VerificationError(f"{rnd}: prediction has unexpected seed {p.get('seed')}")
            if p.get('condition') not in expected_conditions: raise VerificationError(f"{rnd}: prediction has unexpected condition {p.get('condition')}")
            if not p.get('text_id'): raise VerificationError(f"{rnd}: prediction missing text_id")
            if not valid_dist(p.get('human_distribution')): raise VerificationError(f"{rnd}: invalid human_distribution for {p.get('text_id')}")
            if not valid_dist(p.get('predicted_distribution')): raise VerificationError(f"{rnd}: invalid predicted_distribution for {p.get('text_id')}")
            if p.get('predicted_label') not in LABELS: raise VerificationError(f"{rnd}: invalid predicted_label {p.get('predicted_label')!r}")
        prediction_rows.extend(preds)
    # Require exactly one metric row for every round x seed x expected condition.
    metric_keys=[(r['round'],int(r['seed']),r['condition']) for r in metric_rows]
    counts=Counter(metric_keys)
    expected={(rnd,seed,cond) for rnd in rounds for seed in seeds for cond in expected_conditions}
    missing=sorted(expected-set(counts)); duplicate=sorted(k for k,v in counts.items() if v!=1)
    unexpected=sorted(set(counts)-expected)
    if missing or duplicate or unexpected:
        raise VerificationError(f"metric matrix incomplete/invalid: missing={missing[:8]}, duplicate={duplicate[:8]}, unexpected={unexpected[:8]}")
    # Prediction key uniqueness and full test coverage per metric group.
    pred_keys=[(p['round'],int(p['seed']),p['condition'],str(p['text_id'])) for p in prediction_rows]
    pc=Counter(pred_keys)
    duplicate_preds=[k for k,v in pc.items() if v!=1]
    if duplicate_preds: raise VerificationError(f"duplicate prediction keys found (examples: {duplicate_preds[:5]})")
    pred_groups=defaultdict(list)
    for p in prediction_rows: pred_groups[(p['round'],int(p['seed']),p['condition'])].append(p)
    for m in metric_rows:
        key=(m['round'],int(m['seed']),m['condition'])
        group=pred_groups.get(key,[])
        n_test=int(m.get('n_test',0))
        if n_test<=0: raise VerificationError(f"metric group {key} has invalid n_test={n_test}")
        if len(group)!=n_test: raise VerificationError(f"prediction count mismatch for {key}: expected {n_test}, found {len(group)}")
        # Recompute mean JS from prediction-level values when present.
        vals=[p.get('js_divergence_bits') for p in group]
        if all(isinstance(x,(int,float)) and math.isfinite(x) for x in vals) and 'js_divergence_mean_bits' in m:
            if not close(sum(vals)/len(vals),m['js_divergence_mean_bits'],tol=2e-5):
                raise VerificationError(f"JS aggregate mismatch for {key}: predictions={sum(vals)/len(vals):.8f}, metric={m['js_divergence_mean_bits']}")
        checks.append({'round':key[0],'seed':key[1],'condition':key[2],'metric_rows':1,'prediction_rows':len(group),'expected_prediction_rows':n_test,'complete':True})
    # Ensure the two rounds have consistent evaluation settings where expected.
    if len({c['backend'] for c in configs}) != 1: raise VerificationError('mixed backends detected')
    return {'status':'passed','backend':backend,'rounds':rounds,'seeds':sorted(seeds),'conditions':expected_conditions,
            'n_metric_rows':len(metric_rows),'n_prediction_rows':len(prediction_rows),'n_expected_groups':len(expected),
            'n_complete_groups':len(checks),'configs':configs,'checks':checks,'all_complete':True}

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--results-root',required=True,help='Root containing round1/ and round2/; use a backend-specific root')
    ap.add_argument('--backend',required=True,choices=['tiny-bert','tfidf'])
    ap.add_argument('--seeds',nargs='+',type=int,default=[0,1,2,3,4])
    ap.add_argument('--rounds',nargs='+',default=list(ROUNDS),choices=list(ROUNDS))
    ap.add_argument('--allow-three-conditions',action='store_true',help='Use only majority_ce, human_soft_targets, label_smoothing')
    ap.add_argument('--report',default=None,help='Optional JSON output path (default: results-root/verification_report.json)')
    args=ap.parse_args(); root=Path(args.results_root); report=Path(args.report) if args.report else root/'verification_report.json'
    try:
        if len(set(args.seeds))!=len(args.seeds): raise VerificationError('seeds must be unique')
        result=verify(root,args.backend,args.seeds,args.rounds,not args.allow_three_conditions)
    except Exception as exc:
        result={'status':'failed','backend':args.backend,'rounds':args.rounds,'seeds':args.seeds,'all_complete':False,'error':str(exc)}
        report.parent.mkdir(parents=True,exist_ok=True); report.write_text(json.dumps(result,indent=2),encoding='utf-8')
        print(f"VERIFY FAILED: {exc}",file=sys.stderr); print(f"Audit written: {report}",file=sys.stderr); return 1
    report.parent.mkdir(parents=True,exist_ok=True); report.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(f"VERIFY PASSED: backend={args.backend}; rounds={len(args.rounds)}; seeds={len(args.seeds)}; conditions={len(result['conditions'])}; metric_rows={result['n_metric_rows']}; prediction_rows={result['n_prediction_rows']}; complete_groups={result['n_complete_groups']}/{result['n_expected_groups']}")
    print(f"Audit written: {report}"); return 0
if __name__=='__main__': raise SystemExit(main())
