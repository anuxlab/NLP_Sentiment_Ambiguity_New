import csv, json, subprocess, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

CONDITIONS=['majority_ce','human_soft_targets','label_smoothing','human_soft_targets_null_augmented']

def make_run(root, backend='tiny-bert', seeds=(0,1), omit_condition=None, pred_count=2):
    for rnd in ['round1','round2']:
        d=root/rnd; d.mkdir(parents=True)
        args={'round':rnd,'backend':backend,'seeds':list(seeds),'budget':2,'epochs':1,'eval_size':0}
        if backend == 'tiny-bert':
            args['tiny_bert_lr']=2e-5
            effective_lr={'parameter':'tiny_bert_lr','value':2e-5,'optimizer':'torch.optim.AdamW'}
        else:
            args['lr']=0.01
            effective_lr={'parameter':'lr','value':0.01,'optimizer':'torch.optim.AdamW'}
        config={'args':args,'audit':{'conditions':CONDITIONS,'effective_learning_rate':effective_lr}}
        (d/'config_and_audit.json').write_text(json.dumps(config))
        metrics=[]; preds=[]
        for seed in seeds:
            for condition in CONDITIONS:
                if omit_condition==(rnd,seed,condition): continue
                metric={'round':rnd,'backend':backend,'seed':seed,'condition':condition,'n_test':pred_count,'js_divergence_mean_bits':0.25,'macro_f1':0.5}
                if backend == 'tiny-bert':
                    metric['learning_rate']=2e-5
                metrics.append(metric)
                for i in range(pred_count):
                    preds.append({'round':rnd,'backend':backend,'seed':seed,'condition':condition,'text_id':f'{rnd}-{i}','human_distribution':[.25]*4,'predicted_distribution':[.25]*4,'predicted_label':'neutral','js_divergence_bits':.25})
        (d/'metrics.json').write_text(json.dumps(metrics))
        with (d/'metrics.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=metrics[0].keys()); w.writeheader(); w.writerows(metrics)
        (d/'predictions.jsonl').write_text('\n'.join(json.dumps(p) for p in preds)+'\n')

def run(root, backend='tiny-bert', seeds=(0,1)):
    return subprocess.run([sys.executable,str(ROOT/'src/verify_dynasent_artifact.py'),'--results-root',str(root),'--backend',backend,'--seeds',*map(str,seeds)],capture_output=True,text=True)

def test_verifier_accepts_complete_matrix(tmp_path):
    root=tmp_path/'tiny-bert'; make_run(root)
    r=run(root); assert r.returncode==0, r.stderr
    audit=json.loads((root/'verification_report.json').read_text())
    assert audit['all_complete'] and audit['n_metric_rows']==16 and audit['n_prediction_rows']==32

def test_verifier_rejects_missing_condition_and_writes_report(tmp_path):
    root=tmp_path/'tiny-bert'; make_run(root,omit_condition=('round2',1,'label_smoothing'))
    r=run(root); assert r.returncode!=0
    audit=json.loads((root/'verification_report.json').read_text())
    assert audit['status']=='failed' and not audit['all_complete']

def test_verifier_rejects_backend_mismatch(tmp_path):
    root=tmp_path/'tfidf'; make_run(root,backend='tfidf')
    r=run(root,backend='tiny-bert'); assert r.returncode!=0
    assert 'expected backend' in (root/'verification_report.json').read_text()

def test_verifier_rejects_metrics_csv_json_mismatch(tmp_path):
    root=tmp_path/'tiny-bert'; make_run(root)
    p=root/'round1'/'metrics.csv'; p.write_text(p.read_text().replace('majority_ce','other_condition',1))
    r=run(root); assert r.returncode!=0
    assert 'identity rows differ' in (root/'verification_report.json').read_text()


def test_verifier_rejects_legacy_lr_in_tiny_bert_config(tmp_path):
    root=tmp_path/'tiny-bert'; make_run(root)
    p=root/'round1'/'config_and_audit.json'
    config=json.loads(p.read_text())
    config['args']['lr']=0.01
    p.write_text(json.dumps(config))
    r=run(root)
    assert r.returncode != 0
    assert 'unused generic args.lr' in (root/'verification_report.json').read_text()

def test_verifier_rejects_conflicting_effective_learning_rate(tmp_path):
    root=tmp_path/'tiny-bert'; make_run(root)
    p=root/'round1'/'config_and_audit.json'
    config=json.loads(p.read_text())
    config['audit']['effective_learning_rate']['value']=0.01
    p.write_text(json.dumps(config))
    r=run(root)
    assert r.returncode != 0
    assert 'conflicts with args.tiny_bert_lr' in (root/'verification_report.json').read_text()

def test_verifier_rejects_metric_learning_rate_mismatch(tmp_path):
    root=tmp_path/'tiny-bert'; make_run(root)
    p=root/'round1'/'metrics.json'
    rows=json.loads(p.read_text())
    rows[0]['learning_rate']=0.01
    p.write_text(json.dumps(rows))
    # Keep CSV consistent with the altered JSON so the LR check is the failing assertion.
    with (root/'round1'/'metrics.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    r=run(root)
    assert r.returncode != 0
    assert 'metric learning_rate missing or conflicts' in (root/'verification_report.json').read_text()
