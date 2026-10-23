#!/usr/bin/env python3
"""Validate archived DynaSent multi-seed condition metadata against metrics.csv.

Run from repository root:
  python scripts/validate_dynasent_metadata.py
The script expects results/dynasent_multiseed/{round1,round2}/config_and_audit.json
and metrics.csv plus run_completeness.json.
"""
import csv, json, sys
from pathlib import Path

ROOT = Path('.')
BASE = ROOT / 'results' / 'dynasent_multiseed'
PRIMARY = {'majority_ce', 'human_soft_targets', 'label_smoothing'}
ALL = PRIMARY | {'human_soft_targets_null_augmented'}
errors = []

for rnd in ('round1', 'round2'):
    folder = BASE / rnd
    config_path = folder / 'config_and_audit.json'
    metrics_path = folder / 'metrics.csv'
    if not config_path.exists() or not metrics_path.exists():
        errors.append(f'{rnd}: missing config_and_audit.json or metrics.csv')
        continue
    config = json.loads(config_path.read_text())
    declared = set(config.get('audit', {}).get('conditions', []))
    with metrics_path.open(newline='') as f:
        rows = list(csv.DictReader(f))
    # The repository schema uses a column named condition; fail clearly if it changes.
    if rows and 'condition' not in rows[0]:
        errors.append(f'{rnd}: metrics.csv has no condition column')
        continue
    observed = {r.get('condition', '').strip() for r in rows}
    if not ALL.issubset(observed):
        errors.append(f'{rnd}: expected all four conditions in metrics; observed {sorted(observed)}')
    if not ALL.issubset(declared):
        errors.append(f'{rnd}: config conditions omit {sorted(ALL - declared)}')
    if declared - ALL:
        errors.append(f'{rnd}: unexpected declared conditions {sorted(declared - ALL)}')
    for condition in ALL:
        if not any(r.get('condition', '').strip() == condition for r in rows):
            errors.append(f'{rnd}: no metric rows for {condition}')

completeness_path = BASE / 'run_completeness.json'
if completeness_path.exists():
    completeness = json.loads(completeness_path.read_text())
    if not completeness.get('all_complete', False):
        errors.append('run_completeness.json does not report all_complete=true')
else:
    errors.append('missing run_completeness.json')

if errors:
    print('FAIL')
    print('\n'.join(f'- {e}' for e in errors))
    sys.exit(1)
print('PASS: metadata declares all four evaluated conditions; completeness report is green.')
