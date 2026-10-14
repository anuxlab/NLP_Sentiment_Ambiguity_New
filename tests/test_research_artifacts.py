"""Fast integrity checks for the archived experiment records and analysis assets.

These tests do not retrain neural models. They verify reproducibility metadata,
stratified sampling, and derived cross-domain statistics from saved runs.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from data import _stratified_split, subsample  # noqa: E402


def _records():
    return [json.loads(line) for line in (ROOT / "results" / "results.jsonl").read_text().splitlines()]


def test_stratified_split_is_disjoint_and_exhaustive():
    rows = [(f"x{i}", i % 2) for i in range(101)]
    tr, dev, test = _stratified_split(rows, seed=7)
    assert len(set(tr) & set(dev)) == 0
    assert len(set(tr) & set(test)) == 0
    assert len(set(dev) & set(test)) == 0
    assert sorted(tr + dev + test) == list(range(len(rows)))
    for ids in (tr, dev, test):
        labels = [rows[i][-1] for i in ids]
        assert 0 in labels and 1 in labels


def test_stratified_subsample_respects_requested_budget():
    labels = np.array([0] * 500 + [1] * 500)
    for n in (50, 100, 300, 700, 1000):
        idx = subsample(labels, n, seed=2)
        assert len(idx) == n
        assert len(set(idx.tolist())) == n
        if n < len(labels):
            assert abs(int((labels[idx] == 0).sum()) - n / 2) <= 1


def test_saved_full_data_runs_have_three_seeds_per_core_domain_model():
    records = _records()
    full = {"sst2": 6920, "finance": 1377, "airline": 8125}
    for domain, n in full.items():
        for model in ("tiny_mlm", "tiny_scratch", "tfidf_lr"):
            selected = [r for r in records if r.get("domain", "sst2") == domain
                        and r.get("stage") == "curve" and r.get("model") == model and r.get("n") == n]
            assert len(selected) == 3, (domain, model, len(selected))
            assert sorted(r["seed"] for r in selected) == [0, 1, 2]
            assert all("test_f1" in r for r in selected)


def test_directional_drop_claims_match_saved_means():
    records = _records()

    def mean(domain, model, n, metric):
        rs = [r for r in records if r.get("domain", "sst2") == domain
              and r.get("stage") == "curve" and r.get("model") == model and r.get("n") == n]
        key = metric if domain == "sst2" or metric.startswith("cross_") else metric
        vals = []
        for r in rs:
            vals.append(r["test_acc"] if key == "test_acc" else r[key])
        return float(np.mean(vals)) * 100

    # Scratch Tiny-BERT: SST-2 -> airline is the 4.7-point drop, not airline -> SST-2.
    sst = mean("sst2", "tiny_scratch", 6920, "test_acc")
    sst_to_air = mean("sst2", "tiny_scratch", 6920, "cross_airline_acc")
    air = mean("airline", "tiny_scratch", 8125, "test_acc")
    air_to_sst = mean("airline", "tiny_scratch", 8125, "cross_sst2_acc")
    assert abs((sst - sst_to_air) - 4.66) < 0.2
    assert abs((air - air_to_sst) - 35.04) < 0.2

    finance = mean("finance", "tfidf_lr", 1377, "test_acc")
    fin_to_air = mean("finance", "tfidf_lr", 1377, "cross_airline_acc")
    assert abs((finance - fin_to_air) - 60.55) < 0.2


def test_cross_domain_asset_has_accuracy_and_macro_f1_uncertainty():
    import make_assets
    rows, _ = make_assets.cross_domain_matrix()
    assert len(rows) == 27
    assert all(len(row) == 5 for row in rows)
    table = (ROOT / "report" / "tables" / "tab_cross_domain.tex").read_text()
    assert "Macro-F1" in table
    assert "mean $\\pm$ SD" in table
    assert "single seed-0" not in table


def test_paired_bootstrap_reports_both_interval_and_seed_variability():
    from cross_domain_stats import paired_bootstrap_difference
    y = np.array([0, 0, 1, 1, 0, 1])
    pa = np.eye(2, dtype=float)[y]
    pb = np.eye(2, dtype=float)[1 - y]
    result = paired_bootstrap_difference([pa, pa], [pb, pb], y,
                                         metric="accuracy", n_boot=200, seed=123)
    assert result["difference"] == 1.0
    assert result["ci95_low"] == 1.0 and result["ci95_high"] == 1.0
    assert result["n_seeds"] == 2
    assert "conditional on saved seeds" in result["uncertainty_scope"]


def test_multidomain_learning_tables_include_macro_f1_and_accuracy():
    import make_assets
    f1_rows = make_assets.multidomain_learning_tables(make_assets.load(None))
    assert len(f1_rows) == 8
    acc_table = (ROOT / "report" / "tables" / "tab_fin_air.tex").read_text()
    f1_table = (ROOT / "report" / "tables" / "tab_fin_air_f1.tex").read_text()
    assert "Accuracy learning curves" in acc_table
    assert "Macro-F1 learning curves" in f1_table
    assert "mean$\\pm$SD" in f1_table


def test_real_domain_ambiguity_summary_has_all_three_core_models():
    import make_assets
    rows = make_assets.domain_ambiguity_summary()
    assert len(rows) == 6
    assert {r[0] for r in rows} == {"Finance", "Airline"}
    assert all(r[5] != "--" for r in rows)  # AUROC is defined for each core model
    table = (ROOT / "report" / "tables" / "tab_amb_domain_summary.tex").read_text()
    assert "annotator agreement" in table
    assert "Worker confidence" in table


def test_evidence_audit_report_is_explicit_about_scope(tmp_path):
    import subprocess
    output = tmp_path / "audit.json"
    subprocess.run([sys.executable, str(ROOT / "ci" / "audit_evidence.py"),
                    "--output", str(output)], check=True, capture_output=True, text=True)
    report = json.loads(output.read_text())
    assert report["status"] in {"PASS", "WARN"}
    assert report["counts"]["result_records"] > 0
    assert "does not validate the original training run" in report["interpretation_limit"]


def test_github_workflows_automate_matrix_experiments_and_keep_manual_fallback():
    workflow = (ROOT / ".github" / "workflows" / "experiment.yml").read_text()
    assert "push:" in workflow
    assert "schedule:" in workflow
    assert "workflow_dispatch:" in workflow  # optional recovery/debug entry point
    assert "matrix:" in workflow
    assert "domain: [sst2, finance, airline]" in workflow
    assert "name: Pretrain MLM once" in workflow
    assert "name: Merge, analyze, audit" in workflow
    assert "python -m pytest -q tests" in workflow
    assert "Research Experiment (parallel CPU domains)" in workflow
