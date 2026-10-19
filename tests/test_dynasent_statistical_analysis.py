import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from analyze_dynasent_results import run_analysis, _holm

CONDITIONS = ["majority_ce", "human_soft_targets", "label_smoothing", "human_soft_targets_null_augmented"]


def make_input(root: Path, mixed_backend=False, omit_condition=False):
    root.mkdir(parents=True, exist_ok=True)
    for rnd, n in [("round1", 8), ("round2", 6)]:
        d = root / rnd; d.mkdir()
        metric_rows=[]; pred_rows=[]
        conds=CONDITIONS[:-1] if omit_condition else CONDITIONS
        for seed in [0, 1]:
            for ci, cond in enumerate(conds):
                js=[]
                for i in range(n):
                    # Stable human target per test item, repeated across conditions/seeds.
                    human = [1.0,0.0,0.0,0.0] if i % 2 == 0 else [0.8,0.2,0.0,0.0]
                    pred = [0.8,0.1,0.05,0.05] if cond == "human_soft_targets" else [0.7,0.1,0.1,0.1]
                    pp=np.asarray(pred,dtype=float); qq=np.asarray(human,dtype=float); mm=(pp+qq)/2
                    def kl(a,b):
                        mask=a>0
                        return float(np.sum(a[mask]*np.log2(a[mask]/b[mask])))
                    val=0.5*kl(pp,mm)+0.5*kl(qq,mm)
                    js.append(val)
                    pred_rows.append({"round":rnd,"backend":"tfidf" if mixed_backend and rnd=="round2" else "tiny-bert","seed":seed,"condition":cond,
                      "text_id":f"{rnd}-{i}","gold_label":"negative","human_distribution":human,
                      "predicted_distribution":pred,"predicted_label":"negative","js_divergence_bits":val})
                metric_rows.append({"round":rnd,"backend":"tiny-bert","seed":seed,"condition":cond,"n_test":n,
                  "js_divergence_mean_bits":float(np.mean(js)),"js_divergence_median_bits":float(np.median(js)),
                  "macro_f1":0.5,"accuracy":0.6,"soft_brier_score":0.2,"ece_10bin":0.1,"train_time_s":1.0})
        pd.DataFrame(metric_rows).to_csv(d/"metrics.csv",index=False)
        with (d/"predictions.jsonl").open("w") as f:
            for row in pred_rows: f.write(json.dumps(row)+"\n")


def test_holm_monotone_and_bounded():
    got=_holm([0.01,0.04])
    assert all(0 <= x <= 1 for x in got)
    assert got[0] <= got[1]


def test_analysis_writes_primary_and_report(tmp_path):
    inp=tmp_path/"input"; out=tmp_path/"analysis"
    make_input(inp)
    result=run_analysis(inp,out,expected_seeds=[0,1],n_boot=200,n_perm=200,random_seed=4)
    assert result["status"] == "passed"
    assert result["metric_records"] == 16
    assert result["prediction_rows"] == (8+6)*2*4
    primary=pd.read_csv(out/"primary_js_comparisons.csv")
    assert list(primary["round"]) == ["round1","round2"]
    assert "holm_adjusted_p_two_primary_round_tests" in primary
    assert (out/"FINAL_RESULTS_REPORT.md").exists()
    assert (out/"figures"/"primary_js_effects.png").exists()


def test_rejects_mixed_backends(tmp_path):
    inp=tmp_path/"input"; make_input(inp,mixed_backend=True)
    with pytest.raises(ValueError, match="Backend mismatch|backend"):
        run_analysis(inp,tmp_path/"out",expected_seeds=[0,1],n_boot=100,n_perm=100)


def test_rejects_missing_condition_in_metrics(tmp_path):
    inp=tmp_path/"input"; make_input(inp,omit_condition=True)
    with pytest.raises(ValueError, match="Expected all four research conditions"):
        run_analysis(inp,tmp_path/"out",expected_seeds=[0,1],n_boot=100,n_perm=100)
