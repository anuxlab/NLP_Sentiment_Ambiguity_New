import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from run_annotated_softlabel_experiment_controlled import parse_rows


def test_tied_vote_rows_are_not_given_arbitrary_hard_labels(tmp_path):
    p=tmp_path/'rows.jsonl'
    p.write_text('\n'.join([
        json.dumps({'text':'one','split':'train','gold_label':'positive','votes':{'negative':1,'neutral':1,'positive':0}}),
        json.dumps({'text':'two','split':'train','votes':{'negative':0,'neutral':0,'positive':2}}),
        json.dumps({'text':'three','split':'dev','votes':{'negative':0,'neutral':2,'positive':0}}),
        json.dumps({'text':'four','split':'test','votes':{'negative':1,'neutral':1,'positive':1}}),
    ]))
    rows=parse_rows(p,['negative','neutral','positive'])
    assert rows[0]['is_tie'] is True and rows[0]['gold_label'] is None
    assert rows[1]['gold_label']=='positive'
    assert rows[3]['is_tie'] is True and rows[3]['gold_label'] is None


def test_split_overlap_fails_fast(tmp_path):
    p=tmp_path/'rows.jsonl'
    p.write_text('\n'.join([
        json.dumps({'text':'same text','split':'train','votes':{'positive':2}}),
        json.dumps({'text':'same text','split':'dev','votes':{'positive':2}}),
        json.dumps({'text':'test text','split':'test','votes':{'negative':2}}),
    ]))
    try:
        parse_rows(p,['negative','neutral','positive'])
    except ValueError as e:
        assert 'overlap' in str(e)
    else:
        raise AssertionError('expected overlap validation to fail')
