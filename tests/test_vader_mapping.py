import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dynasent_data import LABELS
from run_dynasent_vader_baseline import map_vader


def test_vader_mapping_four_classes():
    assert LABELS[map_vader({"pos": .2, "neg": .2, "compound": 0.0})] == "mixed"
    assert LABELS[map_vader({"pos": .5, "neg": .01, "compound": .5})] == "positive"
    assert LABELS[map_vader({"pos": .01, "neg": .5, "compound": -.5})] == "negative"
    assert LABELS[map_vader({"pos": .01, "neg": .01, "compound": 0.0})] == "neutral"


def test_vader_threshold_boundary_is_neutral():
    assert LABELS[map_vader({"pos": .1, "neg": .1, "compound": .049})] == "neutral"
