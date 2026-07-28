import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from evaluate import Row, duplicate_leakage, grouped_split


def test_source_groups_do_not_cross_split() -> None:
    rows = [
        Row("source a", "candidate 1", 1, "a"),
        Row("source a duplicate", "candidate 2", 0, "a"),
        Row("source b", "candidate 3", 1, "b"),
        Row("source c", "candidate 4", 0, "c"),
        Row("source d", "candidate 5", 0, "d"),
    ]

    calibration, test = grouped_split(rows, test_fraction=0.5)

    assert calibration
    assert test
    assert duplicate_leakage(calibration, test) == 0
