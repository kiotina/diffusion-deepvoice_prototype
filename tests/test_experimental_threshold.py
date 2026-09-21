import pytest

from scripts.experiment_threshold import choose_balanced_threshold, split_external_files


def test_balanced_threshold_uses_only_development_scores_and_tie_breaks_high():
    threshold, score = choose_balanced_threshold([0.1, 0.4], [0.3, 0.5])
    assert threshold == pytest.approx(0.4)
    assert score == pytest.approx(0.75)
    perfect_threshold, perfect = choose_balanced_threshold([0.1, 0.2], [0.8, 0.9])
    assert perfect_threshold == pytest.approx(0.2)
    assert perfect == 1.0


def test_external_split_is_repeatable_and_disjoint():
    rows = [{"file_id": str(index)} for index in range(11)]
    first = split_external_files(rows, seed=7)
    second = split_external_files(list(reversed(rows)), seed=7)
    assert first == second
    assert len(first[0]) == 5 and len(first[1]) == 6
    assert not {row["file_id"] for row in first[0]} & {row["file_id"] for row in first[1]}
    with pytest.raises(ValueError, match="unique"):
        split_external_files([rows[0], rows[0]])


def test_balanced_threshold_rejects_invalid_scores():
    with pytest.raises(ValueError, match="nonempty"):
        choose_balanced_threshold([], [0.2])
    with pytest.raises(ValueError, match="finite"):
        choose_balanced_threshold([float("nan")], [0.2])
