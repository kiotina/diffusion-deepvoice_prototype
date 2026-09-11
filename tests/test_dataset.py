from pathlib import Path

from deepvoice_diffusion.dataset import split_files


def test_file_split_is_reproducible_and_disjoint() -> None:
    files = [Path(f"sample_{index}.wav") for index in range(20)]
    first = split_files(files, 0.8, 0.1, 0.1, seed=42)
    second = split_files(files, 0.8, 0.1, 0.1, seed=42)

    assert first == second
    assert {name: len(values) for name, values in first.items()} == {
        "train": 16,
        "validation": 2,
        "test": 2,
    }
    assert set(first["train"]).isdisjoint(first["validation"])
    assert set(first["train"]).isdisjoint(first["test"])
    assert set(first["validation"]).isdisjoint(first["test"])
    assert set().union(*map(set, first.values())) == set(files)
