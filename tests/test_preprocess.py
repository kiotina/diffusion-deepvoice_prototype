from pathlib import Path

import pytest

from scripts.preprocess import prepare_output_dirs


def test_preprocess_output_must_be_missing_or_empty(tmp_path: Path) -> None:
    output = tmp_path / "processed"
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("existing result", encoding="utf-8")

    with pytest.raises(FileExistsError, match="missing or empty"):
        prepare_output_dirs(output, ("train", "validation", "test"))

    assert marker.read_text(encoding="utf-8") == "existing result"


def test_preprocess_output_directories_are_created(tmp_path: Path) -> None:
    output = tmp_path / "processed"
    directories = prepare_output_dirs(output, ("train", "validation", "test"))

    assert directories["train"]["mels"].is_dir()
    assert directories["validation"]["masks"].is_dir()
