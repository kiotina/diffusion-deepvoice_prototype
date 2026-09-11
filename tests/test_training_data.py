import csv

import numpy as np
import pytest
from torch.utils.data import DataLoader

from deepvoice_diffusion.training_data import MelDataset, read_manifest


def make_manifest(tmp_path):
    rows = []
    for i, split in enumerate(("train", "train", "validation", "test")):
        mel_path = tmp_path / f"mel{i}.npy"
        mask_path = tmp_path / f"mask{i}.npy"
        np.save(mel_path, np.zeros((1, 80, 126), dtype=np.float32))
        mask = np.ones((1, 1, 126), dtype=np.float32)
        mask[..., 90:] = 0
        np.save(mask_path, mask)
        rows.append(dict(split=split, source_path=str(tmp_path / f"source{i}.wav"),
                         segment_index="0", mel_path=mel_path.name, mask_path=mask_path.name))
    path = tmp_path / "manifest.csv"
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_loader_uses_saved_split_and_validates_arrays(tmp_path):
    rows = read_manifest(make_manifest(tmp_path))
    dataset = MelDataset(rows, "train")
    batch = next(iter(DataLoader(dataset, batch_size=2)))
    assert batch["mel"].shape == (2, 1, 80, 126)
    assert batch["mask"].shape == (2, 1, 1, 126)
    assert len(dataset) == 2
    before = dataset.fingerprint()
    np.save(rows[0]["mask_path"], np.zeros((1, 1, 126), dtype=np.float32))
    assert dataset.fingerprint() != before
    with pytest.raises(ValueError, match="at least one"):
        dataset[0]


def test_manifest_rejects_source_leakage(tmp_path):
    path = make_manifest(tmp_path)
    content = path.read_text(encoding="utf-8-sig")
    path.write_text(content.replace("source2.wav", "source0.wav"), encoding="utf-8-sig")
    with pytest.raises(ValueError, match="crosses splits"):
        read_manifest(path)
