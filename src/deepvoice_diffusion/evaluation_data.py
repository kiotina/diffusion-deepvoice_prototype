# 평가 파일 목록 관리
"""기존 source split과 외부 WAV의 정답 상태를 보존하는 파일 목록."""
from __future__ import annotations

import csv
import hashlib
from pathlib import Path

from .data_contract import sha256_file
from .training_data import read_manifest

INVENTORY_FIELDS = ("purpose", "file_id", "path", "source", "split", "declared_label", "label",
                    "label_status", "evidence", "sha256")


def file_id(path: Path) -> str:
    """경로가 같으면 항상 같은 평가용 파일 ID를 만든다."""
    return hashlib.sha256(str(path).casefold().encode("utf-8")).hexdigest()[:20]


def build_inventory(manifest_path: Path, external_dir: Path,
                    verified_fake_manifest: Path | None = None) -> list[dict[str, str]]:
    rows = read_manifest(manifest_path)
    real: dict[str, str] = {}
    for row in rows:
        path = str(Path(row["source_path"]).resolve())
        previous = real.setdefault(path, row["split"])
        if previous != row["split"]:
            raise ValueError(f"Real source crosses splits: {path}")
    inventory = []
    for path, split in sorted(real.items()):
        inventory.append(dict(purpose="evaluation", file_id=file_id(Path(path)), path=path, source="real_manifest",
                              split=split, declared_label="0", label="0", label_status="verified",
                              evidence="existing_real_manifest", sha256=sha256_file(path)))
    verified: dict[str, str] = {}
    if verified_fake_manifest is not None:
        with verified_fake_manifest.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if not {"path", "evidence"}.issubset(reader.fieldnames or []):
                raise ValueError("Verified fake manifest needs path,evidence columns")
            for row in reader:
                source = Path(row["path"])
                if not source.is_absolute():
                    source = verified_fake_manifest.parent / source
                source = source.resolve()
                if not row["evidence"].strip() or str(source) in verified:
                    raise ValueError("Verified fake needs unique paths and nonempty evidence")
                verified[str(source)] = row["evidence"].strip()
    external = sorted(path.resolve() for path in external_dir.rglob("*")
                      if path.is_file() and path.suffix.lower() == ".wav")
    if not external:
        raise ValueError(f"No external WAV files: {external_dir}")
    if set(verified) - {str(path) for path in external}:
        raise ValueError("Verified fake manifest references a WAV outside external_dir")
    for path in external:
        evidence = verified.get(str(path))
        inventory.append(dict(purpose="evaluation", file_id=file_id(path), path=str(path), source="external",
                              split="external", declared_label="1", label="1" if evidence else "",
                              label_status="verified" if evidence else "unverified",
                              evidence=evidence or "", sha256=sha256_file(path)))
    validate_inventory(inventory)
    return inventory


def validate_inventory(rows: list[dict[str, str]], *, check_files: bool = False) -> None:
    if not rows:
        raise ValueError("Inventory is empty")
    paths: dict[str, dict] = {}
    hashes: dict[str, dict] = {}
    ids: set[str] = set()
    for row in rows:
        if set(INVENTORY_FIELDS) - {"purpose"} - set(row):
            raise ValueError("Inventory is missing required fields")
        if row.get("purpose", "evaluation") not in {"evaluation", "functional_smoke"}:
            raise ValueError("Unknown inventory purpose")
        path = str(Path(row["path"]).resolve()).casefold()
        if row["file_id"] in ids or path in paths:
            raise ValueError(f"Duplicate inventory path/id: {row['path']}")
        ids.add(row["file_id"])
        paths[path] = row
        if row["sha256"] in hashes:
            other = hashes[row["sha256"]]
            raise ValueError(f"Duplicate content across inventory entries: {other['path']} and {row['path']}")
        hashes[row["sha256"]] = row
        if row["source"] == "real_manifest":
            if row["split"] not in {"train", "validation", "test"} or row["label"] != "0" or row["label_status"] != "verified":
                raise ValueError("Invalid real manifest label/split")
        elif row["source"] == "external":
            if row["split"] != "external" or row["declared_label"] != "1":
                raise ValueError("Invalid external inventory entry")
            if row["label_status"] == "verified":
                if row["label"] != "1" or not row["evidence"].strip():
                    raise ValueError("Verified external fake requires label=1 and evidence")
            elif row["label_status"] != "unverified" or row["label"] != "":
                raise ValueError("Unverified external label must be empty")
        else:
            raise ValueError("Unknown inventory source")
        if check_files and sha256_file(row["path"]) != row["sha256"]:
            raise ValueError(f"Inventory source changed: {row['path']}")


def validate_inventory_against_manifest(
    rows: list[dict[str, str]],
    manifest_path: Path,
    *,
    allow_real_subset: bool = False,
    check_files: bool = False,
) -> None:
    """inventory의 real 경로와 split이 원본 manifest에서 바뀌지 않았는지 확인한다."""
    validate_inventory(rows, check_files=check_files)
    expected: dict[str, str] = {}
    for row in read_manifest(manifest_path):
        path = str(Path(row["source_path"]).resolve()).casefold()
        previous = expected.setdefault(path, row["split"])
        if previous != row["split"]:
            raise ValueError(f"Real source crosses manifest splits: {row['source_path']}")

    actual = {
        str(Path(row["path"]).resolve()).casefold(): row["split"]
        for row in rows
        if row["source"] == "real_manifest"
    }
    unknown = set(actual) - set(expected)
    if unknown:
        raise ValueError("Inventory contains real files that are absent from the manifest")
    changed = [path for path, split in actual.items() if expected[path] != split]
    if changed:
        raise ValueError("Inventory real split differs from the manifest")
    if not allow_real_subset and set(actual) != set(expected):
        raise ValueError("Evaluation inventory must contain every real manifest source")


def write_inventory(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=INVENTORY_FIELDS)
        writer.writeheader()
        writer.writerows({**row, "purpose": row.get("purpose", "evaluation")} for row in rows)


def read_inventory(path: Path, *, check_files: bool = True) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    validate_inventory(rows, check_files=check_files)
    return rows
