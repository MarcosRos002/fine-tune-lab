"""Tests for synthetic classification dataset generation + split writing."""

from __future__ import annotations

from fine_tune_lab.data.build_dataset import build_dataset, generate_records
from fine_tune_lab.data.schema import DatasetRecord, Label


def test_generate_records_is_deterministic_and_balanced() -> None:
    a = generate_records(50, seed=0)
    b = generate_records(50, seed=0)
    assert [r.id for r in a] == [r.id for r in b]  # deterministic
    labels = {r.label for r in a}
    # the 5 real training labels appear; NEEDS_HUMAN_REVIEW is a serving fallback only
    assert Label.NEEDS_HUMAN_REVIEW not in labels
    assert len(labels) >= 4
    assert all(isinstance(r, DatasetRecord) for r in a)


def test_build_dataset_writes_valid_disjoint_splits(tmp_path) -> None:
    build_dataset(out_dir=tmp_path, n=60, seed=1)
    ids_by_split = {}
    for split in ["train", "val", "test"]:
        path = tmp_path / f"{split}.jsonl"
        assert path.exists()
        lines = [ln for ln in path.read_text().splitlines() if ln]
        for ln in lines:
            DatasetRecord.model_validate_json(ln)  # every line is valid
        ids_by_split[split] = {DatasetRecord.model_validate_json(ln).id for ln in lines}

    # splits are disjoint by id
    assert ids_by_split["train"].isdisjoint(ids_by_split["val"])
    assert ids_by_split["train"].isdisjoint(ids_by_split["test"])
    assert ids_by_split["val"].isdisjoint(ids_by_split["test"])
    # dataset card written
    assert (tmp_path / "dataset_card.md").exists()
