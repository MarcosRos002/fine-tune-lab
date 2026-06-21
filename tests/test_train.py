"""Tests for the LoRA training plumbing that runs without torch.

(`train_lora` / `build_predictor` need torch and are exercised by a real CPU smoke
run / the Colab notebooks, not the unit suite.)
"""

from __future__ import annotations

from fine_tune_lab.data.build_dataset import build_dataset
from fine_tune_lab.data.schema import Label
from fine_tune_lab.train.config import TrainConfig
from fine_tune_lab.train.lora import label_maps, load_split


def test_label_maps_exclude_the_serving_fallback() -> None:
    label2id, id2label = label_maps()
    assert Label.NEEDS_HUMAN_REVIEW.value not in label2id
    assert len(label2id) == 5
    assert set(id2label.values()) == set(label2id)


def test_load_split_returns_prompts_and_valid_label_ids(tmp_path) -> None:
    build_dataset(out_dir=tmp_path, n=50, seed=3)
    texts, labels = load_split(tmp_path, "train")
    assert texts and len(texts) == len(labels)
    assert all("Classify this medical billing claim" in t for t in texts)  # serialize_prompt used
    assert all(0 <= y < 5 for y in labels)


def test_default_config_is_a_cpu_friendly_encoder() -> None:
    cfg = TrainConfig()
    assert "bert" in cfg.base_model.lower()
    assert cfg.lora_rank > 0 and cfg.num_epochs > 0
