"""Training configuration. Base model is a config choice (swappable).

Design note: the classification target is a CLOSED set of labels, so a small
**encoder classifier** with a LoRA adapter is the right tool — cheaper, faster and
more reliable than a generative LLM (which would be overkill for picking 1-of-N).
The default base is a tiny BERT so the loop runs on CPU for development; swap
``base_model`` for a stronger encoder on Colab/Kaggle. See
``docs/modules/train.md`` and ``docs/adr/0001-stack-and-scope.md``.
"""

from __future__ import annotations

from pydantic import BaseModel


class TrainConfig(BaseModel):
    """LoRA training hyperparameters for sequence classification."""

    base_model: str = "google/bert_uncased_L-2_H-128_A-2"  # BERT-tiny; CPU-friendly default
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    target_modules: list[str] = ["query", "value"]  # BERT attention projections
    learning_rate: float = 5e-4
    num_epochs: int = 5
    batch_size: int = 16
    max_length: int = 128
    seed: int = 0
    output_dir: str = "artifacts/lora"
