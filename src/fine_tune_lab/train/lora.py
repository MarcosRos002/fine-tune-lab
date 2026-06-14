"""LoRA/QLoRA training loop (PEFT).

Loads a small base + tokenizer, attaches a LoRA/QLoRA adapter, trains on the
classification dataset, evaluates on the val split, and saves the adapter.
Real runs execute in the Colab/Kaggle notebooks; this module is the importable
library they drive. See ``docs/modules/train.md``.

Phase 0 stub.
"""

from __future__ import annotations

from fine_tune_lab.train.config import TrainConfig


def train_lora(config: TrainConfig | None = None) -> str:
    """Run LoRA/QLoRA training and save the adapter.

    Args:
        config: training hyperparameters; defaults to ``TrainConfig()``.

    Returns:
        Path to the saved adapter directory.
    """
    ...


def main() -> None:
    """CLI entry point (``make train``)."""
    ...


if __name__ == "__main__":
    main()
