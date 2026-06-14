"""Training configuration. Base model is a config choice (swappable).

See ``docs/modules/train.md`` and ``docs/adr/0001-stack-and-scope.md``.
Phase 0 stub.
"""

from __future__ import annotations

from pydantic import BaseModel


class TrainConfig(BaseModel):
    """LoRA/QLoRA training hyperparameters."""

    base_model: str = "Qwen/Qwen2.5-0.5B"  # placeholder; finalize in Phase 1
    use_4bit: bool = False  # QLoRA when True
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list[str] = ["q_proj", "v_proj"]
    learning_rate: float = 2e-4
    num_epochs: int = 3
    batch_size: int = 8
    grad_accum: int = 4
    seed: int = 0
    output_dir: str = "artifacts/lora"
