"""Distillation configuration. See ``docs/modules/distill.md``. Phase 0 stub."""

from __future__ import annotations

from pydantic import BaseModel


class DistillConfig(BaseModel):
    """Knowledge-distillation hyperparameters."""

    teacher_source: str = "dataset"  # 'dataset' soft targets/rationales, or live teacher
    student_base: str = "Qwen/Qwen2.5-0.5B"  # placeholder; finalize in Phase 1
    student_checkpoint: str | None = None  # optionally start from a LoRA checkpoint
    temperature: float = 2.0
    soft_target_weight: float = 0.5
    hard_target_weight: float = 0.5
    learning_rate: float = 1e-4
    num_epochs: int = 3
    seed: int = 0
    output_dir: str = "artifacts/distilled"
