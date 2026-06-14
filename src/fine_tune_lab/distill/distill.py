"""Knowledge distillation: compress teacher judgment into the small student.

Uses teacher labels + rationales and, where available, soft targets
(``teacher_logprobs``) from the dataset. See ``docs/modules/distill.md``.

Phase 0 stub.
"""

from __future__ import annotations

from fine_tune_lab.distill.config import DistillConfig


def distill(config: DistillConfig | None = None) -> str:
    """Distill the teacher into the student and save the model.

    Args:
        config: distillation hyperparameters; defaults to ``DistillConfig()``.

    Returns:
        Path to the saved distilled model directory.
    """
    ...


def main() -> None:
    """CLI entry point (``make distill``)."""
    ...


if __name__ == "__main__":
    main()
