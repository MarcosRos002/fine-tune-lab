"""Evaluation: accuracy + per-class metrics, latency, and the substitutability
checklist on the held-out test split. See ``docs/modules/eval.md``.

Phase 0 stub.
"""

from __future__ import annotations

from typing import Any


def evaluate(model: Any, dataset_path: str) -> dict[str, Any]:
    """Evaluate a model on the held-out test split.

    Runs predictions, validates each output against the classification IO
    contract, and computes accuracy, per-class P/R/F1, and latency.

    Args:
        model: a callable/served model satisfying ``classification_io.md``.
        dataset_path: path to ``test.jsonl``.

    Returns:
        Structured metrics dict (accuracy, per-class, latency, parse/in-enum rates).
    """
    ...


def main() -> None:
    """CLI entry point (``make eval``)."""
    ...


if __name__ == "__main__":
    main()
