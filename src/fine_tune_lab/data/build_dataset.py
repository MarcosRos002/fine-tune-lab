"""Build the synthetic classification dataset.

Load teacher labels / claims-auditor traces, normalize to the record schema,
validate, stratify-split, and write ``train/val/test.jsonl`` + a dataset card.
Never includes real patient data. See ``docs/modules/data.md``.

Phase 0 stub.
"""

from __future__ import annotations

from pathlib import Path


def build_dataset(
    sources: list[str] | None = None,
    out_dir: str | Path = "data/processed",
    seed: int = 0,
) -> None:
    """Build and write the classification dataset splits.

    Args:
        sources: paths/URIs of teacher labels and claims-auditor traces.
        out_dir: directory to write ``train/val/test.jsonl`` and the dataset card.
        seed: RNG seed for reproducible stratified splits.
    """
    ...


def main() -> None:
    """CLI entry point (``make prep-data``)."""
    ...


if __name__ == "__main__":
    main()
