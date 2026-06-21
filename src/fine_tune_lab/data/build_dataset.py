"""Build the synthetic classification dataset.

Generates synthetic claim-classification examples (never real PHI), validates
them against the record schema, writes stratified ``train/val/test.jsonl`` splits
and a dataset card. Deterministic for a given seed.

In production, ``sources`` would point at teacher labels / claims-auditor traces;
here we synthesize a balanced set so the eval/training pipeline is runnable and
testable end to end. See ``docs/modules/data.md``.
"""

from __future__ import annotations

import random
from collections import Counter
from pathlib import Path

from fine_tune_lab.data.schema import ClaimInput, ClaimMetadata, DatasetRecord, Label

# Training labels (NEEDS_HUMAN_REVIEW is a serving fallback, not a ground-truth class).
_TEMPLATES: dict[Label, tuple[str, str]] = {
    Label.CLEAN: ("Routine office visit, established patient, low complexity", "99213"),
    Label.UPCODING_SUSPECTED: (
        "Office visit billed at highest complexity with minimal documentation",
        "99215",
    ),
    Label.UNBUNDLING_SUSPECTED: (
        "Component procedures billed separately instead of the bundled code",
        "80048",
    ),
    Label.MISSING_DOCUMENTATION: (
        "Procedure billed without supporting clinical notes attached",
        "71046",
    ),
    Label.DUPLICATE: ("Identical claim line submitted twice on the same date of service", "93000"),
}
_ICDS = ["E11.9", "I10", "J06.9", "M54.5", "N18.3"]


def generate_records(n: int, seed: int = 0) -> list[DatasetRecord]:
    """Generate ``n`` synthetic, balanced, deterministic dataset records."""
    rng = random.Random(seed)
    labels = list(_TEMPLATES)
    records: list[DatasetRecord] = []
    for i in range(n):
        label = labels[i % len(labels)]
        text, cpt = _TEMPLATES[label]
        records.append(
            DatasetRecord(
                id=f"rec-{seed}-{i:05d}",
                input=ClaimInput(
                    claim_text=text,
                    metadata=ClaimMetadata(
                        cpt_code=cpt,
                        icd_code=rng.choice(_ICDS),
                        amount_usd=round(rng.uniform(50, 500), 2),
                    ),
                ),
                label=label,
                rationale=f"Synthetic example of {label.value}.",
                source="synthetic",
            )
        )
    return records


def _stratified_split(records: list[DatasetRecord], seed: int) -> dict[str, list[DatasetRecord]]:
    rng = random.Random(seed)
    by_label: dict[Label, list[DatasetRecord]] = {}
    for r in records:
        by_label.setdefault(r.label, []).append(r)
    splits: dict[str, list[DatasetRecord]] = {"train": [], "val": [], "test": []}
    for group in by_label.values():
        rng.shuffle(group)
        n = len(group)
        n_train = int(n * 0.8)
        n_val = int(n * 0.1)
        splits["train"] += group[:n_train]
        splits["val"] += group[n_train : n_train + n_val]
        splits["test"] += group[n_train + n_val :]
    return splits


def build_dataset(
    sources: list[str] | None = None,
    out_dir: str | Path = "data/processed",
    seed: int = 0,
    n: int = 200,
) -> None:
    """Build and write the classification dataset splits + a dataset card."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    records = generate_records(n, seed=seed)
    splits = _stratified_split(records, seed=seed)

    for split, rows in splits.items():
        (out / f"{split}.jsonl").write_text("\n".join(r.model_dump_json() for r in rows) + "\n")

    balance = Counter(r.label.value for r in records)
    card = [
        "# Dataset card — synthetic claim classification",
        "",
        f"- Records: {len(records)} (seed={seed}); **synthetic, no PHI**.",
        f"- Splits: train={len(splits['train'])}, val={len(splits['val'])}, "
        f"test={len(splits['test'])}.",
        "",
        "## Class balance",
        *(f"- {label}: {count}" for label, count in sorted(balance.items())),
    ]
    (out / "dataset_card.md").write_text("\n".join(card) + "\n")


def main() -> None:
    """CLI entry point (``make prep-data``)."""
    build_dataset()


if __name__ == "__main__":
    main()
