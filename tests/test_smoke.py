"""Phase 0 smoke tests — confirm the package imports and the contract enum
holds. No features are implemented yet; these guard the scaffold."""

from __future__ import annotations

import fine_tune_lab
from fine_tune_lab.data.schema import ClaimInput, DatasetRecord, Label


def test_package_imports() -> None:
    assert fine_tune_lab.__version__ == "0.0.0"


def test_dataset_record_validates() -> None:
    record = DatasetRecord(
        id="abc-123",
        input=ClaimInput(claim_text="Office visit, established patient"),
        label=Label.CLEAN,
        source="synthetic",
    )
    assert record.label is Label.CLEAN
    assert record.input.claim_text


def test_needs_human_review_is_a_label() -> None:
    # The escalation fallback class must exist (contract requirement).
    assert Label.NEEDS_HUMAN_REVIEW.value == "needs_human_review"
