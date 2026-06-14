"""Pydantic models for the classification dataset records.

Single source of truth for validation. Must match ``docs/contracts/dataset.md``.
Phase 0 stub — fields below are illustrative; finalize the label enum jointly
with claims-auditor's classification module before training at scale.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel


class Label(str, Enum):
    """Classification label enum. Owned jointly with claims-auditor.

    See ``docs/contracts/dataset.md`` — treat as a starting point.
    """

    CLEAN = "clean"
    UPCODING_SUSPECTED = "upcoding_suspected"
    UNBUNDLING_SUSPECTED = "unbundling_suspected"
    MISSING_DOCUMENTATION = "missing_documentation"
    DUPLICATE = "duplicate"
    NEEDS_HUMAN_REVIEW = "needs_human_review"


class ClaimMetadata(BaseModel):
    """Structured metadata accompanying a claim line."""

    cpt_code: str | None = None
    icd_code: str | None = None
    amount_usd: float | None = None


class ClaimInput(BaseModel):
    """The only object the served model sees at inference time."""

    claim_text: str
    metadata: ClaimMetadata = ClaimMetadata()


class DatasetRecord(BaseModel):
    """One training/eval example. See ``docs/contracts/dataset.md``."""

    id: str
    input: ClaimInput
    label: Label
    rationale: str | None = None
    source: str
    teacher_logprobs: dict[str, float] | None = None
