"""Request/response models + shared prompt serialization for serving.

The serialization here is the SINGLE source of truth shared with ``data/`` and
``train/`` to avoid train/serve skew. Output is strict JSON; out-of-enum or
non-JSON output maps to ``needs_human_review``. Must match
``docs/contracts/classification_io.md``.

Phase 0 stub.
"""

from __future__ import annotations

from pydantic import BaseModel

from fine_tune_lab.data.schema import ClaimInput, Label


class ClassificationRequest(BaseModel):
    """Model input — mirrors the dataset ``input`` object."""

    claim_text: str
    metadata: dict | None = None


class ClassificationResponse(BaseModel):
    """Strict model output. See ``docs/contracts/classification_io.md``."""

    label: Label
    confidence: float
    rationale: str | None = None


def serialize_prompt(claim: ClaimInput) -> str:
    """Serialize a claim input into the model prompt.

    Shared by data prep, training, and serving to guarantee no train/serve skew.
    """
    ...


def parse_output(raw: str) -> ClassificationResponse:
    """Parse/constrain raw model output to ``ClassificationResponse``.

    Non-JSON or out-of-enum output is mapped to ``needs_human_review`` and logged,
    never raised to the caller.
    """
    ...
