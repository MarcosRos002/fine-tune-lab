"""Request/response models + shared prompt serialization for serving.

The serialization here is the SINGLE source of truth shared with ``data/`` and
``train/`` to avoid train/serve skew. Output is strict JSON; out-of-enum or
non-JSON output maps to ``needs_human_review``. Must match
``docs/contracts/classification_io.md``.

Phase 0 stub.
"""

from __future__ import annotations

import json
import re

from pydantic import BaseModel

from fine_tune_lab.data.schema import ClaimInput, Label

_VALID_LABELS = {lbl.value for lbl in Label}


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
    m = claim.metadata
    labels = ", ".join(lbl.value for lbl in Label)
    amount = f"{m.amount_usd:.2f}" if m.amount_usd is not None else "NA"
    return (
        "Classify this medical billing claim line into exactly one label.\n"
        f"Claim: {claim.claim_text}\n"
        f"CPT: {m.cpt_code or 'NA'} | ICD: {m.icd_code or 'NA'} | Amount USD: {amount}\n"
        f'Respond with JSON only: {{"label": <one of [{labels}]>, "confidence": <0.0-1.0>}}'
    )


def parse_output(raw: str) -> ClassificationResponse:
    """Parse/constrain raw model output to ``ClassificationResponse``.

    Non-JSON or out-of-enum output is mapped to ``needs_human_review`` and logged,
    never raised to the caller.
    """
    review = ClassificationResponse(label=Label.NEEDS_HUMAN_REVIEW, confidence=0.0)
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return review
    try:
        data = json.loads(match.group(0))
    except (json.JSONDecodeError, ValueError):
        return review
    if not isinstance(data, dict) or data.get("label") not in _VALID_LABELS:
        return review
    try:
        confidence = float(data.get("confidence", 0.5))
    except (TypeError, ValueError):
        confidence = 0.5
    return ClassificationResponse(
        label=Label(data["label"]),
        confidence=max(0.0, min(1.0, confidence)),
        rationale=data.get("rationale"),
    )
