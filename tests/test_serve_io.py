"""Tests for the serving IO: prompt serialization + robust output parsing.

This is the single source of truth that prevents train/serve skew and makes the
cheap model a drop-in for the Claude call in claims-auditor.
"""

from __future__ import annotations

from fine_tune_lab.data.schema import ClaimInput, ClaimMetadata, Label
from fine_tune_lab.serve.io import parse_output, serialize_prompt


def _claim():
    return ClaimInput(
        claim_text="Office visit, established patient, high complexity",
        metadata=ClaimMetadata(cpt_code="99215", icd_code="E11.9", amount_usd=210.0),
    )


def test_serialize_prompt_includes_claim_and_codes() -> None:
    p = serialize_prompt(_claim())
    assert "99215" in p and "E11.9" in p
    assert "Office visit" in p
    assert "upcoding_suspected" in p  # the label options are offered to the model


def test_parse_valid_json() -> None:
    out = parse_output('{"label": "upcoding_suspected", "confidence": 0.91}')
    assert out.label is Label.UPCODING_SUSPECTED
    assert out.confidence == 0.91


def test_parse_json_embedded_in_prose() -> None:
    raw = 'Sure! Here is my answer: {"label": "clean", "confidence": 0.7} hope that helps'
    assert parse_output(raw).label is Label.CLEAN


def test_non_json_maps_to_needs_human_review() -> None:
    out = parse_output("I cannot determine this claim")
    assert out.label is Label.NEEDS_HUMAN_REVIEW
    assert out.confidence == 0.0


def test_out_of_enum_label_maps_to_needs_human_review() -> None:
    out = parse_output('{"label": "definitely_fraud", "confidence": 0.99}')
    assert out.label is Label.NEEDS_HUMAN_REVIEW


def test_confidence_is_clamped() -> None:
    assert parse_output('{"label": "clean", "confidence": 5}').confidence == 1.0
    assert parse_output('{"label": "clean", "confidence": -2}').confidence == 0.0
