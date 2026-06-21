"""Tests for the eval framework: cost accounting, metrics, before/after table."""

from __future__ import annotations

import json

from fine_tune_lab.data.build_dataset import build_dataset, generate_records
from fine_tune_lab.eval.cost import cost_per_request
from fine_tune_lab.eval.metrics import evaluate
from fine_tune_lab.eval.report import render_table
from fine_tune_lab.serve.io import serialize_prompt


# --------------------------------------------------------------------------- #
# cost
# --------------------------------------------------------------------------- #
def test_api_cost_follows_token_pricing() -> None:
    # 1000 reqs, 200k input + 50k output tokens at $0.80/$4.00 per Mtok.
    c = cost_per_request(
        n_requests=1000,
        input_tokens=200_000,
        output_tokens=50_000,
        backend="api",
        price_in_per_mtok=0.80,
        price_out_per_mtok=4.00,
    )
    expected = (200_000 * 0.80 + 50_000 * 4.00) / 1_000_000 / 1000
    assert abs(c - expected) < 1e-12


def test_self_hosted_vllm_is_cheaper_at_scale() -> None:
    api = cost_per_request(
        n_requests=1000, input_tokens=200_000, output_tokens=50_000, backend="api"
    )
    vllm = cost_per_request(
        n_requests=1000,
        input_tokens=200_000,
        output_tokens=50_000,
        backend="vllm",
        gpu_usd_per_hour=1.10,
        throughput_rps=50.0,
    )
    assert vllm < api


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def _write_testset(tmp_path):
    build_dataset(out_dir=tmp_path, n=40, seed=2)
    return tmp_path / "test.jsonl"


def test_perfect_model_scores_accuracy_one(tmp_path) -> None:
    test_path = _write_testset(tmp_path)
    # An oracle: returns the true label for each record (keyed by its prompt).
    records = generate_records(40, seed=2)
    truth = {serialize_prompt(r.input): r.label.value for r in records}

    def oracle(prompt: str) -> str:
        return json.dumps({"label": truth[prompt], "confidence": 0.9})

    m = evaluate(oracle, str(test_path))
    assert m["accuracy"] == 1.0
    assert m["latency_ms"]["p50"] >= 0
    assert m["n"] > 0


def test_garbage_model_has_low_accuracy_and_high_review_rate(tmp_path) -> None:
    test_path = _write_testset(tmp_path)

    def garbage(_prompt: str) -> str:
        return "I have no idea what this is"

    m = evaluate(garbage, str(test_path))
    assert m["accuracy"] == 0.0
    assert m["needs_human_review_rate"] == 1.0


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #
def test_render_table_has_variants_and_headers() -> None:
    results = {
        "Claude Haiku (baseline)": {
            "accuracy": 0.91,
            "latency_ms": {"p50": 420.0},
            "cost_per_1k": 1.30,
        },
        "Fine-tuned (LoRA)": {"accuracy": 0.89, "latency_ms": {"p50": 35.0}, "cost_per_1k": 0.06},
    }
    table = render_table(results)
    assert "Claude Haiku (baseline)" in table
    assert "Fine-tuned (LoRA)" in table
    assert "Accuracy" in table and "atency" in table  # latency header
    assert "0.91" in table and "35" in table
