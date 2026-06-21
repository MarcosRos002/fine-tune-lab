"""Evaluation: accuracy + per-class metrics, latency, and the parse/review rates
on the held-out test split. See ``docs/modules/eval.md``.

The model is any callable ``(prompt: str) -> raw_output: str`` (a served LoRA /
distilled model, or the Claude baseline). Outputs are constrained through the
shared ``serve.io`` contract, so non-JSON / out-of-enum replies count as
``needs_human_review`` rather than crashing eval.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path
from time import perf_counter
from typing import Any

from sklearn.metrics import accuracy_score, classification_report

from fine_tune_lab.data.schema import DatasetRecord, Label
from fine_tune_lab.serve.io import parse_output, serialize_prompt


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    rank = max(1, math.ceil(p / 100 * len(s)))
    return float(s[rank - 1])


def evaluate(model: Callable[[str], str], dataset_path: str) -> dict[str, Any]:
    """Evaluate a model on a held-out ``*.jsonl`` split."""
    records = [
        DatasetRecord.model_validate_json(line)
        for line in Path(dataset_path).read_text().splitlines()
        if line
    ]

    y_true: list[str] = []
    y_pred: list[str] = []
    latencies: list[float] = []
    review = 0
    for r in records:
        prompt = serialize_prompt(r.input)
        t0 = perf_counter()
        raw = model(prompt)
        latencies.append((perf_counter() - t0) * 1000.0)
        pred = parse_output(raw).label
        y_true.append(r.label.value)
        y_pred.append(pred.value)
        if pred is Label.NEEDS_HUMAN_REVIEW:
            review += 1

    n = len(records)
    return {
        "n": n,
        "accuracy": accuracy_score(y_true, y_pred) if n else 0.0,
        "per_class": classification_report(y_true, y_pred, output_dict=True, zero_division=0),
        "latency_ms": {
            "mean": (sum(latencies) / n) if n else 0.0,
            "p50": _percentile(latencies, 50),
            "p95": _percentile(latencies, 95),
        },
        "needs_human_review_rate": (review / n) if n else 0.0,
    }


def main() -> None:  # pragma: no cover - CLI
    """CLI entry point (``make eval``)."""
    raise SystemExit("Provide a served model; see docs/modules/eval.md")


if __name__ == "__main__":  # pragma: no cover
    main()
