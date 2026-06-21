"""Render the before/after metrics table (markdown) for the README.

See ``docs/modules/eval.md``.
"""

from __future__ import annotations

from typing import Any


def _fmt(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def render_table(results: dict[str, dict[str, Any]]) -> str:
    """Render the Haiku-vs-LoRA-vs-distilled before/after table as markdown.

    Each value in ``results`` is a metrics dict (from ``evaluate``) optionally
    augmented with ``cost_per_1k``. Returns a markdown table string.
    """
    header = "| Variant | Accuracy | p50 latency (ms) | Cost / 1k req (USD) |"
    sep = "|---|---|---|---|"
    rows = [header, sep]
    for variant, m in results.items():
        accuracy = _fmt(m.get("accuracy"))
        p50 = _fmt((m.get("latency_ms") or {}).get("p50"))
        cost = _fmt(m.get("cost_per_1k"))
        rows.append(f"| {variant} | {accuracy} | {p50} | {cost} |")
    return "\n".join(rows)
