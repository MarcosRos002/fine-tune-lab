"""Render the before/after metrics table (markdown) for the README.

See ``docs/modules/eval.md``. Phase 0 stub.
"""

from __future__ import annotations

from typing import Any


def render_table(results: dict[str, dict[str, Any]]) -> str:
    """Render the Haiku-vs-LoRA-vs-distilled before/after table as markdown.

    Args:
        results: mapping of variant name -> metrics dict from ``evaluate``.

    Returns:
        Markdown table string.
    """
    ...
