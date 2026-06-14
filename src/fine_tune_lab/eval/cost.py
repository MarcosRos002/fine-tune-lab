"""Cost accounting — a first-class eval metric.

Compares the API baseline (token pricing) against self-hosted vLLM (amortized
GPU-hours) on a like-for-like basis. See ``docs/modules/eval.md``.

Phase 0 stub.
"""

from __future__ import annotations


def cost_per_request(
    *,
    n_requests: int,
    input_tokens: int,
    output_tokens: int,
    backend: str = "api",
) -> float:
    """Return estimated USD cost per request for the given backend.

    Args:
        n_requests: number of requests measured.
        input_tokens: total input tokens across the run.
        output_tokens: total output tokens across the run.
        backend: ``'api'`` (token pricing) or ``'vllm'`` (amortized GPU-hours).

    Returns:
        USD cost per request.
    """
    ...
