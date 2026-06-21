"""Cost accounting — a first-class eval metric.

Compares the API baseline (token pricing) against self-hosted vLLM (amortized
GPU-hours) on a like-for-like basis. The headline of fine-tune-lab is that a small
self-hosted model serves the same classification far cheaper at scale.
See ``docs/modules/eval.md``.
"""

from __future__ import annotations


def cost_per_request(
    *,
    n_requests: int,
    input_tokens: int,
    output_tokens: int,
    backend: str = "api",
    price_in_per_mtok: float = 0.80,
    price_out_per_mtok: float = 4.00,
    gpu_usd_per_hour: float = 1.10,
    throughput_rps: float = 50.0,
) -> float:
    """Return estimated USD cost per request for the given backend.

    - ``api``: token pricing — ``(in*price_in + out*price_out) / 1e6 / n_requests``.
    - ``vllm``: amortized GPU-hours — ``gpu_usd_per_hour / (throughput_rps * 3600)``
      (per request; tokens don't bill directly when you own the GPU).
    """
    if backend == "api":
        total = (input_tokens * price_in_per_mtok + output_tokens * price_out_per_mtok) / 1_000_000
        return total / n_requests
    if backend == "vllm":
        return gpu_usd_per_hour / (throughput_rps * 3600.0)
    raise ValueError(f"unknown backend: {backend!r}")
