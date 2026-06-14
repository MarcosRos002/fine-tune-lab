"""vLLM serving entry point.

Loads the merged adapter / distilled model and exposes the classification IO
contract so the cheap model drops into claims-auditor. See ``docs/modules/serve.md``.

Phase 0 stub. (``vllm`` is an optional extra — Linux+GPU only.)
"""

from __future__ import annotations


def serve(model_path: str = "artifacts/lora", host: str = "0.0.0.0", port: int = 8000) -> None:
    """Start a vLLM endpoint serving the classification contract.

    Args:
        model_path: path to the merged adapter or distilled model.
        host: bind address.
        port: bind port.
    """
    ...


def main() -> None:
    """CLI entry point (``make serve``)."""
    ...


if __name__ == "__main__":
    main()
