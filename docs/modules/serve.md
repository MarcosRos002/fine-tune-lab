# Module: serve

## Purpose
Serve the trained adapter / distilled model behind the **classification IO contract**
(`docs/contracts/classification_io.md`) so it is a drop-in replacement for the Claude call in
claims-auditor. Uses **vLLM** for fast, batched inference; enforces strict-JSON output.

## Interface
- `vllm_server.py` — `serve(config)`: load the model (merged adapter or distilled), start a vLLM
  endpoint exposing the request/response schema. Entry point:
  `python -m fine_tune_lab.serve.vllm_server` (`make serve`).
- `io.py` — request/response Pydantic models + the **shared prompt serialization** (same function
  used by `train/` and `data/` to avoid train/serve skew) and output parsing/constraining.

## Dependencies
- `vllm` (optional extra — Linux+GPU), `transformers`, `pydantic`.
- Upstream: a trained/distilled artifact + the IO contract. Downstream: claims-auditor consumes it.

## How to test
- Unit-test `io.py`: serialization round-trips; out-of-enum / non-JSON output maps to
  `needs_human_review` and is logged.
- Contract test: every response validates against `classification_io.md`.
- vLLM startup is integration-only (needs GPU); guard behind a marker.

## Senior concerns
- **Strict output** — never return prose; constrain/parse to the JSON schema. This format guarantee
  is the main reason we fine-tuned at all.
- **Train/serve skew** — one serialization function shared across modules.
- **Graceful degradation** — unparseable output escalates to human review, not a crash.
- **Throughput vs cost** — batch sizing on free/cheap GPUs feeds the eval cost numbers.
