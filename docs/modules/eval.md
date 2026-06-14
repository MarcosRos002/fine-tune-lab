# Module: eval

## Purpose
Produce the **headline before/after table**: accuracy, p50 latency, and cost/request for
(a) Claude Haiku baseline, (b) fine-tuned LoRA model, (c) distilled model — on the held-out
`test.jsonl`. This is the product. Cost accounting is a first-class metric.

## Interface
- `metrics.py` — `evaluate(model, dataset)`: run predictions, parse against
  `docs/contracts/classification_io.md`, compute accuracy + per-class metrics + latency + cost,
  return a structured result. Entry point: `python -m fine_tune_lab.eval.metrics` (`make eval`).
- `cost.py` — `cost_per_request(...)`: token/$ accounting for API baseline vs self-hosted vLLM
  (amortized GPU-hours). Keeps the dollar figures honest and comparable.
- `report.py` — render the before/after table (markdown) for the README.

## Dependencies
- `scikit-learn` (accuracy, per-class P/R/F1, confusion matrix), `pydantic` (output validation).
- Upstream: contracts (only!) + any model. **Can be built before any model exists** (use baseline +
  stub), then re-run on real artifacts. Downstream: README table.

## How to test
- Unit-test metric computation on a known-label fixture (assert accuracy/F1 values).
- Unit-test the substitutability checklist: 100% parse rate, 100% in-enum labels (see contract).
- Unit-test cost accounting math for both API and self-hosted paths.

## Senior concerns
- **Eval rigor** — held-out `test.jsonl` only; report per-class metrics, not just top-line accuracy.
- **Cost honesty** — amortize GPU cost realistically; compare like-for-like throughput.
- **Latency** — report p50 and p95; a low mean can hide bad tails.
- **Apples-to-apples** — identical prompts/inputs across all three variants.
