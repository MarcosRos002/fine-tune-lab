# Module: data

## Purpose
Build the synthetic claim-classification dataset that every other module consumes. Sources are the
**teacher model** (frontier LLM labels) and **claims-auditor classification traces**. Output is
stratified `train/val/test.jsonl` conforming to `docs/contracts/dataset.md`, plus a dataset card.

## Interface
- `schema.py` — Pydantic models for the record schema (the single source of truth for validation).
- `build_dataset.py` — `build_dataset(...)`: load sources → normalize → validate → split → write.
  Entry point: `python -m fine_tune_lab.data.build_dataset` (`make prep-data`).

## Dependencies
- `pydantic` (validation), `datasets` (IO/splitting), `scikit-learn` (stratified split).
- Upstream: teacher labels + claims-auditor traces. Downstream: train, distill, eval.

## How to test
- Unit-test `schema.py` with valid/invalid records (invalid rows are dropped + logged, never coerced).
- Test stratified splitting preserves class balance and that `test.jsonl` is disjoint from train/val.
- Golden-file a tiny synthetic fixture through `build_dataset` and assert the output schema.

## Senior concerns
- **No PHI, ever.** Strip identifiers from any trace-derived record; assert this in tests.
- **Class imbalance** on rare audit labels — stratify and report balance in the dataset card.
- **Train/serve skew** — the prompt serialization used here must match `serve/` exactly.
- **Leakage** — keep `test.jsonl` fully held out from all training and distillation.
