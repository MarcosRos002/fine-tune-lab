# Dataset contract — classification training data

Status: **DRAFT for Phase 1.** Labels and label set are illustrative until the first data-prep
pass against real claims-auditor traces. Lock this before training runs at scale.

## Format

- One JSON object per line (**JSONL**), UTF-8.
- Three splits as separate files: `train.jsonl`, `val.jsonl`, `test.jsonl`.
- **No real patient data.** Every record is synthetic or derived from claims-auditor traces with
  identifiers removed. PHI must never appear.

## Record schema

```json
{
  "id": "string — stable unique id (e.g. uuid4)",
  "input": {
    "claim_text": "string — the normalized claim/billing line description",
    "metadata": {
      "cpt_code": "string | null",
      "icd_code": "string | null",
      "amount_usd": "number | null"
    }
  },
  "label": "string — one of the enum in `labels` below",
  "rationale": "string | null — teacher's short justification (used by distillation)",
  "source": "string — 'teacher' | 'claims-auditor-trace' | 'synthetic'",
  "teacher_logprobs": "object | null — optional soft targets per label, for distillation"
}
```

### Field notes

- `input` is the **only** thing the served model sees at inference time. It must be reproducible
  from claims-auditor's classification input (see `classification_io.md`).
- `rationale` and `teacher_logprobs` are **training-only signal** from the teacher. They power
  distillation (`distill/`) and are absent at inference.
- `source` lets eval slice quality by data origin and lets us audit class balance.

## Label set (`labels`) — illustrative

| label | meaning |
|---|---|
| `clean` | claim is well-formed and consistent; no audit flag |
| `upcoding_suspected` | billed code appears higher-intensity than documentation supports |
| `unbundling_suspected` | services billed separately that should be bundled |
| `missing_documentation` | required supporting documentation absent/insufficient |
| `duplicate` | likely duplicate of a previously billed line |
| `needs_human_review` | ambiguous; escalate (fallback class) |

> The exact enum is owned jointly with claims-auditor's classification module. Treat the values
> above as a starting point; the authoritative list is whatever both repos agree to in
> `classification_io.md`.

## Splits & balance

- Stratify splits by `label` so rare classes appear in `val`/`test`.
- Record and report class balance in the dataset card produced by `data/`.
- Hold out `test.jsonl` from all training/distillation; it backs the eval table.

## Validation

`data/schema.py` defines the Pydantic models for these records; `make prep-data` validates every
row against them before writing. A row that fails validation is dropped and logged, never coerced.
