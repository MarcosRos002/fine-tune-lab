# Classification IO contract — the drop-in surface

Status: **DRAFT for Phase 1.** This is the contract that makes the cheap model substitutable for
the Claude call inside `claims-auditor`'s classification module. **Both repos must agree on it.**

> Authoritative pairing: https://github.com/MarcosRos002/claims-auditor (classification module).
> If this file and claims-auditor disagree, they are both broken — reconcile before shipping.

## Why this exists

claims-auditor today classifies a claim by calling a frontier model. fine-tune-lab's product is a
small model that answers the **same request** with the **same response shape**, cheaper and faster.
For the swap to be invisible to the rest of claims-auditor, the request and response schemas below
must hold regardless of which backend (Claude vs fine-tuned vs distilled) is wired in.

## Request (model input)

```json
{
  "claim_text": "string — normalized claim/billing line description",
  "metadata": {
    "cpt_code": "string | null",
    "icd_code": "string | null",
    "amount_usd": "number | null"
  }
}
```

This is exactly the `input` object from `dataset.md`. The training pipeline must serialize prompts
from this object the same way the served model expects at inference (single source of truth in
`serve/`).

## Response (model output)

```json
{
  "label": "string — one of the agreed label enum (see dataset.md)",
  "confidence": "number — 0.0..1.0",
  "rationale": "string | null — short justification (optional; off by default for cost)"
}
```

### Rules

- `label` **must** be a member of the agreed enum. Out-of-enum output is a contract violation; the
  serving layer maps any unexpected output to `needs_human_review` and logs it.
- `confidence` is required and calibrated enough to drive an escalation threshold in claims-auditor.
- `rationale` is optional; emit it only when claims-auditor asks (it costs tokens/latency).
- Output is **strict JSON**, no prose wrapper. The served model is constrained/parsed to guarantee
  this; behavior-fine-tuning is largely *for* this format guarantee.

## Substitutability checklist (used by eval)

A candidate model is a valid drop-in when, on `test.jsonl`:
1. 100% of outputs parse as the response schema above.
2. 100% of `label` values are in-enum.
3. Accuracy ≥ baseline (Claude Haiku) within the agreed tolerance.
4. p50 latency and cost/request are reported and lower than baseline.

These four feed directly into the before/after table in the README.
