# Architecture — fine-tune-lab

## Purpose

Produce a **cheap, fast classifier** that is a drop-in replacement for the LLM call in
`claims-auditor`'s classification step, via the progression **Prompt → RAG → Fine-tune → Distill**.

## The progression (and where this repo lives)

```
Prompt  ── frontier model + good prompt. Cheapest to build, priciest to run. = TEACHER + BASELINE
  │
RAG     ── inject missing FACTS via retrieval. (Owned by claims-auditor, not here.)
  │
Fine-tune ── teach a small open model the BEHAVIOR/FORMAT with LoRA/QLoRA.   <-- this repo
  │
Distill ── compress the teacher's judgment into the small model.            <-- this repo
```

fine-tune-lab owns the **Fine-tune** and **Distill** legs. It deliberately does **not** try to
solve "missing knowledge" problems — those are RAG/prompt problems upstream in claims-auditor.

## Data flow

```
                       ┌───────────────────────────────────────────┐
                       │            claims-auditor (flagship)        │
                       │  classification step (currently a Claude    │
                       │  call) emits CLASSIFICATION TRACES          │
                       └───────────────┬─────────────────────────────┘
                                       │  traces (input + label + rationale)
                                       ▼
        ┌──────────────┐      ┌─────────────────┐
        │ Teacher model │────▶│   data/          │  build_dataset.py
        │ (frontier LLM)│ labels│ synthetic claim │  - normalize to dataset schema
        └──────────────┘ +soft │ classification  │  - train / val / test splits
                          targets│ examples (JSONL)│  - NEVER real patient data
                               └────────┬────────┘
                                        │  dataset (conforms to docs/contracts/dataset.schema)
                ┌───────────────────────┼───────────────────────┐
                ▼                       ▼                        ▼
        ┌──────────────┐        ┌──────────────┐         ┌──────────────┐
        │   train/     │        │   distill/   │         │    eval/     │
        │ LoRA/QLoRA   │        │ teacher→stud.│         │ accuracy +   │
        │ (PEFT) →     │        │ soft targets │         │ latency +    │
        │ adapter      │        │ → student    │         │ $/request    │
        └──────┬───────┘        └──────┬───────┘         └──────┬───────┘
               │  adapter / merged      │  distilled model       │  before/after TABLE
               └───────────┬────────────┘                        │
                           ▼                                      ▼
                   ┌──────────────┐                       (headline artifact:
                   │   serve/     │                        Haiku vs LoRA vs Distill)
                   │ vLLM serves  │
                   │ the model on │
                   │ the SAME IO  │◀── must satisfy docs/contracts/classification_io
                   │ contract     │     so it drops into claims-auditor unchanged
                   └──────────────┘
```

## Key design points

- **Contract-first boundary.** The dataset and the model's input/output are pinned in
  `docs/contracts/`. The served model is only useful if it matches claims-auditor's classification
  IO exactly — that contract is the integration surface between the two repos.
- **Teacher = baseline.** The same frontier model that labels the data is also the accuracy bar
  and the cost ceiling we are beating.
- **Two training tracks share one dataset.** `train/` (supervised LoRA) and `distill/` (soft
  targets from the teacher) both consume the dataset contract and can progress in parallel.
- **Eval is the product.** The before/after table (accuracy, latency, cost/request) is the
  deliverable; cost accounting is a first-class metric, not a footnote.
- **Free-tier only.** Training happens on Colab/Kaggle notebooks; the repo code is the reusable
  library those notebooks import.

## Module → doc map

| Module | Code | Doc |
|---|---|---|
| data | `src/fine_tune_lab/data/` | `docs/modules/data.md` |
| train | `src/fine_tune_lab/train/` | `docs/modules/train.md` |
| distill | `src/fine_tune_lab/distill/` | `docs/modules/distill.md` |
| eval | `src/fine_tune_lab/eval/` | `docs/modules/eval.md` |
| serve | `src/fine_tune_lab/serve/` | `docs/modules/serve.md` |
