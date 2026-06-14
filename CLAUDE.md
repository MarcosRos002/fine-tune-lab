# CLAUDE.md — fine-tune-lab

> **This is the best entry point for a fresh Claude Code session.** Read this top-to-bottom,
> then follow the pointers into `docs/`. Phase 0 (scaffold + context) is complete; the next
> concrete work is in `docs/context/handoff.md`.

## What this repo is

**fine-tune-lab** is a LoRA/QLoRA fine-tuning + knowledge-distillation lab. Its concrete,
non-academic goal: **produce a cheap small model that serves the CLASSIFICATION step of the
flagship project (`claims-auditor`, "Veritas") at a fraction of the cost of calling Claude,
at comparable quality.**

The headline artifact is a **before/after table**:

| Variant | Accuracy | p50 latency | Cost / 1k requests |
|---|---|---|---|
| (a) Claude Haiku baseline | _coming_ | _coming_ | _coming_ |
| (b) Fine-tuned small model (LoRA) | _coming_ | _coming_ | _coming_ |
| (c) Distilled model | _coming_ | _coming_ | _coming_ |

This repo demonstrates the mature progression **Prompt → RAG → Fine-tune → Distill**, and the
senior framing that **you fine-tune for BEHAVIOR / FORMAT, not for FACTS** (facts belong in
RAG or the prompt). See `docs/adr/0001-stack-and-scope.md` for the full "when NOT to fine-tune".

## Role in the program

This repo's output (the cheap classifier) is a **drop-in replacement** for the LLM call in
`claims-auditor`'s classification module. The input/output contract that makes this swap safe
lives in `docs/contracts/` and **must stay in sync** with claims-auditor's classification IO.

## Sibling repos (part of a 4-repo program)
- claims-auditor (flagship): https://github.com/MarcosRos002/claims-auditor
- agent-lens (eval/observability): https://github.com/MarcosRos002/agent-lens
- fine-tune-lab (LoRA/distillation): https://github.com/MarcosRos002/fine-tune-lab
- portfolio (website): https://github.com/MarcosRos002/portfolio

Relationship: claims-auditor is measured by agent-lens, fed a cheap model by fine-tune-lab, and exhibited in portfolio.

## Repository map

```
src/fine_tune_lab/
  data/     # build classification dataset from teacher labels / claims-auditor traces
  train/    # LoRA/QLoRA training (PEFT)
  distill/  # knowledge distillation from a frontier teacher
  eval/     # before/after metrics: accuracy, latency, cost/request
  serve/    # vLLM serving of the adapter / distilled model
notebooks/  # colab_lora.ipynb, kaggle_qlora.ipynb — run training on FREE GPU
tests/
docs/
  architecture.md            # Prompt→RAG→Fine-tune→Distill pipeline + ASCII data flow
  contracts/                 # dataset schema + classification IO contract (shared w/ claims-auditor)
  adr/0001-stack-and-scope.md
  modules/                   # one .md per module (data, train, distill, eval, serve)
  orchestration.md           # build order + parallelization + git worktrees
  context/handoff.md         # what's done, what's next
```

## Conventions

- **Python ≥ 3.11.** Package is `fine-tune-lab`, import as `fine_tune_lab`.
- **Contract-first.** Touching dataset shape or model IO? Update `docs/contracts/` in the same change.
- **Docs are context infrastructure.** Each module has a `docs/modules/*.md`. Keep them current.
- **Never real patient data.** Training data is synthetic claim-classification examples whose
  labels come from the teacher model or from claims-auditor's classification traces. No PHI, ever.
- **Lint/format:** `ruff`. **Tests:** `pytest`. See `pyproject.toml`. Run via the `Makefile`.
- **Cost accounting is a first-class output**, not an afterthought — every eval reports $/request.

## How to run training on a FREE GPU

GPU is free-tier only. Pick one:
- **Google Colab (T4)** — open `notebooks/colab_lora.ipynb`. Best for LoRA on a small base model.
- **Kaggle (30h/week)** — open `notebooks/kaggle_qlora.ipynb`. Best for QLoRA (4-bit) on larger bases.
- **HF ZeroGPU** or **Modal ($30 credit)** — alternatives for serving/eval bursts.

Local (CPU/no-GPU) workflow for development:
```bash
make install   # install package + dev deps
make test      # run pytest
make lint      # ruff check
make prep-data # build the classification dataset (stub)
make train     # LoRA training entrypoint (stub; real runs go on Colab/Kaggle)
```

## Where to go next

1. `docs/context/handoff.md` — current status and the immediate next task.
2. `docs/architecture.md` — the pipeline and data flow.
3. `docs/contracts/` — the dataset schema and the classification IO contract (the load-bearing part).
4. `docs/adr/0001-stack-and-scope.md` — why this stack, and when NOT to fine-tune.
5. `docs/orchestration.md` — build order and how to parallelize with git worktrees.
