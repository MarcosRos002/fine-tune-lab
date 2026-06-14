# Orchestration — build order & parallelization

How to build fine-tune-lab efficiently, and how to split work across agents/worktrees without
stepping on the contracts.

## Dependency graph

```
        ┌──────────────────────────────────────────────┐
        │ FOUNDATIONAL (blocks everything) — do first    │
        │   1. docs/contracts/dataset.md                 │
        │   2. docs/contracts/classification_io.md       │
        │   3. src/.../data  (build_dataset + schema)    │
        └───────────────────────┬────────────────────────┘
                                │ dataset contract is frozen
        ┌───────────────────────┼───────────────────────┐
        ▼                       ▼                        ▼
   ┌─────────┐            ┌──────────┐             ┌──────────┐
   │ train/  │            │ distill/ │             │  eval/   │   PARALLEL
   │ LoRA    │            │ teacher→ │             │ metrics  │   (against the
   │ /QLoRA  │            │ student  │             │ harness  │    same contract)
   └────┬────┘            └────┬─────┘             └────┬─────┘
        └───────────┬──────────┘                       │
                    ▼                                   │
              ┌──────────┐                              │
              │  serve/  │  LAST — needs a trained/      │
              │  vLLM    │  distilled artifact + the     │
              └──────────┘  classification_io contract ◀─┘
```

## Phases

1. **Foundational (blocking).** Lock the two contracts, then implement `data/` (schema +
   `build_dataset`). Nothing else can be validated until the dataset shape is real. Output:
   `train.jsonl` / `val.jsonl` / `test.jsonl` + a dataset card.
2. **Parallel.** With the dataset contract frozen, `train/`, `distill/`, and `eval/` can all
   progress independently:
   - `train/` consumes the dataset, emits a LoRA/QLoRA adapter.
   - `distill/` consumes the dataset (+ teacher soft targets/rationales), emits a distilled model.
   - `eval/` can be built against the contract immediately, using the Haiku baseline + a stub model,
     then re-run on real artifacts. Build the metrics harness early; it's the product.
3. **Serve (last).** `serve/` wraps a finished artifact in vLLM, enforcing
   `classification_io.md`. It needs at least one trained artifact and the frozen IO contract.

## Why eval can start early

`eval/` only needs the **contract**, not a finished model. Build the harness against the baseline
and a stub, lock the metrics (accuracy, p50 latency, $/1k requests), then point it at real
adapters/distilled models as they land. This keeps the headline table honest from day one.

## Parallel work with git worktrees

Because the parallel modules share only the (frozen) contracts, they're a natural fit for separate
worktrees — each agent gets an isolated checkout, no merge thrash on shared files.

```bash
# from the main checkout, one worktree per parallel stream
git worktree add ../ftl-train  -b feat/train-lora
git worktree add ../ftl-distill -b feat/distill
git worktree add ../ftl-eval   -b feat/eval-harness

# work in each independently, then merge back to main
git worktree list
git worktree remove ../ftl-train   # when the branch is merged
```

Rules for parallel streams:
- **Do not edit `docs/contracts/` in a parallel branch.** Contract changes are foundational; make
  them on `main` (or a dedicated contract branch) and rebase the streams onto the new contract.
- Each stream owns exactly one module directory under `src/fine_tune_lab/` + its `docs/modules/*.md`.
- `eval/` and `serve/` depend on the contract only — keep that dependency one-directional.
