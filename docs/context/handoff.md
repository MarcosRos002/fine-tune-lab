# Handoff — context for the next session

## Status: Phase 0 complete

Phase 0 was **context-readiness only** — scaffold, documentation, contracts, and minimal stubs.
No features are implemented yet. The repo is structured so a fresh Claude Code session has full
context to start building.

### What exists now
- `CLAUDE.md` — entry point: what this repo is, its role, conventions, how to run on free GPU.
- `README.md` — recruiter-facing: hook, before/after table (placeholders), Prompt→RAG→Fine-tune→Distill.
- `docs/architecture.md` — pipeline + ASCII data flow.
- `docs/contracts/` — **dataset schema** + **classification IO contract** (the load-bearing part).
- `docs/adr/0001-stack-and-scope.md` — stack decision + "when NOT to fine-tune".
- `docs/orchestration.md` — build order, parallelization, git worktrees.
- `docs/modules/*.md` — one per module (data, train, distill, eval, serve).
- `src/fine_tune_lab/**` — package layout with stub functions/classes (docstrings + `...`).
- `pyproject.toml`, `Makefile`, `notebooks/` — config + free-GPU notebooks.

## Next steps (in order)

1. **Define the dataset contract for real.** Finalize the label enum jointly with claims-auditor's
   classification module, then implement `data/schema.py` (Pydantic) to match `docs/contracts/dataset.md`.
2. **Build data prep.** Implement `data/build_dataset.py`: pull teacher labels / claims-auditor
   traces, normalize to the schema, validate, write stratified `train/val/test.jsonl` + a dataset card.
   Run via `make prep-data`.
3. **LoRA training notebook on Colab.** Flesh out `notebooks/colab_lora.ipynb` to import `train/`
   and produce a LoRA adapter on a small base (T4). Keep the loop in `src/...train/lora.py`.
4. **Eval table.** Implement `eval/metrics.py`: Claude Haiku baseline vs LoRA vs distilled —
   accuracy, p50 latency, $/1k requests. Fill the README before/after table with real numbers.
5. Then **distill** (`notebooks/kaggle_qlora.ipynb` + `distill/`), then **serve** (`serve/` vLLM).

## Watch out for
- Keep `docs/contracts/` in sync with claims-auditor — the cheap model must remain a drop-in.
- Never introduce real patient data; synthetic only.
- Build `eval/` early (it only needs the contract) and report cost alongside accuracy.
- Free-tier compute only (Colab T4 / Kaggle 30h / ZeroGPU / Modal $30).
