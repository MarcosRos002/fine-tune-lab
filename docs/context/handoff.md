# Handoff — context for the next session

## Status: Phase 1 — light core implemented (no-GPU parts)

The CPU-runnable core is built and tested (32 tests): the IO contract, synthetic
dataset generation, and the full eval framework (the headline before/after table).
The LoRA/distillation training is the Colab/Kaggle part (code + notebooks), not run
here to keep the dev env light (torch lives in the flagship venv).

### Done (Phase 1)
- `serve/io.py`: `serialize_prompt` (single source of truth, no train/serve skew) +
  `parse_output` (robust JSON extraction; non-JSON / out-of-enum → `needs_human_review`).
- `data/build_dataset.py`: `generate_records` (deterministic, balanced) + `build_dataset`
  (stratified disjoint `train/val/test.jsonl` + dataset card). Synthetic, no PHI.
- `eval/cost.py`: `cost_per_request` (API token pricing vs amortized vLLM GPU-hours).
- `eval/metrics.py`: `evaluate(model, dataset_path)` — accuracy (sklearn), per-class
  P/R/F1, latency p50/p95, `needs_human_review_rate`.
- `eval/report.py`: `render_table` — the Haiku-vs-LoRA-vs-distilled markdown table.
- Verified: a self-hosted vLLM variant comes out ~29× cheaper/1k than the API baseline.

### Phase 0 baseline (still valid)
Scaffold, docs, contracts, stubs — see `CLAUDE.md` + `docs/`.

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

## LoRA training implemented (Phase 1)
`train/lora.py` (`train_lora` + `build_predictor`, sequence-classification LoRA via PEFT, lazy
heavy imports) + `train/config.py` (encoder default `google/bert_uncased_L-2_H-128_A-2`). Torch-free
plumbing (`label_maps`, `load_split`) pinned by `tests/test_train.py`. **Real CPU smoke run:**
BERT-tiny LoRA, 150 steps/~11s → **0.67 test accuracy** (naive baseline 0.20). README table has the
real fine-tuned row. `notebooks/colab_lora.ipynb` is a runnable T4 driver (distilbert).

## Next steps (in order)
1. **Run on Colab T4** with `colab_lora.ipynb` (distilbert) for a higher-accuracy adapter; update the
   README row with the GPU numbers.
2. **Distill** (`notebooks/kaggle_qlora.ipynb` + `distill/`), then **serve** (`serve/vllm_server.py`).
3. **Wire into claims-auditor**: the served model becomes the Pass-1 classifier behind the
   `ClassifierModel` seam (the classification IO contract guarantees the swap is invisible).

## Watch out for
- Keep `docs/contracts/` in sync with claims-auditor — the cheap model must remain a drop-in.
- Never introduce real patient data; synthetic only.
- Build `eval/` early (it only needs the contract) and report cost alongside accuracy.
- Free-tier compute only (Colab T4 / Kaggle 30h / ZeroGPU / Modal $30).
