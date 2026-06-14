# Module: train

## Purpose
Fine-tune a small open base model on the classification dataset using **LoRA/QLoRA** (PEFT), to
teach the **behavior and output format** of `docs/contracts/classification_io.md`. Produces a small
adapter (optionally merged) that `serve/` and `eval/` consume.

## Interface
- `lora.py` — `train_lora(config)`: load base + tokenizer, attach LoRA/QLoRA adapter, run the
  supervised loop on `train.jsonl`, evaluate on `val.jsonl`, save the adapter.
  Entry point: `python -m fine_tune_lab.train.lora` (`make train`). Real runs go on Colab/Kaggle.
- `config.py` — `TrainConfig`: base model id, LoRA rank/alpha/dropout, target modules, 4-bit flag,
  lr, epochs, batch/accum, output dir. Kept swappable so the base model is a config choice.

## Dependencies
- `transformers`, `peft`, `accelerate`, `bitsandbytes` (QLoRA 4-bit), `datasets`.
- Upstream: `data/` (dataset contract). Downstream: `eval/`, `serve/`.

## How to test
- Unit-test config construction and prompt serialization (must match `serve/` and `data/`).
- A tiny smoke "train" on a few rows / 1 step to verify the loop wires up and saves an adapter.
- Heavy training is not unit-tested; it runs in the notebooks against the free GPU.

## Senior concerns
- **Overfitting** to synthetic data — watch val loss, keep adapters low-rank, early-stop.
- **Train/serve prompt skew** — one serialization function, shared with `serve/`.
- **Fine-tune for form, not facts** (see ADR 0001) — don't try to teach knowledge here.
- **Reproducibility** — pin seeds and config; the notebook is a thin driver over this module.
