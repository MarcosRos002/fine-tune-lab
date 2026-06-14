# Module: distill

## Purpose
Compress the **teacher's judgment** into the small student model via knowledge distillation, to push
quality up at the same low inference cost. Uses teacher labels + rationales and, where available,
soft targets (`teacher_logprobs`) from the dataset.

## Interface
- `distill.py` — `distill(config)`: load student (optionally a LoRA-tuned checkpoint from `train/`),
  train against teacher soft targets / rationale-augmented examples, save the distilled model.
  Entry point: `python -m fine_tune_lab.distill.distill` (`make distill`). Runs on Kaggle/Colab.
- `config.py` — `DistillConfig`: teacher source, temperature, soft/hard-target loss weights, student
  base/checkpoint, training hyperparameters.

## Dependencies
- `transformers`, `peft`, `datasets`, `accelerate`.
- Upstream: `data/` (dataset with `rationale` / `teacher_logprobs`). Downstream: `eval/`, `serve/`.

## How to test
- Unit-test the loss combination (soft vs hard target weighting, temperature scaling).
- Smoke-distill on a tiny fixture to verify the loop and checkpoint save.

## Senior concerns
- **Teacher-bias propagation** — the student inherits the teacher's mistakes; eval on held-out data.
- **Soft-target availability** — if `teacher_logprobs` are absent, fall back to rationale/hard-label
  distillation; document which mode produced the artifact.
- **Diminishing returns** — distillation earns its place only if the eval table shows it beats plain
  LoRA at equal cost; otherwise ship the LoRA model.
