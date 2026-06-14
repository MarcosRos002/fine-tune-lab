# ADR 0001 — Stack and scope

- Status: Accepted
- Date: 2026-06-14
- Deciders: Marcos Rostan

## Context

fine-tune-lab must produce a cheap classifier to replace a frontier-model call in claims-auditor,
using only **free-tier** compute, and serve as a portfolio demonstration of senior fine-tuning
judgment. We need to fix the stack and, just as importantly, the **scope** — including when this
technique should *not* be used.

## Decision

**Technique**
- **PEFT with LoRA / QLoRA** (HuggingFace `peft`). LoRA for a small base; **QLoRA (4-bit)** when a
  larger base is needed to fit free-tier VRAM (bitsandbytes). We tune adapters, not full weights.
- **Knowledge distillation** from a frontier teacher: teacher generates labels + rationales +
  (where available) soft targets; the small student learns to imitate the teacher's judgment.
- **Base model:** a small open model (e.g. Qwen2.5-small or Llama-3.x-small). Final choice deferred
  to Phase 1 based on license + VRAM + tokenizer fit; kept swappable behind config.

**Serving**
- **vLLM** for fast, batched inference of the adapter/merged or distilled model. Output is parsed/
  constrained to the strict JSON in `docs/contracts/classification_io.md`.

**Compute (free-tier only)**
- **Google Colab (T4)** and **Kaggle (30h/week)** as primary training environments, driven by the
  notebooks in `notebooks/`. **HF ZeroGPU** / **Modal ($30 credit)** as overflow for serving/eval.
- The repo is the importable library; notebooks are thin drivers so training is reproducible.

**Data**
- Synthetic claim-classification examples; labels from the teacher or claims-auditor traces.
  **Never real patient data.** Schema pinned in `docs/contracts/dataset.md`.

**Tooling**
- Python ≥ 3.11, `ruff` (lint/format), `pytest` (tests), `pydantic` (contract validation).

## When NOT to fine-tune (senior signal — read this before training anything)

Fine-tuning is frequently the **wrong** first move. The mature default order is
**Prompt → RAG → Fine-tune → Distill**, and you should exhaust the cheaper, more reversible options
first.

**Fine-tune for FORM, not for FACTS.**
- Fine-tuning reliably teaches **behavior, style, and output format** (e.g. "always emit this exact
  JSON, choose from this label set, escalate when unsure").
- Fine-tuning is a **poor and risky** way to inject **facts/knowledge**. Facts go stale, bloat the
  training set, and the model still hallucinates between them. **Use RAG or the prompt for facts.**

**Prefer the cheaper lever first:**
1. **Prompting** — if a better prompt / few-shot examples / output schema fixes it, stop here.
2. **RAG** — if the failure is "missing or changing knowledge," retrieve it. Don't bake it in.
3. **Fine-tune** — only when the *task shape is stable* and the remaining problems are
   **format/behavior consistency or cost/latency** (exactly our case: stable classification task,
   need it cheap and reliably formatted).
4. **Distill** — once fine-tuning works, compress teacher judgment to push quality at fixed cost.

**Do NOT fine-tune when:**
- The task or label set is still changing weekly — you'll retrain constantly.
- The real need is fresh/external facts — that's RAG.
- A prompt tweak already meets quality — fine-tuning adds ops cost for no gain.
- You can't measure it — without a held-out eval and cost accounting you can't prove improvement,
  and you risk overfitting to the training distribution.

**Risks we explicitly manage:** overfitting to synthetic data, train/serve prompt skew,
class imbalance on rare audit labels, and teacher-bias propagation through distillation. The
`eval/` module exists specifically to keep these honest (held-out `test.jsonl`, per-class metrics,
cost/latency reported alongside accuracy).

## Consequences

- We can train and serve entirely on free tiers, producing a defensible before/after table.
- Adapter-based tuning keeps artifacts tiny and swappable; the base model stays a config choice.
- The "when NOT to fine-tune" framing is itself a portfolio deliverable — it signals judgment, not
  just the ability to run a training loop.
- Binding to claims-auditor's classification IO contract is a hard dependency; the cheap model is
  only valuable if it remains a true drop-in.
