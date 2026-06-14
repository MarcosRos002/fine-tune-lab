# fine-tune-lab

**Turn an expensive Claude call into a cheap, fast, fine-tuned model — without losing quality.**

`fine-tune-lab` is a LoRA/QLoRA fine-tuning and knowledge-distillation lab. Its concrete goal is
to produce a small, cheap model that serves the **classification step** of the flagship
[`claims-auditor`](https://github.com/MarcosRos002/claims-auditor) ("Veritas") at a fraction of
the cost of calling a frontier model — at comparable accuracy.

It demonstrates the full senior progression **Prompt → RAG → Fine-tune → Distill**, and the
discipline behind it: you fine-tune for **behavior and format**, not for **facts**.

---

## The headline result

The whole project earns its keep with one table (real numbers land as training completes):

| Variant | Accuracy | p50 latency | Cost / 1k requests |
|---|---|---|---|
| Claude Haiku baseline | _coming_ | _coming_ | _coming_ |
| Fine-tuned small model (LoRA) | _coming_ | _coming_ | _coming_ |
| Distilled model | _coming_ | _coming_ | _coming_ |

> Goal: match the baseline's accuracy at a small fraction of its cost and latency.

---

## The narrative: Prompt → RAG → Fine-tune → Distill

1. **Prompt** — start with a frontier model (Claude) and a good prompt. Cheapest to build,
   most expensive to run. This is the **teacher** and the **baseline** to beat.
2. **RAG** — when the model needs *facts* it doesn't have, retrieve them. (Handled upstream in
   claims-auditor; fine-tuning does **not** replace retrieval.)
3. **Fine-tune** — once the *task shape* is stable, teach a small open model the
   **behavior and output format** with LoRA/QLoRA. Cheap to run.
4. **Distill** — compress the teacher's judgment into the small model using teacher-generated
   labels (soft targets / rationales), pushing quality up at the same low cost.

Senior framing: **fine-tune for form, not facts.** If the failure is "missing knowledge," reach
for RAG. If the failure is "wrong format / inconsistent behavior / too expensive," fine-tune.

---

## How to reproduce on a free GPU

No paid hardware required. Everything runs on free tiers.

- **Google Colab (T4):** open [`notebooks/colab_lora.ipynb`](notebooks/colab_lora.ipynb).
- **Kaggle (30h/week):** open [`notebooks/kaggle_qlora.ipynb`](notebooks/kaggle_qlora.ipynb).
- **Alternatives:** HF ZeroGPU, Modal ($30 credit).

Local dev (no GPU needed for data prep / eval scaffolding):

```bash
make install   # install package + dev deps
make prep-data # build synthetic classification dataset from teacher labels
make test      # run the test suite
make lint      # ruff
```

Training data is **100% synthetic** claim-classification examples — labels come from the teacher
model or from claims-auditor classification traces. **No real patient data, ever.**

---

## Links

- Architecture & data flow: [`docs/architecture.md`](docs/architecture.md)
- Dataset schema + classification IO contract: [`docs/contracts/`](docs/contracts/)
- Stack decision & "when NOT to fine-tune": [`docs/adr/0001-stack-and-scope.md`](docs/adr/0001-stack-and-scope.md)
- For contributors / AI agents: [`CLAUDE.md`](CLAUDE.md)

## Part of a 4-repo portfolio program

- [claims-auditor](https://github.com/MarcosRos002/claims-auditor) — flagship multimodal medical-billing audit agent ("Veritas")
- [agent-lens](https://github.com/MarcosRos002/agent-lens) — eval + observability for agents
- **fine-tune-lab** (this repo) — LoRA/QLoRA + distillation
- [portfolio](https://github.com/MarcosRos002/portfolio) — Next.js website exhibiting all three

Relationship: claims-auditor is measured by agent-lens, fed a cheap model by fine-tune-lab, and exhibited in portfolio.

## License

MIT — see [LICENSE](LICENSE).
