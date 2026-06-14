# Contracts — fine-tune-lab

These contracts are the **load-bearing** part of the repo. Two things must stay pinned:

1. **`dataset.md`** — the schema of the synthetic classification dataset that `train/` and
   `distill/` consume and that `data/` produces.
2. **`classification_io.md`** — the input/output contract of the *served model*. This **must
   match `claims-auditor`'s classification module** so the cheap model is a drop-in replacement
   for the Claude call there.

## Rule

If you change the dataset shape or the model IO, update the matching file here **in the same
change**, and check whether `claims-auditor`'s classification module needs a corresponding update.
The whole value proposition ("swap the expensive call for a cheap model") depends on these two
contracts agreeing across both repos.

- Flagship classification module to stay in sync with:
  https://github.com/MarcosRos002/claims-auditor
