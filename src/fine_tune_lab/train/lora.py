"""LoRA training loop for the claim classifier (PEFT, sequence classification).

A small encoder + a LoRA adapter learns the closed label set. The heavy stack
(``torch``/``transformers``/``peft``/``datasets``) is imported **lazily** so the
light dev env can import this module (and unit-test the data plumbing) without
installing torch. Real training runs on Colab/Kaggle GPU (see notebooks) or on
CPU with a tiny base for a smoke run. See ``docs/modules/train.md``.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from fine_tune_lab.data.schema import DatasetRecord, Label
from fine_tune_lab.serve.io import serialize_prompt
from fine_tune_lab.train.config import TrainConfig

# NEEDS_HUMAN_REVIEW is a serving fallback, never a trained class.
TRAIN_LABELS: list[str] = [lbl.value for lbl in Label if lbl is not Label.NEEDS_HUMAN_REVIEW]


def label_maps() -> tuple[dict[str, int], dict[int, str]]:
    """Return (label2id, id2label) for the trained classes."""
    label2id = {name: i for i, name in enumerate(TRAIN_LABELS)}
    id2label = dict(enumerate(TRAIN_LABELS))
    return label2id, id2label


def load_split(dataset_dir: str | Path, split: str) -> tuple[list[str], list[int]]:
    """Read a ``*.jsonl`` split into (prompt texts, label ids) — no torch needed.

    Uses ``serialize_prompt`` so training text matches serving exactly (no skew).
    """
    label2id, _ = label_maps()
    texts: list[str] = []
    labels: list[int] = []
    for line in (Path(dataset_dir) / f"{split}.jsonl").read_text().splitlines():
        if not line:
            continue
        record = DatasetRecord.model_validate_json(line)
        texts.append(serialize_prompt(record.input))
        labels.append(label2id[record.label.value])
    return texts, labels


def train_lora(
    dataset_dir: str | Path = "data/processed", config: TrainConfig | None = None
) -> str:
    """Run LoRA training over the classification dataset; save + return the adapter dir."""
    config = config or TrainConfig()
    from datasets import Dataset
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    set_seed(config.seed)
    label2id, id2label = label_maps()
    texts, labels = load_split(dataset_dir, "train")

    tokenizer = AutoTokenizer.from_pretrained(config.base_model)

    def _tok(batch):
        return tokenizer(
            batch["text"], truncation=True, max_length=config.max_length, padding="max_length"
        )

    train_ds = Dataset.from_dict({"text": texts, "labels": labels}).map(_tok, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        config.base_model, num_labels=len(label2id), id2label=id2label, label2id=label2id
    )
    model = get_peft_model(
        model,
        LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=config.lora_rank,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=config.target_modules,
        ),
    )

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=config.output_dir,
            num_train_epochs=config.num_epochs,
            per_device_train_batch_size=config.batch_size,
            learning_rate=config.learning_rate,
            logging_steps=20,
            report_to=[],
            seed=config.seed,
            save_strategy="no",
        ),
        train_dataset=train_ds,
    )
    trainer.train()
    model.save_pretrained(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)
    return config.output_dir


def build_predictor(
    adapter_dir: str | Path, *, base_model: str | None = None, max_length: int = 128
) -> Callable[[str], str]:
    """Load a trained adapter into a ``(prompt) -> JSON string`` predictor.

    The JSON output satisfies the ``serve.io`` classification contract, so the
    served LoRA model is a drop-in for the Claude call in claims-auditor.
    """
    import json

    import torch
    from peft import PeftModel
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    label2id, id2label = label_maps()
    base = base_model or TrainConfig().base_model
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
    model = AutoModelForSequenceClassification.from_pretrained(
        base, num_labels=len(label2id), id2label=id2label, label2id=label2id
    )
    model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()

    def predict(prompt: str) -> str:
        enc = tokenizer(prompt, truncation=True, max_length=max_length, return_tensors="pt")
        with torch.no_grad():
            probs = torch.softmax(model(**enc).logits[0], dim=-1)
        idx = int(probs.argmax())
        return json.dumps({"label": id2label[idx], "confidence": round(float(probs[idx]), 4)})

    return predict


def main() -> None:  # pragma: no cover - CLI
    """CLI entry point (``make train``)."""
    print(train_lora())


if __name__ == "__main__":  # pragma: no cover
    main()
