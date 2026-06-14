.PHONY: install test lint format prep-data train distill eval serve help

help:
	@echo "Targets:"
	@echo "  install    Install package + dev dependencies (editable)"
	@echo "  test       Run pytest"
	@echo "  lint       Run ruff check"
	@echo "  format     Run ruff format"
	@echo "  prep-data  Build the synthetic classification dataset (stub)"
	@echo "  train      LoRA/QLoRA training entrypoint (stub; real runs on Colab/Kaggle)"
	@echo "  distill    Distillation entrypoint (stub)"
	@echo "  eval       Produce the before/after metrics table (stub)"
	@echo "  serve      Serve the adapter/distilled model via vLLM (stub)"

install:
	python -m pip install -e ".[dev]"

test:
	pytest

lint:
	ruff check .

format:
	ruff format .

prep-data:
	python -m fine_tune_lab.data.build_dataset

train:
	python -m fine_tune_lab.train.lora

distill:
	python -m fine_tune_lab.distill.distill

eval:
	python -m fine_tune_lab.eval.metrics

serve:
	python -m fine_tune_lab.serve.vllm_server
