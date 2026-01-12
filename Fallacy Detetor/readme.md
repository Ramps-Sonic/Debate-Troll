# Fallacy Detection Agent (debatetroll)

This folder provides a packaged logical fallacy detector agent.
It takes input text and outputs a structured "fallacy_signal" JSON.

## What you get
- `run_agent.py`: the only entrypoint you need
- `inference.py`: model loading + sliding-window inference
- `model.py`, `train.py`, `merge_lora.py`, `dataset.py`, `config.py`: training utilities (optional)

## Setup

### 1) Create environment
Python >= 3.9 recommended.

```bash
pip install -r requirements.txt
