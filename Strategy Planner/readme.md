# StrategyPlanner Agent (DebateTroll) — Usage

This module generates a **structured rebuttal strategy plan** (JSON) given:
- an opponent argument text
- a detected fallacy type + confidence

It does **NOT** generate the final rebuttal essay. Instead it outputs:
- primary strategy
- 2–4 concrete attack steps
- risk assessment + mitigation
- success probability
- timing advice
- evidence needs

## Files You Need
To run the agent, keep these files in the same folder:
- run_agent.py  (the only entry script you run)
- config.py
- llm_client.py
- schema.py
- strategy_planner.py
- prompts.py
- templates.py

## Install
Python 3.8+ recommended.

```bash
pip install -U requests
