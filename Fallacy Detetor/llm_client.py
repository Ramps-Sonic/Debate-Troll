import os
import json
from typing import List, Dict, Any, Optional

import requests


class LLMClient:
    """
    Simple OpenAI-compatible chat completions client using requests.
    Works for many providers that expose OpenAI-compatible endpoints.

    Expected endpoint:
      POST {api_base}/v1/chat/completions
    """

    def __init__(self, api_key: str, api_base: str, model: str, timeout: int = 120):
        self.api_key = api_key
        self.api_base = api_base.rstrip("/")
        self.model = model
        self.timeout = timeout

    def chat(self, messages: List[Dict[str, str]], temperature: float = 0.4, max_tokens: int = 800) -> str:
        url = f"{self.api_base}/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]


def build_client_from_env(default_model: str = "qwen-max") -> LLMClient:
    """
    Use env vars:
      LLM_API_KEY
      LLM_API_BASE  (e.g. https://api.openai.com or your gateway)
      LLM_MODEL
    """
    api_key = os.environ.get("LLM_API_KEY", "")
    api_base = os.environ.get("LLM_API_BASE", "")
    model = os.environ.get("LLM_MODEL", default_model)
    if not api_key or not api_base:
        raise RuntimeError("Missing env vars: LLM_API_KEY and/or LLM_API_BASE.")
    return LLMClient(api_key=api_key, api_base=api_base, model=model)
