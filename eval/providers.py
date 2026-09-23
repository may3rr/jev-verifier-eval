"""Thin clients for the two model families compared in the paper."""
from __future__ import annotations

import json
import os
import pathlib
import re
import time
import urllib.error
import urllib.request

JEV_URL = os.environ.get("TYPESAFE_SYSTEMONE_URL", "https://api.typesafe.ai/v1/systemone")
QWEN_URL = os.environ.get("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1/chat/completions")

SECRET_KEYS = ("TYPESAFE_API_KEY", "JEV_MODEL", "SILICONFLOW_API_KEY")


def load_secrets() -> dict[str, str]:
    values: dict[str, str] = {}
    zshrc = pathlib.Path.home() / ".zshrc"
    if zshrc.exists():
        pattern = re.compile(r"\s*export\s+([A-Z_][A-Z0-9_]*)=(.*)$")
        for line in zshrc.read_text(errors="ignore").splitlines():
            match = pattern.match(line)
            if match:
                values[match.group(1)] = match.group(2).strip().strip('"').strip("'")
    for key in SECRET_KEYS:
        if os.environ.get(key):
            values[key] = os.environ[key]
    return values


def _post(url: str, payload: dict, key: str, timeout: int, retries: int = 7) -> dict:
    body = json.dumps(payload, ensure_ascii=False).encode()
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        method="POST",
    )
    last: Exception | None = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode())
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:400]
            if exc.code in (429, 500, 502, 503, 504):
                last = RuntimeError(f"HTTP {exc.code}: {detail}")
                time.sleep(min(3 * 2**attempt, 30) + 0.5 * attempt)
                continue
            raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc
        except Exception as exc:  # noqa: BLE001 - network flake, retried
            last = exc
            time.sleep(min(2**attempt, 12) + 0.3 * attempt)
    raise RuntimeError(f"request failed after {retries} attempts: {last}")


class JevClient:
    """Typed System One client. Returns calibrated probabilities for two propositions."""

    def __init__(self, secrets: dict[str, str] | None = None, timeout: int = 90) -> None:
        secrets = secrets or load_secrets()
        if not secrets.get("TYPESAFE_API_KEY"):
            raise SystemExit("TYPESAFE_API_KEY is not configured")
        self.key = secrets["TYPESAFE_API_KEY"]
        self.model = secrets.get("JEV_MODEL") or "jev-latest"
        self.timeout = timeout
        self.observed_model: str | None = None

    def score(self, state: dict, questions: dict) -> dict:
        payload = {"model": self.model, "state": state, "questions": questions}
        data = _post(JEV_URL, payload, self.key, self.timeout)
        self.observed_model = data.get("model", self.observed_model)
        answers = data.get("answers", {})
        probabilities = {}
        for name, answer in answers.items():
            value = answer.get("noul") if isinstance(answer, dict) else answer
            if isinstance(value, (int, float)):
                probabilities[name] = float(value)
        usage = data.get("usage", {}) or {}
        return {
            "probabilities": probabilities,
            "usage": usage,
            "raw": json.dumps(data, ensure_ascii=False),
            "model": data.get("model"),
        }


class QwenClient:
    """OpenAI-compatible chat client for the small Qwen checkpoints."""

    def __init__(self, secrets: dict[str, str] | None = None, timeout: int = 120) -> None:
        secrets = secrets or load_secrets()
        if not secrets.get("SILICONFLOW_API_KEY"):
            raise SystemExit("SILICONFLOW_API_KEY is not configured")
        self.key = secrets["SILICONFLOW_API_KEY"]
        self.timeout = timeout

    def chat(
        self,
        model: str,
        messages: list[dict],
        max_tokens: int,
        temperature: float = 0.0,
        enable_thinking: bool = False,
        samples: int = 1,
    ) -> dict:
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "n": samples,
            "enable_thinking": enable_thinking,
        }
        started = time.time()
        data = _post(QWEN_URL, payload, self.key, self.timeout)
        latency_ms = (time.time() - started) * 1000.0
        choices = data.get("choices", [])
        texts = [choice.get("message", {}).get("content") or "" for choice in choices]
        usage = data.get("usage", {}) or {}
        finish = [choice.get("finish_reason") for choice in choices]
        return {
            "texts": texts,
            "usage": usage,
            "latency_ms": latency_ms,
            "finish_reasons": finish,
            "raw": json.dumps(data, ensure_ascii=False),
        }
