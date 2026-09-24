"""Local pretrained NLI baseline.

DeBERTa-v3-large fine-tuned on MNLI + FEVER-NLI + ANLI + WANLI. The model takes
(evidence as premise, claim as hypothesis) and returns softmax probabilities over
entailment / neutral / contradiction, which map onto SUPPORTS / NEI / REFUTES.
For binary datasets the NEI mass is dropped and the vector renormalised, the
same accounting the JEV conditions use.

Runs fully locally (Apple Silicon MPS); latency is measured per call.
"""
from __future__ import annotations

import json
import os
import time

from prompts import render_evidence

MODEL_NAME = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
SHORT_NAME = "deberta-v3-large-mnli-fever-anli"
LABEL_MAP = {"entailment": "SUPPORTS", "neutral": "NEI", "contradiction": "REFUTES"}


class NliClient:
    def __init__(self, model: str = MODEL_NAME, device: str | None = None):
        from transformers import pipeline

        device = device or os.environ.get("NLI_DEVICE", "mps")

        self.model = model
        self.pipe = pipeline("text-classification", model=model, device=device)

    def score(self, claim: str, evidence: str, allowed: tuple[str, ...], max_chars: int = 6000) -> dict:
        premise = render_evidence(evidence, max_chars)
        t0 = time.time()
        scores = self.pipe({"text": premise, "text_pair": claim},
                           truncation=True, max_length=512, top_k=None)
        latency_ms = (time.time() - t0) * 1000
        raw = {LABEL_MAP[s["label"].lower()]: float(s["score"]) for s in scores}
        total = sum(raw.get(label, 0.0) for label in allowed)
        distribution = {label: raw.get(label, 0.0) / total for label in allowed} if total else {}
        prediction = max(distribution, key=distribution.get) if distribution else None
        return {
            "prediction": prediction,
            "distribution": distribution,
            "raw": json.dumps(raw),
            "usage": {},
            "latency_ms": latency_ms,
            "model_version": SHORT_NAME,
            "finish_reason": "local",
            "meta": {"nli_scores": raw, "truncated_to": 512},
        }
