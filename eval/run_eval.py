"""Run one verification system over the unified claim-evidence corpora.

Every request is logged with the raw response, token usage, and latency so the
result tables can be rebuilt from the JSONL alone.
"""
from __future__ import annotations

import argparse
import collections
import sys
import concurrent.futures as cf
import json
import random
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from prompts import (
    DIGIT_TO_LABEL,
    cot_messages,
    direct_messages,
    jev_payload,
    parse_cot,
    parse_direct,
)
from providers import JevClient, QwenClient, load_secrets

ROOT = Path(__file__).resolve().parent.parent
UNIFIED = ROOT / "data" / "unified"
RESULTS = ROOT / "results" / "predictions"

SYSTEMS = [
    "jev",
    "nli",
    "qwen-direct-compact",
    "qwen-direct-expanded",
    "qwen-cot-compact",
    "qwen-cot-expanded",
]

lock = threading.Lock()


def slug(text: str) -> str:
    return text.replace("/", "-").replace(".", "-").lower()


def allowed_labels(dataset: str) -> tuple[str, ...]:
    """Decision space actually present in a dataset, so a binary benchmark is not
    penalised for lacking an NEI class."""
    path = UNIFIED / f"{dataset}.jsonl"
    labels = []
    for line in path.open():
        if line.strip():
            label = json.loads(line)["label"]
            if label not in labels:
                labels.append(label)
    return tuple(label for label in ("NEI", "SUPPORTS", "REFUTES") if label in labels)


def load_records(datasets: list[str], per_dataset: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    records: list[dict] = []
    for dataset in datasets:
        path = UNIFIED / f"{dataset}.jsonl"
        if not path.exists():
            raise SystemExit(f"missing unified file {path}; run data/build_unified.py first")
        rows = [json.loads(line) for line in path.open() if line.strip()]
        rows = [row for row in rows if row["split"] in {"dev", "test"}] or rows
        by_label: dict[str, list[dict]] = collections.defaultdict(list)
        for row in rows:
            by_label[row["label"]].append(row)
        picked: list[dict] = []
        per_label = max(1, per_dataset // max(1, len(by_label)))
        for label, bucket in sorted(by_label.items()):
            rng.shuffle(bucket)
            picked.extend(bucket[:per_label])
        rng.shuffle(picked)
        records.extend(picked[:per_dataset])
    return records


def score_jev(client: JevClient, record: dict, max_chars: int, allowed: tuple[str, ...]) -> dict:
    state, questions = jev_payload(record["claim"], record["evidence"], max_chars, allowed)
    result = client.score(state, questions)
    probs = result["probabilities"]
    p_support = float(probs.get("supports", 0.0))
    if "REFUTES" in allowed:
        p_refute = float(probs.get("refutes", 0.0))
        p_nei = max(0.0, 1.0 - p_support - p_refute) if "NEI" in allowed else 0.0
    else:
        p_refute = 1.0 - p_support
        p_nei = 0.0
    total = p_support + p_refute + p_nei or 1.0
    distribution = {label: value / total for label, value in
                    (("SUPPORTS", p_support), ("REFUTES", p_refute), ("NEI", p_nei)) if label in allowed}
    prediction = max(distribution, key=distribution.get)
    return {
        "prediction": prediction,
        "distribution": distribution,
        "raw": result["raw"][:4000],
        "usage": result["usage"],
        "latency_ms": None,
        "model_version": result.get("model"),
    }


def score_qwen(client: QwenClient, record: dict, system: str, model: str, max_tokens: int, max_chars: int,
               allowed: tuple[str, ...]) -> dict:
    expanded = system.endswith("expanded")
    cot = "-cot-" in system
    messages = (
        cot_messages(record["claim"], record["evidence"], expanded, allowed, max_chars)
        if cot
        else direct_messages(record["claim"], record["evidence"], expanded, allowed, max_chars)
    )
    started = time.time()
    result = client.chat(model, messages, max_tokens=max_tokens, enable_thinking=False)
    latency_ms = result["latency_ms"] if result.get("latency_ms") else (time.time() - started) * 1000.0
    text = result["texts"][0] if result["texts"] else ""
    prediction = (parse_cot if cot else parse_direct)(text)
    if prediction not in allowed:
        prediction = None
    return {
        "prediction": prediction,
        "distribution": None,
        "raw": text[:4000],
        "usage": result["usage"],
        "latency_ms": latency_ms,
        "model_version": model,
        "finish_reason": (result.get("finish_reasons") or [None])[0],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--systems", nargs="+", default=["jev"], choices=SYSTEMS)
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--datasets", nargs="+", default=["fever", "scifact", "hover", "vitaminc", "climate_fever"])
    parser.add_argument("--per-dataset", type=int, default=200)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--direct-max-tokens", type=int, default=1)
    parser.add_argument("--max-evidence-chars", type=int, default=6000)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--tag", default=None, help="output file suffix, e.g. pilot")
    args = parser.parse_args()

    secrets = load_secrets()
    allowed_by_dataset = {name: allowed_labels(name) for name in args.datasets}
    print(f"label spaces: {allowed_by_dataset}", flush=True)
    records = load_records(args.datasets, args.per_dataset, args.seed)
    by_dataset = collections.Counter(r["dataset"] for r in records)
    print(f"planned {len(records)} items: {dict(by_dataset)}", flush=True)

    RESULTS.mkdir(parents=True, exist_ok=True)
    jev = JevClient(secrets)
    qwen = QwenClient(secrets)
    nli = None
    if "nli" in args.systems:
        from nli_client import MODEL_NAME, NliClient, SHORT_NAME

        local = ROOT / "models" / "deberta-nli"
        nli = NliClient(model=str(local) if local.exists() else MODEL_NAME)
        nli.model = SHORT_NAME

    for system in args.systems:
        suffix = f"__{args.tag}" if args.tag else ""
        log_model = jev.model if system == "jev" else (nli.model if system == "nli" else args.model)
        out_path = RESULTS / f"{system}__{slug(log_model)}{suffix}.jsonl"
        done: set[str] = set()
        if out_path.exists():
            for line in out_path.open():
                if not line.strip():
                    continue
                previous = json.loads(line)
                # Only treat a row as finished when it produced a usable prediction,
                # so rate-limited rows are retried on the next invocation.
                if previous.get("prediction"):
                    done.add(previous["id"])
        todo = [record for record in records if record["id"] not in done]
        print(f"[{system}] {len(todo)} to run, {len(done)} cached -> {out_path.name}", flush=True)

        def work(record: dict) -> dict:
            started = time.time()
            try:
                allowed = allowed_by_dataset[record["dataset"]]
                if system == "jev":
                    payload = score_jev(jev, record, args.max_evidence_chars, allowed)
                elif system == "nli":
                    payload = nli.score(record["claim"], record["evidence"], allowed, args.max_evidence_chars)
                else:
                    max_tokens = args.max_tokens if "-cot-" in system else args.direct_max_tokens
                    payload = score_qwen(qwen, record, system, args.model, max_tokens, args.max_evidence_chars, allowed)
                error = None
            except Exception as exc:  # noqa: BLE001 - recorded per item
                payload = {"prediction": None, "distribution": None, "raw": "", "usage": {}, "latency_ms": None}
                error = f"{type(exc).__name__}: {exc}"[:400]
            return {
                "id": record["id"],
                "dataset": record["dataset"],
                "split": record["split"],
                "system": system,
                "model": payload.get("model_version") or args.model,
                "gold": record["label"],
                "prediction": payload["prediction"],
                "distribution": payload["distribution"],
                "raw": payload["raw"],
                "usage": payload["usage"],
                "latency_ms": payload["latency_ms"] if payload["latency_ms"] is not None else (time.time() - started) * 1000.0,
                "error": error,
                "finish_reason": payload.get("finish_reason"),
                "claim": record["claim"],
                "n_evidence": len(record["evidence"]),
                "evidence_chars": sum(len(e) for e in record["evidence"]),
            }

        started = time.time()
        written = 0
        with out_path.open("a") as handle:
            # The local HF pipeline is not thread-safe; NLI runs sequentially.
            stream = (work(record) for record in todo) if system == "nli" else None
            if stream is None:
                pool = cf.ThreadPoolExecutor(args.concurrency)
                stream = pool.map(work, todo)
            for payload in stream:
                with lock:
                    handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
                written += 1
                if written % 25 == 0:
                    rate = written / max(time.time() - started, 1e-6)
                    print(f"  {written}/{len(todo)} ({rate:.2f}/s)", flush=True)
        print(f"[{system}] finished {written} in {time.time() - started:.0f}s", flush=True)


if __name__ == "__main__":
    main()
