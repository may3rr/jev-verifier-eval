#!/usr/bin/env python3
"""JEV service time on the Qwen latency sample (150 cases, concurrency 1).

Qwen and NLI latency is measured on the machine that serves them, so it holds no
network time. JEV is only reachable over the internet. To put it on the same
footing, every case is sent over one kept-alive connection and followed at once
by a request without credentials, which the API gateway rejects without running
the model. Both travel the same path, so the difference between the two is the
time the service spends on the judgement. A pair that had to open a new
connection is repeated, so handshakes never enter the numbers. A single pair is
at the mercy of network jitter, so every case is measured in five rounds and
keeps the median difference.

Writes results/sensitivity/jev__jev-latest__lat1.jsonl with the usual row fields;
latency_ms is the service time, e2e_ms and rtt_ms are the medians of the raw timings.
"""
from __future__ import annotations

import collections
import json
import statistics
import os
import sys
import time
from pathlib import Path

import urllib3

sys.path.insert(0, str(Path(__file__).resolve().parent))

from prompts import jev_payload
from providers import JEV_URL, load_secrets
from run_eval import allowed_labels, load_records

ROOT = Path(__file__).resolve().parent.parent
SENS = ROOT / "results" / "sensitivity"
SAMPLE = SENS / "qwen-direct-compact__qwen-qwen3-5-9b__vllm-lat1.jsonl"
OUT = SENS / "jev__jev-latest__lat1.jsonl"
DATASETS = ["fever", "scifact", "hover", "vitaminc", "climate_fever"]
MAX_CHARS = 6000
WARMUP = 3
ROUNDS = 5


def open_pool():
    proxy = os.environ.get("https_proxy") or os.environ.get("HTTPS_PROXY")
    manager = urllib3.ProxyManager(proxy) if proxy else urllib3.PoolManager()
    return manager.connection_from_url(JEV_URL)


def timed_post(pool, body: bytes, headers: dict) -> tuple[float, urllib3.BaseHTTPResponse]:
    started = time.perf_counter()
    response = pool.urlopen("POST", urllib3.util.parse_url(JEV_URL).path, body=body, headers=headers,
                            retries=False, preload_content=True)
    return (time.perf_counter() - started) * 1000.0, response


def measure(pool, body: bytes, key: str) -> dict:
    auth = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    bare = {"Content-Type": "application/json"}
    for _ in range(5):
        opened = pool.num_connections
        e2e, real = timed_post(pool, body, auth)
        rtt, rejected = timed_post(pool, b"{}", bare)
        if pool.num_connections == opened and real.status == 200 and rejected.status in (401, 403):
            return {"e2e_ms": e2e, "rtt_ms": rtt, "data": json.loads(real.data)}
        time.sleep(1.0)
    raise RuntimeError(f"no clean pair: real {real.status}, rejected {rejected.status}")


def to_row(record: dict, allowed: tuple[str, ...], timings: list[dict], model: str) -> dict:
    """Same decision rule as run_eval.score_jev; the judgement comes from the first round."""
    data = timings[0]["data"]
    probs = {name: (a.get("noul") if isinstance(a, dict) else a) for name, a in data.get("answers", {}).items()}
    p_support = float(probs.get("supports") or 0.0)
    if "REFUTES" in allowed:
        p_refute = float(probs.get("refutes") or 0.0)
        p_nei = max(0.0, 1.0 - p_support - p_refute) if "NEI" in allowed else 0.0
    else:
        p_refute, p_nei = 1.0 - p_support, 0.0
    total = p_support + p_refute + p_nei or 1.0
    distribution = {label: value / total for label, value in
                    (("SUPPORTS", p_support), ("REFUTES", p_refute), ("NEI", p_nei)) if label in allowed}
    return {
        "id": record["id"], "dataset": record["dataset"], "split": record["split"], "system": "jev",
        "model": data.get("model") or model, "gold": record["label"],
        "prediction": max(distribution, key=distribution.get), "distribution": distribution,
        "raw": json.dumps(data, ensure_ascii=False)[:4000], "usage": data.get("usage", {}),
        "latency_ms": statistics.median(t["e2e_ms"] - t["rtt_ms"] for t in timings),
        "e2e_ms": statistics.median(t["e2e_ms"] for t in timings),
        "rtt_ms": statistics.median(t["rtt_ms"] for t in timings),
        "error": None, "finish_reason": None,
    }


def main() -> None:
    secrets = load_secrets()
    key, model = secrets["TYPESAFE_API_KEY"], secrets.get("JEV_MODEL") or "jev-latest"
    ids = [json.loads(line)["id"] for line in SAMPLE.open() if line.strip()]
    # the Qwen latency run drew its cases with run_eval --per-dataset 30 and the default seed
    records = {r["id"]: r for r in load_records(DATASETS, 30, 2026)}
    missing = [i for i in ids if i not in records]
    if missing:
        raise SystemExit(f"{len(missing)} sample ids not reproduced, e.g. {missing[:3]}")
    allowed = {d: allowed_labels(d) for d in DATASETS}

    pool = open_pool()
    bodies = {}
    for i in ids:
        r = records[i]
        state, questions = jev_payload(r["claim"], r["evidence"], MAX_CHARS, allowed[r["dataset"]])
        bodies[i] = json.dumps({"model": model, "state": state, "questions": questions}, ensure_ascii=False).encode()
    for i in ids[:WARMUP]:  # opens the connection; not recorded
        measure(pool, bodies[i], key)

    timings = collections.defaultdict(list)
    for round_ in range(1, ROUNDS + 1):  # rounds sweep the whole sample, so a case's repeats are spread in time
        for i in ids:
            timings[i].append(measure(pool, bodies[i], key))
        print(f"  round {round_}/{ROUNDS}", flush=True)
    with OUT.open("w") as handle:
        for i in ids:
            r = records[i]
            handle.write(json.dumps(to_row(r, allowed[r["dataset"]], timings[i], model), ensure_ascii=False) + "\n")

    rows = [json.loads(line) for line in OUT.open()]
    median = lambda key_: sorted(r[key_] for r in rows)[len(rows) // 2]
    print(f"wrote {OUT.name}: {len(rows)} cases, {pool.num_connections} connection(s) opened")
    print(f"median end-to-end {median('e2e_ms'):.0f} ms, round trip {median('rtt_ms'):.0f} ms, "
          f"service {median('latency_ms'):.0f} ms")
    print("by dataset:", dict(collections.Counter(r["dataset"] for r in rows)))


if __name__ == "__main__":
    main()
