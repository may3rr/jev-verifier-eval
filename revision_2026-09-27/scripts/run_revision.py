#!/usr/bin/env python3
"""Revision runs on the self-hosted vLLM server (2026-09-27).

Phases, all Qwen3.5-9B, temperature 0, native thinking off, top-20 logprobs,
evidence capped at 6000 characters, exactly as eval/run_eval.py:

  lsf       HoVer (all 1,846) with the few-shot block filtered to the legal labels,
            for qwen-direct-expanded (1 token) and qwen-cot-expanded (900 tokens)
  tok2048   every reasoning row that stopped at the 900-token limit, rerun at 2048:
            qwen-cot-compact and the non-HoVer qwen-cot-expanded rows from the eval
            set, plus the truncated rows of the lsf HoVer run above
  lat       the shared 150-case latency sample, one request at a time, for the rows
            the revision replaces (HoVer few-shot rows, truncated rows), plus a
            30-case re-timing of unchanged qwen-direct-compact rows to compare this
            machine with the one that timed the original sample

Rows keep the run_eval.py schema, so eval/paper_numbers.py can read them.
Usage: python3 run_revision.py jobs.json outdir
"""
from __future__ import annotations

import concurrent.futures as cf
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "eval"))

from prompts import (RULES, _definition_block, _digits, _fewshot, _task_block, cot_messages, direct_messages,
                     parse_cot, parse_direct, render_evidence)
from providers import QwenClient
from run_eval import decision_distribution

MODEL = "Qwen/Qwen3.5-9B"
MAX_CHARS = 6000
TOP = 20


def direct_expanded_lsf(claim: str, evidence: list[str], allowed: tuple[str, ...]) -> list[dict]:
    """prompts.direct_messages(expanded=True) with the few-shot block filtered like cot_messages(label_safe=True)."""
    user = (
        "You are a claim verification service. Decide whether the evidence entails the claim.\n"
        f"{_definition_block(allowed)}\n\n{RULES}\n\n{_fewshot(allowed)}"
        f"{_task_block(claim, render_evidence(evidence, MAX_CHARS))}\n\n"
        f"Reply with exactly one character: {_digits(allowed)}."
    )
    return [{"role": "user", "content": user}]


def messages(job: dict) -> list[dict]:
    rec, allowed = job["record"], tuple(job["allowed"])
    expanded = job["system"].endswith("expanded")
    if "-cot-" in job["system"]:
        return cot_messages(rec["claim"], rec["evidence"], expanded, allowed, MAX_CHARS, label_safe=job["label_safe"])
    if job["label_safe"]:
        assert expanded
        return direct_expanded_lsf(rec["claim"], rec["evidence"], allowed)
    return direct_messages(rec["claim"], rec["evidence"], expanded, allowed, MAX_CHARS)


def run_one(client: QwenClient, job: dict) -> dict:
    rec, allowed = job["record"], tuple(job["allowed"])
    cot = "-cot-" in job["system"]
    started = time.time()
    try:
        res = client.chat(MODEL, messages(job), max_tokens=job["max_tokens"], enable_thinking=False, top_logprobs=TOP)
        text = res["texts"][0] if res["texts"] else ""
        pred = (parse_cot if cot else parse_direct)(text)
        if pred not in allowed:
            pred = None
        dist = decision_distribution((res.get("logprobs") or [[]])[0], cot, allowed) if pred else None
        payload = {"prediction": pred, "distribution": dist, "raw": text[:4000], "usage": res["usage"],
                   "latency_ms": res["latency_ms"], "finish_reason": (res.get("finish_reasons") or [None])[0]}
        error = None
    except Exception as exc:  # noqa: BLE001 - recorded per row
        payload = {"prediction": None, "distribution": None, "raw": "", "usage": {},
                   "latency_ms": (time.time() - started) * 1000, "finish_reason": None}
        error = f"{type(exc).__name__}: {exc}"[:400]
    return {"id": rec["id"], "dataset": rec["dataset"], "split": rec["split"], "system": job["system"],
            "model": MODEL, "gold": rec["label"], **payload, "error": error,
            "claim": rec["claim"], "n_evidence": len(rec["evidence"]),
            "evidence_chars": sum(len(e) for e in rec["evidence"]),
            "revision": {"phase": job["phase"], "max_tokens": job["max_tokens"], "label_safe": job["label_safe"]}}


def run(jobs: list[dict], out: Path, concurrency: int, timeout: int) -> list[dict]:
    done = {}
    if out.exists():
        for line in out.open():
            if line.strip():
                r = json.loads(line)
                if not r.get("error"):
                    done[(r["system"], r["id"])] = r
    todo = [j for j in jobs if (j["system"], j["record"]["id"]) not in done]
    print(f"{out.name}: {len(todo)} to run, {len(done)} cached", flush=True)
    client = QwenClient({"SILICONFLOW_API_KEY": "x"}, timeout=timeout)
    t0 = time.time()
    with out.open("a") as fh, cf.ThreadPoolExecutor(concurrency) as pool:
        for n, row in enumerate(pool.map(lambda j: run_one(client, j), todo), 1):
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
            done[(row["system"], row["id"])] = row
            if n % 100 == 0:
                print(f"  {n}/{len(todo)} {time.time() - t0:.0f}s", flush=True)
    print(f"{out.name}: finished in {time.time() - t0:.0f}s", flush=True)
    return [done[(j["system"], j["record"]["id"])] for j in jobs]


def main() -> None:
    jobs = json.load(open(sys.argv[1]))
    outdir = Path(sys.argv[2])
    outdir.mkdir(parents=True, exist_ok=True)

    # 1. label-safe HoVer runs, batched as in the main run
    lsf_rows = run(jobs["lsf"], outdir / "lsf_hover.jsonl", 64, 600)

    # 2. every 900-token truncation rerun at 2048, including those of step 1
    trunc = list(jobs["tok2048"])
    for job, row in zip(jobs["lsf"], lsf_rows):
        if "-cot-" in job["system"] and row.get("finish_reason") == "length":
            trunc.append({**job, "phase": "tok2048", "max_tokens": 2048})
    json.dump(trunc, open(outdir / "tok2048_jobs.json", "w"))
    run(trunc, outdir / "tok2048.jsonl", 32, 900)

    # 3. latency sample, one request at a time; a 900-token truncation is retried at 2048
    lat_rows = run(jobs["lat"], outdir / "lat1.jsonl", 1, 900)
    retry = [{**j, "phase": "lat-tok2048", "max_tokens": 2048} for j, r in zip(jobs["lat"], lat_rows)
             if "-cot-" in j["system"] and r.get("finish_reason") == "length" and j["max_tokens"] == 900]
    run(retry, outdir / "lat1_tok2048.jsonl", 1, 900)
    print("REVISION_DONE", flush=True)


if __name__ == "__main__":
    main()
