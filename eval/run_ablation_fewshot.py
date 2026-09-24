#!/usr/bin/env python3
"""Ablation: expanded-CoT on HoVer with a label-safe few-shot block.

The shipped expanded prompt embeds a worked example ending in ``Answer: 0``
unconditionally, so on binary HoVer the prompt itself demonstrates an
out-of-space answer. This arm re-runs the same 300 HoVer cases with the
few-shot filtered to admissible labels (``label_safe=True``) to separate
few-shot label leakage from reasoning-induced abstention.

Output goes to results/sensitivity/ so it never enters the main metrics glob.
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from providers import QwenClient  # noqa: E402
from prompts import cot_messages, parse_cot  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
UNIFIED = ROOT / "data" / "unified" / "hover.jsonl"
ALLOWED = ("SUPPORTS", "REFUTES")
MODEL = "Qwen/Qwen3.5-9B"
MAX_TOKENS = 900
MAX_CHARS = 6000


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="full", help="run whose HoVer cases to reuse, e.g. full, vllm, fullset")
    parser.add_argument("--concurrency", type=int, default=8)
    args = parser.parse_args()
    OUT = ROOT / "results" / "sensitivity" / f"qwen-cot-expanded-lsf__qwen-qwen3-5-9b__{args.tag}.jsonl"
    MAIN = ROOT / "results" / "predictions" / f"qwen-cot-expanded__qwen-qwen3-5-9b__{args.tag}.jsonl"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    # same frozen HoVer cases as the main run
    done_ids = {json.loads(l)["id"] for l in open(MAIN) if l.strip()}
    records = {json.loads(l)["id"]: json.loads(l)
               for l in open(UNIFIED) if l.strip()}
    ids = sorted(i for i in done_ids if i in records)  # hover ids only
    have = {json.loads(l)["id"] for l in open(OUT) if l.strip()} if OUT.exists() else set()
    todo = [records[i] for i in ids if i not in have]
    print(f"hover cases={len(ids)} done={len(have)} todo={len(todo)}", flush=True)

    client = QwenClient(timeout=120)
    completed = 0
    with OUT.open("a") as fh:
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futs = {}
            for rec in todo:
                msgs = cot_messages(rec["claim"], rec["evidence"], True, ALLOWED,
                                    MAX_CHARS, label_safe=True)
                futs[pool.submit(client.chat, MODEL, msgs, MAX_TOKENS, 0.0, False)] = rec
            for fut in as_completed(futs):
                rec = futs[fut]
                out = fut.result()
                text = out["texts"][0] if out.get("texts") else ""
                pred = parse_cot(text)
                if pred not in ALLOWED:
                    pred = None
                row = {"id": rec["id"], "dataset": rec["dataset"], "gold": rec["label"],
                       "system": "qwen-cot-expanded-lsf", "model": "qwen-qwen3-5-9b",
                       "prediction": pred,
                       "raw": text[:4000],
                       "finish_reason": (out.get("finish_reasons") or [None])[0],
                       "usage": out.get("usage"), "latency_ms": out.get("latency_ms"),
                       "error": None if text else "empty",
                       "tag": "ablfewshot"}
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                completed += 1
                if completed % 25 == 0:
                    print(f"{completed}/{len(todo)}", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
