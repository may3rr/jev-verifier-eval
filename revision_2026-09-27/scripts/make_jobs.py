#!/usr/bin/env python3
"""Build the job list for run_revision.py from the original prediction logs (read only)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "eval"))
from run_eval import allowed_labels  # noqa: E402

PRED = ROOT / "results" / "predictions"
SENS = ROOT / "results" / "sensitivity"
OUT = Path(__file__).resolve().parent.parent / "runs" / "jobs.json"
DATASETS = ["fever", "scifact", "hover", "vitaminc", "climate_fever"]


def load(path: Path) -> dict[str, dict]:
    latest = {}
    for line in path.open():
        if line.strip():
            r = json.loads(line)
            latest[r["id"]] = r
    return latest


def main() -> None:
    records = {}
    for d in DATASETS:
        for line in (ROOT / "data" / "unified" / f"{d}.jsonl").open():
            if line.strip():
                r = json.loads(line)
                records[r["id"]] = r
    allowed = {d: list(allowed_labels(d)) for d in DATASETS}

    def job(system, rid, phase, max_tokens, label_safe):
        rec = records[rid]
        return {"system": system, "phase": phase, "max_tokens": max_tokens, "label_safe": label_safe,
                "allowed": allowed[rec["dataset"]], "record": rec}

    ev = {s: load(PRED / f"{s}__qwen-qwen3-5-9b__eval.jsonl") for s in ("qwen-cot-compact", "qwen-cot-expanded")}
    hover = sorted(i for i, r in ev["qwen-cot-expanded"].items() if r["dataset"] == "hover")
    lsf = [job(s, i, "lsf", 1 if "direct" in s else 900, True)
           for s in ("qwen-direct-expanded", "qwen-cot-expanded") for i in hover]
    tok = [job(s, i, "tok2048", 2048, False) for s, rows in ev.items() for i, r in sorted(rows.items())
           if r.get("finish_reason") == "length" and not (s == "qwen-cot-expanded" and r["dataset"] == "hover")]

    lat = []
    for s in ("qwen-direct-compact", "qwen-direct-expanded", "qwen-cot-compact", "qwen-cot-expanded"):
        rows = load(SENS / f"{s}__qwen-qwen3-5-9b__vllm-lat1.jsonl")
        for i, r in sorted(rows.items()):
            few_hover = s.endswith("expanded") and r["dataset"] == "hover"
            if few_hover:
                lat.append(job(s, i, "lat-lsf", 1 if "direct" in s else 900, True))
            elif "-cot-" in s and r.get("finish_reason") == "length":
                lat.append(job(s, i, "lat-tok2048", 2048, False))
            elif s == "qwen-direct-compact" and r["dataset"] == "fever":
                lat.append(job(s, i, "lat-check", 1, False))  # unchanged rows, re-timed to compare machines
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"lsf": lsf, "tok2048": tok, "lat": lat}, ensure_ascii=False))
    from collections import Counter
    for k, v in (("lsf", lsf), ("tok2048", tok), ("lat", lat)):
        print(k, len(v), dict(Counter((j["system"], j["phase"]) for j in v)))


if __name__ == "__main__":
    main()
