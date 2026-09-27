#!/usr/bin/env python3
"""Assemble the revised prediction and latency files (the originals are only read).

Revised data = original eval-set rows, with
  - HoVer rows of qwen-direct-expanded and qwen-cot-expanded replaced by the
    label-safe few-shot runs (runs/lsf_hover.jsonl);
  - reasoning rows that stopped at the 900-token limit replaced by their 2048-token
    reruns (runs/tok2048.jsonl). --scope all replaces every finish_reason=length row;
    --scope nojudgment only those that gave no judgement; --scope none keeps the
    900-token rows, so truncations count as errors.
The latency sample (150 cases per Qwen setting) gets the same replacements from
runs/lat1.jsonl and runs/lat1_tok2048.jsonl. JEV and NLI are copied unchanged.

Writes <out>/predictions/<system>__<model>__eval.jsonl and <out>/latency/<system>__lat1.jsonl.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REV = ROOT / "revision_2026-09-27"
RUNS = REV / "runs"
PRED = ROOT / "results" / "predictions"
SENS = ROOT / "results" / "sensitivity"
QWEN = ["qwen-direct-compact", "qwen-direct-expanded", "qwen-cot-compact", "qwen-cot-expanded"]


def load(path: Path) -> dict[str, dict]:
    latest = {}
    for line in path.open():
        if line.strip():
            r = json.loads(line)
            latest[r["id"]] = r
    return latest


def load_runs(name: str) -> dict[tuple[str, str], dict]:
    out = {}
    path = RUNS / name
    if path.exists():
        for line in path.open():
            if line.strip():
                r = json.loads(line)
                if not r.get("error"):
                    out[(r["system"], r["id"])] = r
    return out


def truncated(row: dict, scope: str) -> bool:
    if "-cot-" not in row["system"] or row.get("finish_reason") != "length" or scope == "none":
        return False
    return scope == "all" or not row.get("prediction")


def revise(system: str, rows: dict[str, dict], lsf: dict, tok: dict, scope: str) -> tuple[dict, dict]:
    stats = {"lsf": 0, "tok2048": 0, "missing": 0}
    out = {}
    for rid, row in rows.items():
        if system.endswith("expanded") and row["dataset"] == "hover":
            if (system, rid) not in lsf:
                stats["missing"] += 1
            else:
                row = lsf[(system, rid)]
                stats["lsf"] += 1
        if truncated(row, scope):
            if (system, rid) not in tok:
                stats["missing"] += 1
            else:
                row = tok[(system, rid)]
                stats["tok2048"] += 1
        out[rid] = row
    return out, stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", choices=["all", "nojudgment", "none"], default="all")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    out = Path(args.out) if args.out else REV / f"data_{args.scope}"
    (out / "predictions").mkdir(parents=True, exist_ok=True)
    (out / "latency").mkdir(parents=True, exist_ok=True)

    lsf, tok = load_runs("lsf_hover.jsonl"), load_runs("tok2048.jsonl")
    lat_new, lat_tok = load_runs("lat1.jsonl"), load_runs("lat1_tok2048.jsonl")
    report = {}
    for path in sorted(PRED.glob("*__eval.jsonl")):
        system = path.name.split("__")[0]
        if system not in QWEN:
            shutil.copyfile(path, out / "predictions" / path.name)
            continue
        rows, stats = revise(system, load(path), lsf, tok, args.scope)
        (out / "predictions" / path.name).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows.values()))
        # latency sample: the lat1 run holds the HoVer few-shot rows and the 900-token truncations
        lat = load(next(SENS.glob(f"{system}__*__vllm-lat1.jsonl")))
        lat_lsf = {k: v for k, v in lat_new.items() if v["revision"]["phase"] == "lat-lsf"}
        lat_trunc = {**{k: v for k, v in lat_new.items() if v["revision"]["phase"] == "lat-tok2048"}, **lat_tok}
        lrows, lstats = revise(system, lat, lat_lsf, lat_trunc, args.scope)
        (out / "latency" / f"{system}__lat1.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in lrows.values()))
        report[system] = {"eval": stats, "latency": lstats}
    (out / "merge_report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report))


if __name__ == "__main__":
    main()
