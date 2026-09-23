"""Deduplicate prediction logs, keeping the most informative row per case id."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PREDICTIONS = ROOT / "results" / "predictions"

for path in sorted(PREDICTIONS.glob("*.jsonl")):
    best: dict[str, dict] = {}
    order: list[str] = []
    for line in path.open():
        if not line.strip():
            continue
        row = json.loads(line)
        key = row["id"]
        if key not in best:
            order.append(key)
            best[key] = row
            continue
        incumbent = best[key]
        if row.get("prediction") is not None:
            best[key] = row
        elif incumbent.get("prediction") is None:
            best[key] = row  # keep the most recent failed attempt
    with path.open("w") as handle:
        for key in order:
            handle.write(json.dumps(best[key], ensure_ascii=False) + "\n")
    print(f"{path.name}: {len(order)} unique rows")
