"""Paired system comparisons and trivial-calibration baselines from prediction logs.

McNemar's test with continuity correction on the paired sample (every system
sees the same 300 case ids). Protocol failures count as errors, which is the
accounting a deployment gate would apply.
"""
from __future__ import annotations

import collections
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PREDICTIONS = ROOT / "results" / "predictions"


def load() -> dict[str, dict[str, dict]]:
    by_system: dict[str, dict[str, dict]] = collections.defaultdict(dict)
    for path in sorted(PREDICTIONS.glob("*.jsonl")):
        for line in path.open():
            if not line.strip():
                continue
            row = json.loads(line)
            by_system[row["system"]][row["id"]] = row
    return by_system


def correct(row: dict) -> bool:
    return bool(row["prediction"]) and row["prediction"] == row["gold"]


def mcnemar(a: dict[str, dict], b: dict[str, dict]) -> tuple[int, int, float]:
    only_a = only_b = 0
    for case_id in a.keys() & b.keys():
        ca, cb = correct(a[case_id]), correct(b[case_id])
        if ca and not cb:
            only_a += 1
        elif cb and not ca:
            only_b += 1
    stat = (abs(only_a - only_b) - 1) ** 2 / (only_a + only_b) if only_a + only_b else 0.0
    p = math.erfc(math.sqrt(stat / 2)) if stat else 1.0
    return only_a, only_b, p


def uniform_brier(rows: list[dict]) -> float | None:
    """Brier score of a uniform predictor over the dataset's label space."""
    dist_rows = [r for r in rows if r.get("distribution")]
    if not dist_rows:
        return None
    labels = list(dist_rows[0]["distribution"])
    k = len(labels)
    return sum(sum((1.0 / k - (1.0 if r["gold"] == label else 0.0)) ** 2 for label in labels)
               for r in dist_rows) / len(dist_rows)


def main() -> None:
    systems = load()
    names = sorted(systems)
    shared = set.intersection(*(set(v) for v in systems.values()))
    print(f"shared ids: {len(shared)}; failures count as errors\n")

    n_pairs = len(names) * (len(names) - 1) // 2
    bonf = 0.05 / n_pairs
    print(f"McNemar, continuity-corrected (cells = cases correct only by that system); "
          f"Bonferroni alpha across {n_pairs} pairs = {bonf:.4f}:")
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            wa, wb, p = mcnemar(systems[a], systems[b])
            flag = " *" if p < bonf else ""
            print(f"  {a:>22s} vs {b:<22s}  {wa:3d} / {wb:<3d}  p={p:.4f}{flag}")

    print("\nUniform-predictor Brier baselines vs probability-emitting systems (per dataset):")
    for system in ("jev", "nli"):
        rows_by_ds: dict[str, list[dict]] = collections.defaultdict(list)
        for row in systems.get(system, {}).values():
            rows_by_ds[row["dataset"]].append(row)
        for ds, rows in sorted(rows_by_ds.items()):
            scored = [r for r in rows if r.get("distribution")]
            actual = sum(sum((r["distribution"][l] - (1.0 if r["gold"] == l else 0.0)) ** 2
                             for l in r["distribution"]) for r in scored) / max(len(scored), 1)
            print(f"  {system:4s} {ds:14s}  uniform={uniform_brier(rows):.3f}  actual={actual:.3f}")


if __name__ == "__main__":
    main()
