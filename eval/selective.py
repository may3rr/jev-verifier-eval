"""Selective verification: how much can a verifier decide on its own?

A verifier that returns a label distribution lets the service auto-decide the
cases it is most confident about and send the rest to human review. For every
system with distributions this script ranks cases by top-label probability and
reports:

- accuracy of the auto-decided cases at fixed coverage (with bootstrap CIs),
- AURC, the area under the risk-coverage curve (lower is better),
- the largest auto-decision rate that keeps accuracy at or above a target,
  i.e. the human-review share the service must staff for that target.

Protocol failures have no distribution; they rank last (always sent to review)
and count as errors only once coverage reaches them.

Writes results/selective.json and results/selective.md; the figures come from
make_figs.py.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PRED = ROOT / "results" / "predictions"
OUT = ROOT / "results"

COVERAGES = (0.3, 0.5, 0.7, 0.9, 1.0)
TARGETS = (0.90, 0.95)
MIN_ACCEPTED = 30  # do not credit a target reached on a handful of cases
DATASETS = ("fever", "scifact", "hover", "vitaminc", "climate_fever")

DISPLAY = {
    "jev": "JEV",
    "nli": "NLI-DeBERTa",
    "qwen-direct-compact": "Qwen零样本直答",
    "qwen-direct-expanded": "Qwen少样本直答",
    "qwen-cot-compact": "Qwen零样本推理",
    "qwen-cot-expanded": "Qwen少样本推理",
}


def load(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.open() if line.strip()]
    # keep the last attempt per id (resumed runs append)
    latest = {row["id"]: row for row in rows}
    return list(latest.values())


def ranked_correct(rows: list[dict]) -> list[int]:
    """1/0 correctness, ordered from most to least confident."""
    def confidence(row: dict) -> float:
        dist = row.get("distribution")
        return max(dist.values()) if row.get("prediction") and dist else -1.0
    ordered = sorted(rows, key=lambda r: (-confidence(r), r["id"]))
    return [int(bool(r.get("prediction")) and r["prediction"] == r["gold"]) for r in ordered]


def acc_at(correct: list[int], coverage: float) -> float:
    k = max(1, round(len(correct) * coverage))
    return sum(correct[:k]) / k


def aurc(correct: list[int]) -> float:
    hits, risks = 0, []
    for k, c in enumerate(correct, 1):
        hits += c
        risks.append(1 - hits / k)
    return sum(risks) / len(risks)


def max_coverage(correct: list[int], target: float) -> float:
    hits, best = 0, 0.0
    for k, c in enumerate(correct, 1):
        hits += c
        if k >= MIN_ACCEPTED and hits / k >= target:
            best = k / len(correct)
    return best


def bootstrap(rows: list[dict], stat, n: int = 1000, seed: int = 2026) -> tuple[float, float]:
    rng = random.Random(seed)
    values = sorted(stat(ranked_correct([rng.choice(rows) for _ in rows])) for _ in range(n))
    return values[int(0.025 * n)], values[int(0.975 * n) - 1]


def summarise(rows: list[dict], with_ci: bool) -> dict:
    correct = ranked_correct(rows)
    out: dict = {"n": len(rows), "aurc": aurc(correct), "acc": {}, "auto_rate": {}}
    for cov in COVERAGES:
        out["acc"][f"{cov:.1f}"] = acc_at(correct, cov)
    for target in TARGETS:
        out["auto_rate"][f"{target:.2f}"] = max_coverage(correct, target)
    if with_ci:
        out["acc_ci"] = {f"{cov:.1f}": bootstrap(rows, lambda c, cov=cov: acc_at(c, cov)) for cov in (0.5,)}
        out["aurc_ci"] = bootstrap(rows, aurc)
        out["auto_rate_ci"] = {f"{t:.2f}": bootstrap(rows, lambda c, t=t: max_coverage(c, t)) for t in TARGETS}
    out["curve"] = [acc_at(correct, k / 100) for k in range(5, 101)]
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tags", nargs="+", default=["full"],
                        help="prediction file tags to include, e.g. full vllm")
    args = parser.parse_args()

    results: dict = {}
    for tag in args.tags:
        for path in sorted(PRED.glob(f"*__{tag}.jsonl")):
            rows = load(path)
            if not any(r.get("distribution") for r in rows):
                print(f"skip {path.name}: no distributions")
                continue
            system = path.name.split("__")[0]
            name = DISPLAY.get(system, system)
            if len(args.tags) > 1 and name in results:
                name = f"{name}[{tag}]"
            results[name] = {
                "file": path.name,
                "pooled": summarise(rows, with_ci=True),
                "by_dataset": {d: summarise([r for r in rows if r["dataset"] == d], with_ci=False)
                               for d in DATASETS if any(r["dataset"] == d for r in rows)},
            }

    (OUT / "selective.json").write_text(json.dumps(results, ensure_ascii=False, indent=1))

    lines = ["| 系统 | 30% | 50% [95%CI] | 70% | 90% | 100% | AURC [95%CI] | 准确率≥0.90时自动判定比例 | ≥0.95 |",
             "|---|---|---|---|---|---|---|---|---|"]
    for name, res in results.items():
        p = res["pooled"]
        lo, hi = p["acc_ci"]["0.5"]
        alo, ahi = p["aurc_ci"]
        lines.append(
            f"| {name} | {p['acc']['0.3']:.3f} | {p['acc']['0.5']:.3f} [{lo:.3f}, {hi:.3f}] | {p['acc']['0.7']:.3f} | "
            f"{p['acc']['0.9']:.3f} | {p['acc']['1.0']:.3f} | {p['aurc']:.3f} [{alo:.3f}, {ahi:.3f}] | "
            f"{p['auto_rate']['0.90']:.1%} | {p['auto_rate']['0.95']:.1%} |")
    lines += ["", "按数据集：50%覆盖率准确率 / AURC / 准确率≥0.90时自动判定比例", "",
              "| 系统 | " + " | ".join(DATASETS) + " |", "|---|" + "---|" * len(DATASETS)]
    for name, res in results.items():
        cells = []
        for d in DATASETS:
            s = res["by_dataset"].get(d)
            cells.append(f"{s['acc']['0.5']:.3f} / {s['aurc']:.3f} / {s['auto_rate']['0.90']:.0%}" if s else "—")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    (OUT / "selective.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
