"""Scoring, calibration, and failure-feature analysis for prediction logs."""
from __future__ import annotations

import argparse
import collections
import json
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PREDICTIONS = ROOT / "results" / "predictions"
RESULTS = ROOT / "results"
LABELS = ["SUPPORTS", "REFUTES", "NEI"]

NEGATION = re.compile(r"\b(no|not|never|without|cannot|can't|doesn't|does not|isn't|is not|nor|neither)\b", re.I)
SCOPE = re.compile(r"\b(all|every|always|any|none|entire|national|nationwide|worldwide|most|only)\b", re.I)
MODALITY = re.compile(r"\b(may|might|could|can|must|should|likely|possibly|probably|generally|sometimes)\b", re.I)
NUMERIC = re.compile(r"\b\d+(?:[.,]\d+)?%?\b")
CAUSAL = re.compile(r"\b(cause[sd]?|because|due to|leads? to|results? in|therefore)\b", re.I)


def macro_f1(y_true: list[str], y_pred: list[str], labels: list[str] = LABELS) -> float:
    scores = []
    for label in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)
        if tp == 0 and (fp or fn):
            scores.append(0.0)
        elif tp + fp + fn == 0:
            continue
        else:
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return sum(scores) / len(scores) if scores else 0.0


def per_class(y_true: list[str], y_pred: list[str], label: str) -> dict:
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "support": tp + fn, "predicted": tp + fp}


def brier(rows: list[dict]) -> float | None:
    """Standard multiclass Brier score: the squared error summed over the
    dataset's own label space, averaged over rows. Range [0, 2]; a uniform
    predictor scores 0.5 on a binary space and 2/3 on a three-way space."""
    scored = [r for r in rows if r.get("distribution") and r.get("prediction")]
    if not scored:
        return None
    total = 0.0
    for row in scored:
        dist = row["distribution"]
        total += sum((dist[label] - (1.0 if row["gold"] == label else 0.0)) ** 2 for label in dist)
    return total / len(scored)


def ece(rows: list[dict], bins: int = 10) -> float | None:
    """Equal-frequency expected calibration error of the top-class probability."""
    scored = [r for r in rows if r.get("distribution") and r.get("prediction")]
    if len(scored) < bins:
        return None
    confidence = [max(r["distribution"].values()) for r in scored]
    correct = [1.0 if r["prediction"] == r["gold"] else 0.0 for r in scored]
    order = sorted(range(len(scored)), key=lambda i: confidence[i])
    total = 0.0
    for chunk in range(bins):
        start = chunk * len(order) // bins
        stop = (chunk + 1) * len(order) // bins
        if stop <= start:
            continue
        idx = order[start:stop]
        gap = abs(sum(confidence[i] for i in idx) / len(idx) - sum(correct[i] for i in idx) / len(idx))
        total += gap * len(idx) / len(order)
    return total


def features(row: dict) -> dict:
    claim = row.get("claim", "")
    evidence_chars = row.get("evidence_chars") or 0
    numbers_claim = set(NUMERIC.findall(claim))
    return {
        "long_evidence": evidence_chars > 2500,
        "many_records": (row.get("n_evidence") or 0) > 3,
        "negation_in_claim": bool(NEGATION.search(claim)),
        "scope_word_in_claim": bool(SCOPE.search(claim)),
        "modality_in_claim": bool(MODALITY.search(claim)),
        "numeric_in_claim": bool(numbers_claim),
        "causal_in_claim": bool(CAUSAL.search(claim)),
        "claim_len_gt_20": len(claim.split()) > 20,
    }


def wilson(low_success: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total == 0:
        return (0.0, 1.0)
    phat = low_success / total
    denom = 1 + z * z / total
    centre = phat + z * z / (2 * total)
    margin = z * math.sqrt(phat * (1 - phat) / total + z * z / (4 * total * total))
    return ((centre - margin) / denom, (centre + margin) / denom)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pattern", default="*.jsonl")
    parser.add_argument("--tag", default=None)
    args = parser.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    rows_by_run: dict[tuple, list[dict]] = collections.defaultdict(list)
    for path in sorted(PREDICTIONS.glob(args.pattern)):
        if args.tag and args.tag not in path.name:
            continue
        for line in path.open():
            if not line.strip():
                continue
            row = json.loads(line)
            rows_by_run[(row["system"], row["model"])].append(row)

    summary = []
    failure_rows = []
    per_class_rows = []
    for (system, model), rows in sorted(rows_by_run.items()):
        for dataset in sorted({r["dataset"] for r in rows}):
            subset = [r for r in rows if r["dataset"] == dataset]
            scored = [r for r in subset if r["prediction"] and not r.get("error")]
            y_true = [r["gold"] for r in scored]
            y_pred = [r["prediction"] for r in scored]
            accuracy = sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(scored) if scored else 0.0
            lo, hi = wilson(sum(1 for t, p in zip(y_true, y_pred) if t == p), len(scored))
            summary.append(
                {
                    "system": system,
                    "model": model,
                    "dataset": dataset,
                    "n_planned": len(subset),
                    "n_scored": len(scored),
                    "parse_failures": len(subset) - len(scored),
                    "accuracy": accuracy,
                    "accuracy_ci_low": lo,
                    "accuracy_ci_high": hi,
                    "macro_f1": macro_f1(y_true, y_pred),
                    "brier": brier(scored),
                    "ece": ece(scored),
                    "mean_latency_ms": sum(r["latency_ms"] or 0 for r in subset) / max(len(subset), 1),
                    "p95_latency_ms": sorted(r["latency_ms"] or 0 for r in subset)[int(0.95 * (len(subset) - 1))] if subset else 0.0,
                    "mean_output_tokens": sum((r.get("usage") or {}).get("output_tokens") or (r.get("usage") or {}).get("completion_tokens") or 0 for r in subset) / max(len(subset), 1),
                }
            )
            for label in LABELS:
                stats = per_class(y_true, y_pred, label)
                per_class_rows.append({"system": system, "model": model, "dataset": dataset, "label": label, **stats})
            for name in list(features(scored[0]).keys()) if scored else []:
                for value in (False, True):
                    bucket = [r for r in scored if features(r)[name] is value]
                    bad = sum(1 for r in bucket if r["prediction"] != r["gold"])
                    if len(bucket) >= 10:
                        failure_rows.append(
                            {
                                "system": system,
                                "model": model,
                                "dataset": dataset,
                                "feature": name,
                                "value": value,
                                "n": len(bucket),
                                "error_rate": bad / len(bucket),
                            }
                        )

    def dump(name: str, records: list[dict]) -> None:
        if not records:
            return
        path = RESULTS / name
        keys = list(records[0].keys())
        with path.open("w") as handle:
            handle.write(",".join(keys) + "\n")
            for record in records:
                handle.write(",".join("" if record[k] is None else str(record[k]) for k in keys) + "\n")
        print(f"wrote {path} ({len(records)} rows)")

    dump("summary.csv", summary)
    dump("per_class.csv", per_class_rows)
    dump("failure_features.csv", failure_rows)
    for row in summary:
        print(
            f"{row['system']:22s} {row['dataset']:14s} n={row['n_scored']:4d} "
            f"acc={row['accuracy']:.3f} macroF1={row['macro_f1']:.3f} "
            f"brier={'n/a' if row['brier'] is None else round(row['brier'], 3)}"
        )


if __name__ == "__main__":
    main()
