#!/usr/bin/env python3
"""Cut the paper's evaluation set out of the full-split runs.

The evaluation set keeps every labelled evaluation case of SciFact, HoVer and
Climate-FEVER and a seeded random 3,000 of FEVER and VitaminC (run_eval.py
--full --cap 3000). JEV and NLI were run on the full splits; this script writes
<system>__<model>__eval.jsonl files restricted to the evaluation-set ids, so
every system is compared on the same cases. It also reports, for JEV and NLI,
the sample estimate next to the full-split value.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_eval import load_records

ROOT = Path(__file__).resolve().parent.parent
PRED = ROOT / "results" / "predictions"
DATASETS = ["fever", "scifact", "hover", "vitaminc", "climate_fever"]
CAP = 3000
SEED = 2026


def main() -> None:
    ids = {r["id"] for r in load_records(DATASETS, 0, SEED, full=True, cap=CAP)}
    full_ids = {r["id"] for r in load_records(DATASETS, 0, SEED, full=True)}
    report = {}
    for path in sorted(PRED.glob("*__fullset.jsonl")):
        latest = {}
        for line in path.open():
            if line.strip():
                row = json.loads(line)
                # keep a row with a prediction over a failed retry of the same id
                if row["id"] not in latest or row.get("prediction") or not latest[row["id"]].get("prediction"):
                    latest[row["id"]] = row
        rows = [latest[i] for i in sorted(ids) if i in latest]
        missing = len(ids) - len(rows)
        out = path.with_name(path.name.replace("__fullset.jsonl", "__eval.jsonl"))
        out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
        print(f"{out.name}: {len(rows)} rows, {missing} missing")
        if full_ids <= latest.keys():
            system = path.name.split("__")[0]
            report[system] = {}
            for d in DATASETS:
                all_rows = [r for i, r in latest.items() if i in full_ids and r["dataset"] == d]
                sample = [r for r in rows if r["dataset"] == d]
                acc = lambda rs: sum(bool(r.get("prediction")) and r["prediction"] == r["gold"] for r in rs) / len(rs)
                report[system][d] = {"n_full": len(all_rows), "acc_full": acc(all_rows),
                                     "n_sample": len(sample), "acc_sample": acc(sample)}
    (ROOT / "results" / "evalset_vs_full.json").write_text(json.dumps(report, indent=1))
    for system, by_ds in report.items():
        for d, v in by_ds.items():
            print(f"  {system:4s} {d:14s} sample {v['acc_sample']:.3f} (n={v['n_sample']})"
                  f"  full {v['acc_full']:.3f} (n={v['n_full']})")


if __name__ == "__main__":
    main()
