#!/usr/bin/env python3
"""Every number the Chinese manuscript reports, computed from the prediction logs.

Source: the `eval` files written by make_evalset.py, the same 9,681 cases for
every system (SciFact, HoVer and Climate-FEVER in full, a seeded random 3,000 of
FEVER and VitaminC). Qwen ran on the self-hosted vLLM server with token
probabilities. Pass --tag full-sample/vllm to reproduce the earlier 1,464-case tables. Protocol failures count as errors
everywhere: in accuracy, in precision/recall/F1 (as a prediction outside every
label) and in the paired tests. Pooled columns are computed over every case of the set.

Writes results/paper_numbers.json, which zh/build_docx.py reads for its tables.
"""
from __future__ import annotations

import json
import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from metrics import brier, ece
from selective import summarise

ROOT = Path(__file__).resolve().parent.parent
PRED = ROOT / "results" / "predictions"
SENS = ROOT / "results" / "sensitivity"
OUT = ROOT / "results" / "paper_numbers.json"

SYSTEMS = ["jev", "qwen-direct-compact", "qwen-direct-expanded",
           "qwen-cot-compact", "qwen-cot-expanded", "nli"]
TAG = "eval"
LABEL = {
    "jev": "JEV",
    "nli": "NLI-DeBERTa",
    "qwen-direct-compact": "Qwen零样本直答",
    "qwen-direct-expanded": "Qwen少样本直答",
    "qwen-cot-compact": "Qwen零样本推理",
    "qwen-cot-expanded": "Qwen少样本推理",
}
DATASETS = ["fever", "scifact", "hover", "vitaminc", "climate_fever"]
SPACE = {d: ["SUPPORTS", "REFUTES"] if d == "hover" else ["SUPPORTS", "REFUTES", "NEI"] for d in DATASETS}
JEV_IN, QIN, QOUT = 0.042e-6, 0.10e-6, 0.15e-6  # $/token list prices
GPU_USD_PER_HOUR = 2.80 / 7.1                   # A100 40GB rental, CNY 2.80/h
BATCH_SECONDS = {"qwen-direct-compact": 56, "qwen-direct-expanded": 82,
                 "qwen-cot-compact": 183, "qwen-cot-expanded": 315}  # results/run_vllm.log
BATCH_CASES = 1464  # the timed batch run above


def load(path: Path) -> list[dict]:
    latest = {}
    for line in path.open():
        if line.strip():
            row = json.loads(line)
            latest[row["id"]] = row
    return list(latest.values())


def load_system(system: str) -> list[dict]:
    return load(next(PRED.glob(f"{system}__*__{TAG}.jsonl")))


def correct(row: dict) -> bool:
    return bool(row.get("prediction")) and row["prediction"] == row["gold"]


def prf(rows: list[dict], labels: list[str]) -> tuple[float, float, float]:
    """Accuracy, macro precision and macro F1; a failure is a prediction of no label."""
    acc = sum(correct(r) for r in rows) / len(rows)
    ps, fs = [], []
    for label in labels:
        tp = sum(1 for r in rows if r["gold"] == label and r.get("prediction") == label)
        fp = sum(1 for r in rows if r["gold"] != label and r.get("prediction") == label)
        fn = sum(1 for r in rows if r["gold"] == label and r.get("prediction") != label)
        if tp + fn == 0:
            continue
        p = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn)
        ps.append(p)
        fs.append(2 * p * rec / (p + rec) if p + rec else 0.0)
    return acc, sum(ps) / len(ps), sum(fs) / len(fs)


def mcnemar(a: list[dict], b: list[dict]) -> tuple[int, int, float]:
    bi = {r["id"]: r for r in b}
    only_a = only_b = 0
    for r in a:
        ca, cb = correct(r), correct(bi[r["id"]])
        only_a += ca and not cb
        only_b += cb and not ca
    stat = (abs(only_a - only_b) - 1) ** 2 / (only_a + only_b) if only_a + only_b else 0.0
    return only_a, only_b, math.erfc(math.sqrt(stat / 2)) if stat else 1.0


def uniform_brier(rows: list[dict], dataset: str) -> float:
    k = len(SPACE[dataset])
    return (1 - 1 / k) ** 2 + (k - 1) / k ** 2


def pct(x: float) -> str:
    return f"{100 * x:.1f}"


def f3(x: float | None) -> str:
    return "—" if x is None else f"{x:.3f}"


def main() -> None:
    data = {s: load_system(s) for s in SYSTEMS}
    ids = {s: {r["id"] for r in rows} for s, rows in data.items()}
    assert all(v == ids["jev"] for v in ids.values()), "systems must share the same case ids"
    by_ds = {s: {d: [r for r in rows if r["dataset"] == d] for d in DATASETS} for s, rows in data.items()}
    out: dict = {"systems": {s: LABEL[s] for s in SYSTEMS}, "n": len(data["jev"])}

    # main table: Acc / macro-P / macro-F1 per dataset and pooled
    metrics, main = {}, [["系统"] + [c for d in DATASETS + ["pooled"] for c in ("Acc", "P", "F1")]]
    for s in SYSTEMS:
        m = {d: prf(by_ds[s][d], SPACE[d]) for d in DATASETS}
        m["pooled"] = prf(data[s], ["SUPPORTS", "REFUTES", "NEI"])
        metrics[s] = m
        main.append([LABEL[s]] + [pct(v) for d in DATASETS + ["pooled"] for v in m[d]])
    out["metrics"] = metrics
    out["table_main"] = main

    # calibration: Brier per dataset, pooled Brier and ECE, share of saturated confidence
    calib = [["系统", "FEVER", "SciFact", "HoVer", "VitaminC", "Climate-\nFEVER", "合并\nBrier", "合并\nECE", "置信度\n≥0.99占比"]]
    calib.append(["均匀预测"] + [f3(uniform_brier([], d)) for d in DATASETS] + ["—", "—", "—"])
    cal = {}
    for s in SYSTEMS:
        with_dist = [r for r in data[s] if r.get("distribution") and r.get("prediction")]
        sat = sum(max(r["distribution"].values()) >= 0.99 for r in with_dist) / len(with_dist)
        cal[s] = {"brier": {d: brier(by_ds[s][d]) for d in DATASETS}, "brier_pooled": brier(data[s]),
                  "ece": ece(data[s]), "saturated": sat}
        calib.append([LABEL[s]] + [f3(cal[s]["brier"][d]) for d in DATASETS]
                     + [f3(cal[s]["brier_pooled"]), f3(cal[s]["ece"]), f"{sat:.1%}"])
    out["calibration"] = cal
    out["table_calibration"] = calib

    # selective verification
    sel, table = {}, [["系统", "自动判定\n30%", "自动判定\n50%", "自动判定\n70%", "全部\n自动判定", "AURC",
                       "准确率≥0.90\n自动判定比例", "准确率≥0.95\n自动判定比例"]]
    for s in SYSTEMS:
        res = summarise(data[s], with_ci=True)
        res["by_dataset"] = {d: summarise(by_ds[s][d], with_ci=False) for d in DATASETS}
        res.pop("curve")
        for v in res["by_dataset"].values():
            v.pop("curve")
        sel[s] = res
        table.append([LABEL[s], f3(res["acc"]["0.3"]), f3(res["acc"]["0.5"]), f3(res["acc"]["0.7"]),
                      f3(res["acc"]["1.0"]), f3(res["aurc"]),
                      f"{res['auto_rate']['0.90']:.1%}", f"{res['auto_rate']['0.95']:.1%}"])
    out["selective"] = sel
    out["table_selective"] = table

    # efficiency: Qwen latency from the concurrency-1 sample, JEV and NLI from the main run
    eff, table = {}, [["系统", "中位延迟\n(ms)", "p95延迟\n(ms)", "平均输出\n词元", "协议失败", "每千次成本\n(托管价)", "每千次成本\n(自建)"]]
    for s in SYSTEMS:
        lat_rows = load(next(SENS.glob(f"{s}__*__vllm-lat1.jsonl"))) if s.startswith("qwen") else data[s]
        lat = sorted(r["latency_ms"] for r in lat_rows if r.get("latency_ms") and not r.get("error"))
        usage = [r.get("usage") or {} for r in data[s]]
        usage = [json.loads(u.replace("'", '"')) if isinstance(u, str) else u for u in usage]
        out_tok = statistics.mean(u.get("completion_tokens", u.get("output_tokens", 0)) or 0 for u in usage)
        if s == "jev":
            hosted = sum(u.get("input_tokens", 0) for u in usage) * JEV_IN / len(usage) * 1000
        elif s == "nli":
            hosted = None
        else:
            hosted = sum(u.get("prompt_tokens", 0) * QIN + u.get("completion_tokens", 0) * QOUT
                         for u in usage) / len(usage) * 1000
        selfhost = (GPU_USD_PER_HOUR * BATCH_SECONDS[s] / 3600 / BATCH_CASES * 1000) if s in BATCH_SECONDS else None
        fails = sum(not r.get("prediction") for r in data[s])
        eff[s] = {"lat_median": statistics.median(lat), "lat_p95": lat[int(0.95 * len(lat)) - 1],
                  "lat_n": len(lat), "out_tokens": out_tok, "failures": fails,
                  "cost_hosted": hosted, "cost_selfhost": selfhost}
        table.append([LABEL[s], f"{eff[s]['lat_median']:.0f}", f"{eff[s]['lat_p95']:.0f}", f"{out_tok:.1f}",
                      str(fails), "—" if hosted is None else f"${hosted:.3f}",
                      "本地" if s == "nli" else ("—" if selfhost is None else f"${selfhost:.4f}")])
    out["efficiency"] = eff
    out["table_efficiency"] = table

    # paired McNemar tests, Bonferroni over all pairs
    pairs = [(a, b) for i, a in enumerate(SYSTEMS) for b in SYSTEMS[i + 1:]]
    out["mcnemar"] = {f"{a}|{b}": dict(zip(("only_a", "only_b", "p"), mcnemar(data[a], data[b]))) for a, b in pairs}
    out["bonferroni_alpha"] = 0.05 / len(pairs)

    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    for name in ("table_main", "table_calibration", "table_selective", "table_efficiency"):
        print(name)
        for row in out[name]:
            print("  " + " | ".join(row))
    print("McNemar:")
    for k, v in out["mcnemar"].items():
        print(f"  {k:45s} {v['only_a']:3d}/{v['only_b']:<3d} p={v['p']:.4g}{' *' if v['p'] < out['bonferroni_alpha'] else ''}")


if __name__ == "__main__":
    main()
