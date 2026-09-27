#!/usr/bin/env python3
"""Revision copy of eval/paper_numbers.py (2026-09-27): same metrics, revised data.

Reads the files assembled by merge.py (--data, default data_all): Qwen rows with the
label-safe HoVer reruns and the 2048-token truncation reruns; JEV and NLI unchanged.
Qwen latency comes from <data>/latency, JEV and NLI latency from the original files.
Writes <data>/paper_numbers.json. Everything below the data paths is the original
script, plus derived() at the end for the numbers quoted in the text.

Original docstring:
Every number the Chinese manuscript reports, computed from the prediction logs.

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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "eval"))

from metrics import brier, ece
from selective import summarise

REV = ROOT / "revision_2026-09-27"
DATA = REV / "data_all"
PRED = DATA / "predictions"
LAT = DATA / "latency"
ORIG_PRED = ROOT / "results" / "predictions"
SENS = ROOT / "results" / "sensitivity"
OUT = DATA / "paper_numbers.json"


def set_data(path: Path) -> None:
    global DATA, PRED, LAT, OUT
    DATA, PRED, LAT, OUT = path, path / "predictions", path / "latency", path / "paper_numbers.json"

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
ESCALATE = 0.2  # share of least-confident fast answers handed to reasoning in §6.1


def load(path: Path) -> list[dict]:
    latest = {}
    for line in path.open():
        if line.strip():
            row = json.loads(line)
            latest[row["id"]] = row
    return list(latest.values())


def load_system(system: str) -> list[dict]:
    return load(next(PRED.glob(f"{system}__*__{TAG}.jsonl")))


def latency_sample(system: str) -> list[dict]:
    """Latency rows: the same 150 cases for every system, one request at a time, time spent serving.

    Qwen and JEV have dedicated runs on these cases; JEV's round trip over the
    internet is measured and removed in measure_jev_latency.py. NLI ran one case at
    a time on the serving machine in its full-split run, which holds 138 of them.
    """
    files = sorted(LAT.glob(f"{system}__lat1.jsonl")) or sorted(SENS.glob(f"{system}__*lat1.jsonl"))
    if files:
        rows = load(files[0])
    else:
        ids = {r["id"] for r in load(next(SENS.glob("qwen-direct-compact__*__vllm-lat1.jsonl")))}
        rows = [r for r in load(next(ORIG_PRED.glob(f"{system}__*__fullset.jsonl"))) if r["id"] in ids]
    return [r for r in rows if r.get("latency_ms") is not None and not r.get("error")]


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


def escalation(fast: list[dict], slow: list[dict]) -> dict:
    """Fast answer decides; its least confident ESCALATE share is answered by the slow setting.

    Cases are ranked as in selective.py (top-label probability, ties by id).
    Returns pooled and per-dataset accuracy of the fast setting alone and of the cascade.
    """
    slow_by_id = {r["id"]: r for r in slow}
    ranked = sorted(fast, key=lambda r: (-(max(r["distribution"].values()) if r.get("prediction") and r.get("distribution") else -1.0), r["id"]))
    handed = {r["id"] for r in ranked[len(ranked) - round(len(ranked) * ESCALATE):]}
    final = [slow_by_id[r["id"]] if r["id"] in handed else r for r in fast]

    def acc(rows: list[dict]) -> float:
        return sum(correct(r) for r in rows) / len(rows)
    res = {"fast": acc(fast), "cascade": acc(final), "escalated": len(handed), "by_dataset": {}}
    for d in DATASETS:
        f = [r for r in fast if r["dataset"] == d]
        c = [r for r in final if r["dataset"] == d]
        res["by_dataset"][d] = {"fast": acc(f), "cascade": acc(c), "escalated": sum(r["id"] in handed for r in f)}
    return res


def derived(out: dict, data: dict) -> dict:
    """Numbers the manuscript quotes in the text, from the same inputs as the tables."""
    m, cal, sel, eff = out["metrics"], out["calibration"], out["selective"], out["efficiency"]
    pairs = (("qwen-direct-compact", "qwen-cot-compact"), ("qwen-direct-expanded", "qwen-cot-expanded"))
    d: dict = {"equal_weight_acc": {s: statistics.mean(m[s][ds][0] for ds in DATASETS) for s in SYSTEMS}}
    d["cot_minus_direct_pp"] = {f"{a}>{b}": {ds: 100 * (m[b][ds][0] - m[a][ds][0]) for ds in DATASETS + ["pooled"]}
                                for a, b in pairs}
    d["review_per_10k_at_0.90"] = {s: 10000 * (1 - sel[s]["auto_rate"]["0.90"]) for s in SYSTEMS}
    lat = {s: sorted(r["latency_ms"] for r in latency_sample(s)) for s in SYSTEMS}
    d["latency"] = {f"{a}>{b}": {"median_ratio": eff[b]["lat_median"] / eff[a]["lat_median"],
                                 "p95_ratio": eff[b]["lat_p95"] / eff[a]["lat_p95"],
                                 "share_over_10s": sum(x > 10000 for x in lat[b]) / len(lat[b]),
                                 "cost_ratio": eff[b]["cost_hosted"] / eff[a]["cost_hosted"]} for a, b in pairs}
    d["failures"] = {}
    for s in SYSTEMS:
        c = {}
        for r in data[s]:
            if "-cot-" in s and r.get("finish_reason") == "length":
                key = "length_with_judgement" if r.get("prediction") else "length_no_judgement"
            elif not r.get("prediction"):
                key = "error" if r.get("error") else "no_legal_label"
            else:
                continue
            c[f"{r['dataset']}:{key}"] = c.get(f"{r['dataset']}:{key}", 0) + 1
        d["failures"][s] = c
    return d


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=None, help="folder written by merge.py (default data_all)")
    parser.add_argument("--no-ci", action="store_true",
                        help="skip the bootstrap intervals (not reported in the paper; tables unchanged)")
    args = parser.parse_args()
    if args.data:
        set_data(Path(args.data).resolve())
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
        res = summarise(data[s], with_ci=not args.no_ci)
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

    # efficiency: latency from the shared 150-case sample, tokens and cost from the main run
    eff, table = {}, [["系统", "中位延迟\n(ms)", "p95延迟\n(ms)", "平均输出\n词元", "协议失败", "每千次成本\n(托管价)", "每千次成本\n(自建)"]]
    for s in SYSTEMS:
        lat = sorted(r["latency_ms"] for r in latency_sample(s))
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

    # fast-to-slow escalation (§6.1): same model direct -> reasoning, and JEV -> reasoning
    out["escalation"] = {f"{a}>{b}": escalation(data[a], data[b])
                         for a, b in (("qwen-direct-compact", "qwen-cot-compact"), ("jev", "qwen-cot-compact"))}

    # paired McNemar tests, Bonferroni over all pairs
    pairs = [(a, b) for i, a in enumerate(SYSTEMS) for b in SYSTEMS[i + 1:]]
    out["mcnemar"] = {f"{a}|{b}": dict(zip(("only_a", "only_b", "p"), mcnemar(data[a], data[b]))) for a, b in pairs}
    out["bonferroni_alpha"] = 0.05 / len(pairs)

    out["derived"] = derived(out, data)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    for name in ("table_main", "table_calibration", "table_selective", "table_efficiency"):
        print(name)
        for row in out[name]:
            print("  " + " | ".join(row))
    print(f"Escalation of the least confident {ESCALATE:.0%}:")
    for k, v in out["escalation"].items():
        per = "  ".join(f"{d}={x['fast']:.3f}->{x['cascade']:.3f}" for d, x in v["by_dataset"].items())
        print(f"  {k:40s} {v['fast']:.4f} -> {v['cascade']:.4f}  ({per})")
    print("McNemar:")
    for k, v in out["mcnemar"].items():
        print(f"  {k:45s} {v['only_a']:3d}/{v['only_b']:<3d} p={v['p']:.4g}{' *' if v['p'] < out['bonferroni_alpha'] else ''}")


if __name__ == "__main__":
    main()
