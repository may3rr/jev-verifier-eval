#!/usr/bin/env python3
"""Markdown tables 3-6 and the old/new comparison of every number the text quotes.

Old: results/paper_numbers.json (original). New: <data>/paper_numbers.json.
The old file has no "derived" block, so the old side is recomputed from the
original files with paper_numbers_rev.py on data_orig (merge.py --scope none with
no reruns reproduces the original JSON exactly, checked on 2026-09-27).
Usage: python3 report_tables.py OLD_DIR NEW_DIR > tables.md
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

S = ["jev", "qwen-direct-compact", "qwen-direct-expanded", "qwen-cot-compact", "qwen-cot-expanded", "nli"]
DS = ["fever", "scifact", "hover", "vitaminc", "climate_fever"]
DSL = {"fever": "FEVER", "scifact": "SciFact", "hover": "HoVer", "vitaminc": "VitaminC",
       "climate_fever": "Climate-FEVER", "pooled": "合并"}


def md(rows: list[list[str]]) -> str:
    head = [c.replace("\n", "") for c in rows[0]]
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    out += ["| " + " | ".join(c.replace("\n", "") for c in r) + " |" for r in rows[1:]]
    return "\n".join(out)


def tables(n: dict) -> str:
    main = n["table_main"]
    head = ["系统"] + [f"{DSL[d]} {m}" for d in DS + ["pooled"] for m in ("Acc", "P", "F1")]
    parts = ["**表3　各模型在五个数据集上的准确率、宏平均精确率与宏平均F1值（%）**", "", md([head] + main[1:]), "",
             "**表4　各模型的Brier分数与期望校准误差**", "", md(n["table_calibration"]), "",
             "**表5　选择性核验结果**", "", md(n["table_selective"]), "",
             "**表6　效率、协议可靠性与成本**", "", md(n["table_efficiency"])]
    return "\n".join(parts)


def quoted(n: dict) -> list[tuple[str, str, str]]:
    """(location, description, value) for every number the text quotes."""
    m, cal, sel, eff, esc, d = n["metrics"], n["calibration"], n["selective"], n["efficiency"], n["escalation"], n["derived"]
    acc = lambda s, ds="pooled": f"{m[s][ds][0]:.3f}"
    pc = lambda x: f"{100 * x:.1f}%"
    q = []
    add = lambda loc, what, v: q.append((loc, what, v))
    add("摘要/5.1/7", "JEV 合并准确率", acc("jev"))
    add("摘要/5.1", "Qwen零样本直答 合并准确率", acc("qwen-direct-compact"))
    add("5.1", "JEV 合并宏F1", f"{m['jev']['pooled'][2]:.3f}")
    add("5.1", "Qwen零样本直答 合并宏F1", f"{m['qwen-direct-compact']['pooled'][2]:.3f}")
    add("5.1", "Qwen少样本直答 合并准确率", acc("qwen-direct-expanded"))
    add("5.1", "Qwen零样本推理 合并准确率", acc("qwen-cot-compact"))
    add("5.1", "Qwen少样本推理 合并准确率", acc("qwen-cot-expanded"))
    add("5.1", "NLI 合并准确率", acc("nli"))
    for ds in DS:
        add("5.1", f"JEV−Qwen零样本直答 {DSL[ds]} 准确率差（百分点）",
            f"{100 * (m['jev'][ds][0] - m['qwen-direct-compact'][ds][0]):+.1f}")
    add("5.1", "等权平均准确率 JEV", f"{d['equal_weight_acc']['jev']:.3f}")
    add("5.1", "等权平均准确率 Qwen零样本直答", f"{d['equal_weight_acc']['qwen-direct-compact']:.3f}")
    for s in S[1:5]:
        add("5.1（新增）", f"等权平均准确率 {n['systems'][s]}", f"{d['equal_weight_acc'][s]:.3f}")
    for k, lab in (("qwen-direct-compact>qwen-cot-compact", "零样本"), ("qwen-direct-expanded>qwen-cot-expanded", "少样本")):
        for ds in DS + ["pooled"]:
            add("5.1/图3", f"推理−直答 {lab} {DSL[ds]}（百分点）", f"{d['cot_minus_direct_pp'][k][ds]:+.1f}")
    for s in ("qwen-cot-compact", "qwen-cot-expanded"):
        add("摘要/5.2", f"{n['systems'][s]} 置信度≥0.99占比", pc(cal[s]["saturated"]))
        add("摘要/5.2", f"{n['systems'][s]} 合并ECE", f"{cal[s]['ece']:.3f}")
        add("5.2", f"{n['systems'][s]} Climate-FEVER Brier", f"{cal[s]['brier']['climate_fever']:.3f}")
    add("5.2", "Qwen少样本直答 合并ECE", f"{cal['qwen-direct-expanded']['ece']:.3f}")
    add("5.3", "Qwen零样本推理 自动判定30%准确率", f"{sel['qwen-cot-compact']['acc']['0.3']:.3f}")
    add("5.3", "Qwen少样本推理 自动判定30%准确率", f"{sel['qwen-cot-expanded']['acc']['0.3']:.3f}")
    for s in S:
        add("摘要/5.3/7", f"{n['systems'][s]} 0.90目标自动判定比例", pc(sel[s]["auto_rate"]["0.90"]))
        add("5.3", f"{n['systems'][s]} 0.95目标自动判定比例", pc(sel[s]["auto_rate"]["0.95"]))
    add("5.3", "HoVer 0.90目标下各模型最大自动判定比例",
        pc(max(sel[s]["by_dataset"]["hover"]["auto_rate"]["0.90"] for s in S)))
    for s in S:
        add("5.3/图6", f"{n['systems'][s]} HoVer AURC", f"{sel[s]['by_dataset']['hover']['aurc']:.3f}")
    for s in S:
        add("5.4/表6", f"{n['systems'][s]} 中位延迟 ms", f"{eff[s]['lat_median']:.0f}")
        add("5.4/表6", f"{n['systems'][s]} p95延迟 ms", f"{eff[s]['lat_p95']:.0f}")
        add("表6", f"{n['systems'][s]} 平均输出词元", f"{eff[s]['out_tokens']:.1f}")
        add("5.5/表6/图9", f"{n['systems'][s]} 协议失败", str(eff[s]["failures"]))
        if eff[s]["cost_hosted"] is not None:
            add("5.4/表6", f"{n['systems'][s]} 每千次成本（托管价）", f"${eff[s]['cost_hosted']:.3f}")
    for k, lab in (("qwen-direct-compact>qwen-cot-compact", "零样本"), ("qwen-direct-expanded>qwen-cot-expanded", "少样本")):
        v = d["latency"][k]
        add("5.4", f"{lab}推理/直答 中位延迟倍数", f"{v['median_ratio']:.1f}")
        add("5.4", f"{lab}推理/直答 p95延迟倍数", f"{v['p95_ratio']:.1f}")
        add("5.4", f"{lab}推理 超过10秒的请求占比", pc(v["share_over_10s"]))
        add("5.4", f"{lab}推理/直答 成本增幅", f"{100 * (v['cost_ratio'] - 1):.0f}%")
    for k, lab in (("qwen-direct-compact>qwen-cot-compact", "Qwen直答→推理"), ("jev>qwen-cot-compact", "JEV→推理")):
        add("摘要/6.1", f"级联 {lab} 合并", f"{esc[k]['fast']:.3f}→{esc[k]['cascade']:.3f}")
        for ds in DS:
            x = esc[k]["by_dataset"][ds]
            add("6.1", f"级联 {lab} {DSL[ds]}", f"{x['fast']:.3f}→{x['cascade']:.3f}")
    for s in S:
        add("6.2", f"{n['systems'][s]} 每日一万条需复核（0.90目标）", f"{d['review_per_10k_at_0.90'][s]:.0f}")
    return q


def main() -> None:
    old = json.load(open(Path(sys.argv[1]) / "paper_numbers.json"))
    new = json.load(open(Path(sys.argv[2]) / "paper_numbers.json"))
    print(tables(new))
    print("\n\n## 新旧对照（全部文中数字，变化的加粗）\n")
    print("| 位置 | 数字 | 旧值 | 新值 |\n|---|---|---|---|")
    for (loc, what, a), (_, _, b) in zip(quoted(old), quoted(new)):
        print(f"| {loc} | {what} | {a} | {'**' + b + '**' if a != b else b} |")


if __name__ == "__main__":
    main()
