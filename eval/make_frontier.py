#!/usr/bin/env python3
"""Speed/cost--accuracy Pareto frontier figure for the paper.

Two panels: (a) mean accuracy vs median request latency, (b) mean accuracy vs
cost per 1k decisions. Accuracy counts protocol failures as errors; requests
that ended in provider errors are excluded from latency and cost. Writes
results/frontier.pdf and results/frontier.png.
"""
import json
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
PRED = ROOT / "results" / "predictions"
OUT = ROOT / "results" / "frontier.pdf"
OUT_PNG = ROOT / "results" / "frontier.png"

JEV_IN, QIN, QOUT = 0.042e-6, 0.10e-6, 0.15e-6  # $/token list prices

STYLE = {
    "jev":                    {"label": "JEV",                  "color": "#1f6fb2"},
    "nli":                    {"label": "NLI-DeBERTa",          "color": "#2a9d8f"},
    "qwen-direct-compact":    {"label": "Q-direct (compact)",   "color": "#e76f51"},
    "qwen-direct-expanded":   {"label": "Q-direct (expanded)",  "color": "#bdbdbd"},
    "qwen-cot-compact":       {"label": "Q-reason (compact)",   "color": "#bdbdbd"},
    "qwen-cot-expanded":      {"label": "Q-reason (expanded)",  "color": "#bdbdbd"},
}
OFFSETS = {
    "lat": {"jev": (8, 6, "left"), "nli": (8, 6, "left"),
            "qwen-direct-compact": (0, 10, "center"),
            "qwen-direct-expanded": (8, -13, "left"), "qwen-cot-compact": (8, 6, "left"),
            "qwen-cot-expanded": (8, -13, "left")},
    "cost1k": {"jev": (8, 6, "left"), "nli": (8, 6, "left"),
             "qwen-direct-compact": (0, -16, "center"),
             "qwen-direct-expanded": (10, -14, "left"), "qwen-cot-compact": (-8, 8, "right"),
             "qwen-cot-expanded": (8, -13, "left")},
}


def load(path: Path) -> list[dict]:
    seen = {}
    for line in open(path):
        if not line.strip():
            continue
        r = json.loads(line)
        k = r["id"]
        if k not in seen or (r["prediction"] and not seen[k]["prediction"]):
            seen[k] = r
    return list(seen.values())


def frontier(pts, key):
    out, best = [], -1.0
    for p in sorted(pts, key=lambda p: p[key]):
        if p["acc"] > best:
            out.append(p)
            best = p["acc"]
    return out


def main() -> None:
    pts = []
    for f in sorted(PRED.glob("*__full.jsonl")):
        system = f.name.split("__")[0]
        rows = load(f)
        lat = [r["latency_ms"] for r in rows if r.get("latency_ms") and not r.get("error")]
        correct = sum(1 for r in rows if r["prediction"] and r["prediction"] == r["gold"])
        cost = 0.0
        for r in rows:
            u = r.get("usage") or {}
            if u.get("input_tokens"):
                cost += u["input_tokens"] * JEV_IN
            elif u.get("prompt_tokens"):
                cost += u["prompt_tokens"] * QIN + u.get("completion_tokens", 0) * QOUT
        pts.append({"system": system,
                    "lat": statistics.median(lat),
                    "cost1k": 0.0 if system == "nli" else cost / len(rows) * 1000,
                    "acc": correct / len(rows)})

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.1, 2.9), sharey=True)

    for ax, key, xlabel, xlim in [
        (ax1, "lat", "Median request latency (ms, log scale)", None),
        (ax2, "cost1k", "Cost per 1k decisions (USD)", (-0.006, 0.118)),
    ]:
        fr = frontier(pts, key)
        names = {p["system"] for p in fr}
        for p in pts:
            on = p["system"] in names
            s = STYLE[p["system"]]
            ax.scatter(p[key], p["acc"], s=55 if on else 42,
                       color=s["color"] if on else "#c9c9c9",
                       edgecolor="white", linewidth=0.6, zorder=3)
            dx, dy, ha = OFFSETS[key][p["system"]]
            ax.annotate(s["label"], (p[key], p["acc"]),
                        textcoords="offset points", xytext=(dx, dy), ha=ha,
                        fontsize=7, color="#333333" if on else "#9a9a9a")
        ax.plot([p[key] for p in fr], [p["acc"] for p in fr],
                "--", color="#666666", lw=1.1, zorder=2)
        ax.set_xlabel(xlabel, fontsize=8.5)
        ax.tick_params(labelsize=7.5)
        ax.grid(axis="y", color="#e6e6e6", lw=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        if xlim:
            ax.set_xlim(*xlim)
        else:
            ax.set_xscale("log")

    ax1.set_ylabel("Pooled accuracy (failures as errors)", fontsize=8.5)
    ax1.set_ylim(0.58, 0.78)
    ax2.annotate("local $\\approx$ \\$0", (0.001, 0.600), fontsize=6.5, color="#9a9a9a")
    fig.tight_layout()
    fig.savefig(OUT)
    fig.savefig(OUT_PNG, dpi=220)
    print("wrote", OUT, "and", OUT_PNG)
    for key in ("lat", "cost1k"):
        print(f"  {key} frontier:", [STYLE[p["system"]]["label"] for p in frontier(pts, key)])


if __name__ == "__main__":
    main()
