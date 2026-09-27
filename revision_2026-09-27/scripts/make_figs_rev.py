#!/usr/bin/env python3
"""Revision copy of eval/make_figs.py (2026-09-27): same figures and style, revised data.

Reads the files assembled by merge.py (--data, default data_all) and writes
fig2..fig9 as PNG (300 dpi) and SVG to revision_2026-09-27/figs. Only the data
paths, the output names and the PNG resolution differ from the original script.

Original docstring:
All analysis figures for the Chinese manuscript, in one shared style.

Figures (results/figs/*.png and *.pdf):
  fig_acc_dataset      per-dataset accuracy with Wilson 95% intervals
  fig_fast_slow        direct answer vs reasoning, per dataset (dumbbell)
  fig_selective        pooled selective-verification curves
  fig_selective_ds     selective-verification curves per dataset
  fig_reliability      reliability diagrams with confidence histograms
  fig_latency          empirical CDF of request latency
  fig_frontier         accuracy against latency and cost
  fig_failures         protocol failures by type

Style: matplotlib defaults at 13pt, DejaVu Sans for Latin letters and digits and
宋体 for Chinese. JEV is the method under study and is drawn as a thick solid
line in a warm color; every other system is a thin dashed line in a muted color,
with the four Qwen settings sharing a blue-green family. In dot plots a filled
marker means fast thinking and a hollow one slow thinking. Every figure is
10.5 in wide, so at the 15.5 cm text width all text prints at the same size,
about 7.5pt.

Qwen rows come from the tag given by --qwen-tag (default: eval, the 9,681-case
evaluation set). Latency is time spent serving, on the same 150 cases for every
system (paper_numbers.latency_sample). Systems
without label distributions are left out of the calibration and selective
figures automatically.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "eval"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FormatStrFormatter, FuncFormatter, MultipleLocator

import figstyle as fs
import paper_numbers_rev
from paper_numbers_rev import latency_sample
from prompts import parse_cot, parse_direct
from selective import acc_at, ranked_correct
from metrics import ece as ece_table

# ---------------------------------------------------------------- style

FONT_SIZE = 13
FONT_FAMILY = ["DejaVu Sans", fs.SONG]  # per-glyph fallback: Latin and digits from DejaVu Sans, Chinese from 宋体
FIG_W = 10.5            # inches; the paper scales every figure to the 6.1 in text width
PNG_DPI = 300
GRID_ALPHA = 0.3
LEGEND = {"loc": "lower center", "frameon": False, "handlelength": 3}
LEGEND_GAP = 0.12       # inches between the bottom legend and the axes above it

MAIN = "jev"
MAIN_LINE = {"linewidth": 2.2, "linestyle": "-", "zorder": 10}
BASE_LINE = {"linewidth": 1.2, "linestyle": "--", "zorder": 3}
REF_LINE = {"color": "#8c8c8c", "linewidth": 1.1, "linestyle": ":", "zorder": 1}  # targets, calibration, frontier
MAIN_MS, BASE_MS = 10, 7.5
NOTE = "#555555"        # annotation text

SYSTEMS = [
    "jev", "nli",
    "qwen-direct-compact", "qwen-direct-expanded",
    "qwen-cot-compact", "qwen-cot-expanded",
]
LABEL = {
    "jev": "JEV",
    "nli": "NLI-DeBERTa",
    "qwen-direct-compact": "Qwen零样本直答",
    "qwen-direct-expanded": "Qwen少样本直答",
    "qwen-cot-compact": "Qwen零样本推理",
    "qwen-cot-expanded": "Qwen少样本推理",
}
# JEV: Okabe-Ito vermilion. Baselines: muted, the Qwen settings in one blue-green
# family (direct in blue, reasoning in green-teal, zero-shot darker), NLI in plum.
# Every pair is at least 13 ΔE apart (OKLab ×100) and at least 7 under simulated
# protanopia and deuteranopia; line width and dash carry the main/baseline split in print.
COLOR = {
    "jev": "#D55E00",
    "nli": "#966685",
    "qwen-direct-compact": "#30537F",
    "qwen-direct-expanded": "#6A8DC7",
    "qwen-cot-compact": "#1B7053",
    "qwen-cot-expanded": "#51AA94",
}
MARKER = {
    "jev": "o",
    "nli": "s",
    "qwen-direct-compact": "^",
    "qwen-direct-expanded": "D",
    "qwen-cot-compact": "^",
    "qwen-cot-expanded": "D",
}
SLOW = {"qwen-cot-compact", "qwen-cot-expanded"}
FAILURE_COLOR = {"space": "#3b3b3b", "length": "#8c8c8c", "other": "#cfcfcf"}

DATASETS = ["fever", "scifact", "hover", "vitaminc", "climate_fever"]
DATASET_LABEL = {
    "fever": "FEVER",
    "scifact": "SciFact",
    "hover": "HoVer",
    "vitaminc": "VitaminC",
    "climate_fever": "Climate-FEVER",
}


def apply_style() -> None:
    fs.register_fonts()
    plt.rcdefaults()
    plt.rcParams.update({
        "font.family": FONT_FAMILY,
        "font.size": FONT_SIZE,
        "pdf.fonttype": 42,
    })


def line_kw(s: str) -> dict:
    return {"color": COLOR[s], **(MAIN_LINE if s == MAIN else BASE_LINE)}


def marker_kw(s: str, size: float | None = None) -> dict:
    """Filled marker for fast thinking, hollow for slow thinking; JEV larger and on top."""
    main = s == MAIN
    return {
        "marker": MARKER[s],
        "markersize": size or (MAIN_MS if main else BASE_MS),
        "markeredgecolor": COLOR[s],
        "markeredgewidth": 1.6 if main else 1.2,
        "markerfacecolor": "white" if s in SLOW else COLOR[s],
        "color": COLOR[s],
        "zorder": 11 if main else 5,
    }


def line_handles(systems) -> list:
    return [Line2D([], [], label=LABEL[s], **line_kw(s)) for s in systems]


def marker_handles(systems) -> list:
    return [Line2D([], [], linestyle="none", label=LABEL[s], **marker_kw(s)) for s in systems]


def grid(ax, axis: str = "both") -> None:
    ax.grid(axis=axis, alpha=GRID_ALPHA)
    ax.set_axisbelow(True)


def unit_ticks(axis, step: float, fmt: str) -> None:
    axis.set_major_locator(MultipleLocator(step))
    axis.set_major_formatter(FormatStrFormatter(fmt))


def data_ylim(values, step: float = 0.1, top: float = 1.0) -> tuple[float, float]:
    """From the tick at or below the data minimum to `top`, with a small margin on both ends."""
    low = math.floor(min(values) / step + 1e-9) * step
    pad = 0.03 * (top - low)
    return low - pad, top + pad


def bottom_legend(fig, handles, ncol: int, **kw) -> None:
    """Legend centred under the whole figure; tight_layout leaves exactly its height free."""
    legend = fig.legend(handles=handles, ncol=ncol, bbox_to_anchor=(0.5, 0.0), **{**LEGEND, **kw})
    fig.canvas.draw()
    height = legend.get_window_extent().height / fig.dpi + LEGEND_GAP
    fig.tight_layout(rect=[0, height / fig.get_figheight(), 1, 1])


OUT_FIGS = fs.ROOT / "revision_2026-09-27" / "figs"
FIG_NO = {"fig_acc_dataset": 2, "fig_fast_slow": 3, "fig_reliability": 4, "fig_selective": 5,
          "fig_selective_ds": 6, "fig_latency": 7, "fig_frontier": 8, "fig_failures": 9}


def save(fig, name: str) -> None:
    OUT_FIGS.mkdir(parents=True, exist_ok=True)
    stem = f"fig{FIG_NO[name]}"
    fig.savefig(OUT_FIGS / f"{stem}.svg", bbox_inches="tight")
    fig.savefig(OUT_FIGS / f"{stem}.png", dpi=PNG_DPI, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- data

PRED = paper_numbers_rev.PRED
JEV_IN, QIN, QOUT = 0.042e-6, 0.10e-6, 0.15e-6  # $/token list prices
FAILURE_KINDS = [("space", "判断空间外的答案"), ("length", "推理被截断"), ("other", "无法解析或请求失败")]


def load_file(path: Path) -> list[dict]:
    seen: dict[str, dict] = {}
    for line in path.open():
        if not line.strip():
            continue
        row = json.loads(line)
        prev = seen.get(row["id"])
        if prev is None or (row.get("prediction") and not prev.get("prediction")):
            seen[row["id"]] = row
    return list(seen.values())


def load_all(qwen_tag: str) -> dict[str, list[dict]]:
    data = {}
    for system in SYSTEMS:
        # the eval set holds every system; the older runs kept JEV and NLI under "full"
        tag = qwen_tag if system.startswith("qwen") or qwen_tag == "eval" else "full"
        files = sorted(PRED.glob(f"{system}__*__{tag}.jsonl"))
        if files:
            data[system] = load_file(files[0])
    return data


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def scorable_acc(rows: list[dict]) -> tuple[float, int, int]:
    """Accuracy with protocol failures counted as errors, as in the main table."""
    k = sum(1 for r in rows if r.get("prediction") and r["prediction"] == r["gold"])
    return (k / len(rows) if rows else float("nan")), k, len(rows)


def by_dataset(rows: list[dict], ds: str) -> list[dict]:
    return [r for r in rows if r["dataset"] == ds]


def has_dist(rows: list[dict]) -> bool:
    return sum(1 for r in rows if r.get("distribution")) > 0.9 * len(rows)


def failure_type(row: dict) -> str | None:
    if row.get("prediction"):
        return None
    if row.get("error"):
        return "other"
    cot = "-cot-" in row.get("system", "")
    # direct prompts allow one token, so a length stop there is an unparsable token, not a truncation
    if cot and row.get("finish_reason") == "length":
        return "length"
    parser = parse_cot if cot else parse_direct
    return "space" if parser(row.get("raw") or "") else "other"


def accuracy_table(data) -> dict:
    """{system: {dataset: (accuracy, wilson low, wilson high)}}"""
    out = {}
    for s in (s for s in SYSTEMS if s in data):
        out[s] = {}
        for ds in DATASETS:
            acc, k, n = scorable_acc(by_dataset(data[s], ds))
            out[s][ds] = (acc, *wilson(k, n))
    return out


def fast_slow_pairs(data) -> list[dict]:
    """Per prompt: direct and reasoning accuracy on each dataset plus their five-dataset mean."""
    pairs = [("qwen-direct-compact", "qwen-cot-compact", "零样本提示"),
             ("qwen-direct-expanded", "qwen-cot-expanded", "少样本提示")]
    out = []
    for fast, slow, title in pairs:
        if fast not in data or slow not in data:
            continue
        fast_acc = [scorable_acc(by_dataset(data[fast], ds))[0] for ds in DATASETS]
        slow_acc = [scorable_acc(by_dataset(data[slow], ds))[0] for ds in DATASETS]
        out.append({"fast": fast, "slow": slow, "title": title,
                    "fast_acc": fast_acc + [float(np.mean(fast_acc))],
                    "slow_acc": slow_acc + [float(np.mean(slow_acc))]})
    return out


def selective_curves(data, coverage: np.ndarray, dataset: str | None = None) -> dict:
    """{system: accuracy of the auto-decided cases at each coverage}"""
    out = {}
    for s in (s for s in SYSTEMS if s in data and has_dist(data[s])):
        rows = data[s] if dataset is None else by_dataset(data[s], dataset)
        correct = ranked_correct(rows)
        out[s] = np.array([acc_at(correct, x) for x in coverage])
    return out


def reliability_bins(data, bins: int = 10) -> dict:
    """{system: (mean confidence per bin, accuracy per bin, ECE, all confidences)} with equal-count bins."""
    out = {}
    for s in (s for s in SYSTEMS if s in data and has_dist(data[s])):
        # sort by confidence only (stable), as metrics.ece does, so tied confidences bin the same way
        pairs = sorted(((max(r["distribution"].values()), int(r["prediction"] == r["gold"]))
                        for r in data[s] if r.get("distribution") and r.get("prediction")), key=lambda p: p[0])
        chunks = np.array_split(np.array(pairs), bins)
        conf = [c[:, 0].mean() for c in chunks]
        acc = [c[:, 1].mean() for c in chunks]
        ece = ece_table(data[s])  # the number printed in table 4
        out[s] = (conf, acc, ece, np.array([p[0] for p in pairs]))
    return out


def latency_ecdf(data) -> dict:
    """{system: (sorted latencies in ms, cumulative share, median)}"""
    out = {}
    for s in (s for s in SYSTEMS if s in data):
        lat = np.sort([r["latency_ms"] for r in latency_sample(s)])
        out[s] = (lat, np.arange(1, len(lat) + 1) / len(lat), float(np.median(lat)))
    return out


def frontier_points(data) -> list[dict]:
    """Pooled accuracy, median latency and hosted cost per thousand judgements, per system."""
    pts = []
    for s in (s for s in SYSTEMS if s in data):
        rows = data[s]
        lat = [r["latency_ms"] for r in latency_sample(s)]
        cost = 0.0
        for r in rows:
            u = r.get("usage") or {}
            if u.get("input_tokens"):
                cost += u["input_tokens"] * JEV_IN
            elif u.get("prompt_tokens"):
                cost += u["prompt_tokens"] * QIN + u.get("completion_tokens", 0) * QOUT
        pts.append({"s": s, "lat": statistics.median(lat), "acc": scorable_acc(rows)[0],
                    "cost": 0.0 if s == "nli" else cost / len(rows) * 1000})
    return pts


def pareto(pts: list[dict], key: str) -> list[dict]:
    out, best = [], -1.0
    for p in sorted(pts, key=lambda p: p[key]):
        if p["acc"] > best:
            out.append(p)
            best = p["acc"]
    return out


def failure_counts(data) -> dict:
    """{system: ({failure kind: count}, requests)}"""
    out = {}
    for s in (s for s in SYSTEMS if s in data):
        counts = {k: 0 for k, _ in FAILURE_KINDS}
        for r in data[s]:
            kind = failure_type(r)
            if kind:
                counts[kind] += 1
        out[s] = (counts, len(data[s]))
    return out


# ---------------------------------------------------------------- figures

def plot_acc_dataset(table: dict) -> None:
    systems = list(table)
    fig, ax = plt.subplots(figsize=(FIG_W, 6.4))
    offsets = np.linspace(0.32, -0.32, len(systems))  # JEV on top of each group
    for i, ds in enumerate(DATASETS):
        y0 = len(DATASETS) - 1 - i
        for off, s in zip(offsets, systems):
            acc, lo, hi = table[s][ds]
            width = MAIN_LINE["linewidth"] if s == MAIN else BASE_LINE["linewidth"]
            ax.plot([lo, hi], [y0 + off] * 2, color=COLOR[s], linewidth=width,
                    zorder=10 if s == MAIN else 4, solid_capstyle="butt")
            ax.plot([acc], [y0 + off], linestyle="none", **marker_kw(s))
    ax.set_yticks(range(len(DATASETS)))
    ax.set_yticklabels([DATASET_LABEL[d] for d in reversed(DATASETS)])
    ax.set_ylim(-0.55, len(DATASETS) - 0.45)
    ax.set_xlim(0.4, 1.0)
    unit_ticks(ax.xaxis, 0.1, "%.1f")
    ax.set_xlabel("准确率")
    ax.set_ylabel("数据集")
    grid(ax, "x")
    handles = [Line2D([], [], label=LABEL[s], linewidth=MAIN_LINE["linewidth"] if s == MAIN else BASE_LINE["linewidth"],
                      **{k: v for k, v in marker_kw(s).items() if k != "zorder"}) for s in systems]
    bottom_legend(fig, handles, ncol=3)
    save(fig, "fig_acc_dataset")


def plot_fast_slow(pairs: list[dict]) -> None:
    rows = [DATASET_LABEL[d] for d in DATASETS] + ["五数据集均值"]
    fig, axes = plt.subplots(1, len(pairs), figsize=(FIG_W, 5.6), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, p in zip(axes, pairs):
        for i, (a, b) in enumerate(zip(p["fast_acc"], p["slow_acc"])):
            y = len(rows) - 1 - i
            ax.plot([a, b], [y, y], color="#c4c4c4", linewidth=2.4, zorder=1, solid_capstyle="round")
            ax.plot([a], [y], linestyle="none", **marker_kw(p["fast"], 9))
            ax.plot([b], [y], linestyle="none", **marker_kw(p["slow"], 9))
            delta = round(b * 100, 1) - round(a * 100, 1)  # from the one-decimal values printed in table 3
            ax.text(1.02, y, f"{delta:+.1f}".replace("-", "−"), transform=ax.get_yaxis_transform(),
                    va="center", ha="left", color="black" if abs(delta) >= 1 else NOTE,
                    weight="bold" if i == len(rows) - 1 else "normal")
        ax.text(1.02, len(rows) - 0.45, "推理−直答\n（百分点）", transform=ax.get_yaxis_transform(),
                ha="left", va="bottom", fontsize=FONT_SIZE - 2, color=NOTE)
        ax.axhline(0.5, color="#8c8c8c", linewidth=0.8)
        ax.set_title(p["title"])
        ax.set_xlim(0.45, 1.0)
        unit_ticks(ax.xaxis, 0.1, "%.1f")
        ax.set_ylim(-0.6, len(rows) - 0.4)
        ax.set_xlabel("准确率")
        grid(ax, "x")
    axes[0].set_yticks(range(len(rows)))
    axes[0].set_yticklabels(list(reversed(rows)))
    axes[0].set_ylabel("数据集")
    handles = marker_handles([h for p in pairs for h in (p["fast"], p["slow"])])
    bottom_legend(fig, handles, ncol=4, handletextpad=0.4)
    save(fig, "fig_fast_slow")


def plot_selective(coverage: np.ndarray, curves: dict) -> None:
    fig, ax = plt.subplots(figsize=(FIG_W, 5.4))
    for s, curve in curves.items():
        ax.plot(coverage, curve, **line_kw(s))
    for target in (0.90, 0.95):
        ax.axhline(target, **REF_LINE)
        ax.text(0.995, target + 0.003, f"目标准确率 {target:.2f}", color=NOTE, fontsize=FONT_SIZE - 2,
                va="bottom", ha="right", transform=ax.get_yaxis_transform())
    ax.set_xlim(-0.02, 1.02)
    unit_ticks(ax.xaxis, 0.2, "%.1f")
    ax.set_ylim(*data_ylim(np.concatenate(list(curves.values()))))
    unit_ticks(ax.yaxis, 0.1, "%.1f")
    ax.set_xlabel("自动判定比例（其余送人工复核）")
    ax.set_ylabel("自动判定部分的准确率")
    grid(ax)
    bottom_legend(fig, line_handles(curves), ncol=3)
    save(fig, "fig_selective")


def plot_selective_ds(coverage: np.ndarray, curves: dict) -> None:
    # three panels on top, two centred below; y ranges differ by dataset, so no shared y axis
    fig = plt.figure(figsize=(FIG_W, 8.2))
    grid_spec = fig.add_gridspec(2, 6)
    slots = [grid_spec[0, 0:2], grid_spec[0, 2:4], grid_spec[0, 4:6], grid_spec[1, 1:3], grid_spec[1, 3:5]]
    for slot, ds in zip(slots, DATASETS):
        ax = fig.add_subplot(slot)
        for s, curve in curves[ds].items():
            ax.plot(coverage, curve, **line_kw(s))
            ax.plot([coverage[-1]], [curve[-1]], linestyle="none", **marker_kw(s, 8 if s == MAIN else 6))
        ax.axhline(0.90, **REF_LINE)
        ax.set_title(DATASET_LABEL[ds])
        ax.set_xlim(-0.02, 1.04)
        unit_ticks(ax.xaxis, 0.2, "%.1f")
        ax.set_ylim(*data_ylim(np.concatenate(list(curves[ds].values()))))
        unit_ticks(ax.yaxis, 0.1, "%.1f")
        ax.set_xlabel("自动判定比例")
        ax.set_ylabel("准确率")
        grid(ax)
    systems = list(curves[DATASETS[0]])
    handles = line_handles(systems) + [Line2D([], [], label="目标准确率 0.90", **REF_LINE)]
    bottom_legend(fig, handles, ncol=4)
    save(fig, "fig_selective_ds")


def plot_reliability(bins: dict) -> None:
    systems = list(bins)
    ncol = 3
    nrow = math.ceil(len(systems) / ncol)
    # each panel stacks the reliability curve over its confidence histogram, so this
    # figure uses constrained layout, which handles the nested grid
    fig = plt.figure(figsize=(FIG_W, 4.3 * nrow), layout="constrained")
    outer = fig.add_gridspec(nrow, ncol)
    for j, s in enumerate(systems):
        r, c = divmod(j, ncol)
        inner = outer[r, c].subgridspec(2, 1, height_ratios=[3.2, 1])
        ax = fig.add_subplot(inner[0])
        hax = fig.add_subplot(inner[1], sharex=ax)
        conf, acc, ece, allconf = bins[s]
        ax.plot([0, 1], [0, 1], **REF_LINE)
        ax.plot(conf, acc, **line_kw(s))
        ax.plot(conf, acc, linestyle="none", **marker_kw(s, 6.5 if s == MAIN else 5))
        ax.set_title(LABEL[s])
        ax.text(0.04, 0.95, f"ECE = {ece:.3f}", transform=ax.transAxes, ha="left", va="top")
        ax.set_xlim(0.3, 1.02)
        ax.set_ylim(0.17, 1.03)
        unit_ticks(ax.yaxis, 0.2, "%.1f")
        ax.tick_params(labelbottom=False)
        ax.set_ylabel("实际准确率")
        grid(ax)
        hax.hist(allconf, bins=np.linspace(0.3, 1.0, 15), color=COLOR[s], alpha=0.6,
                 edgecolor="white", linewidth=0.6)
        hax.set_yticks([])
        unit_ticks(hax.xaxis, 0.2, "%.1f")
        hax.set_xlabel("最高类别概率")
        hax.set_ylabel("案例数")
        grid(hax, "x")
    handles = line_handles(systems) + [Line2D([], [], label="完全校准", **REF_LINE)]
    fig.legend(handles=handles, ncol=4, **{**LEGEND, "loc": "outside lower center"})
    save(fig, "fig_reliability")


def plot_latency(ecdf: dict) -> None:
    fig, ax = plt.subplots(figsize=(FIG_W, 5.4))
    for s, (lat, share, median) in ecdf.items():
        ax.plot(lat, share, **line_kw(s))
        ax.plot([median], [0.5], linestyle="none", **marker_kw(s, 9 if s == MAIN else 6.5))
    ax.axhline(0.5, **REF_LINE)
    ax.text(24, 0.52, "中位数", color=NOTE, fontsize=FONT_SIZE - 2, va="bottom")
    ax.set_xscale("log")
    ax.set_xlim(20, 30000)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.set_ylim(-0.03, 1.03)
    unit_ticks(ax.yaxis, 0.2, "%.1f")
    ax.set_xlabel("延迟（毫秒，对数刻度）")
    ax.set_ylabel("累计比例")
    grid(ax)
    bottom_legend(fig, line_handles(ecdf), ncol=3)
    save(fig, "fig_latency")


def plot_frontier(pts: list[dict]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(FIG_W, 5.2))
    panels = [(axes[0], "lat", "准确率与延迟", "延迟中位数（毫秒，对数刻度）"),
              (axes[1], "cost", "准确率与成本", "每千次判断成本（美元）")]
    for ax, key, title, xlabel in panels:
        fr = pareto(pts, key)
        ax.plot([p[key] for p in fr], [p["acc"] for p in fr], **REF_LINE)
        for p in pts:
            ax.plot([p[key]], [p["acc"]], linestyle="none", **marker_kw(p["s"], 12 if p["s"] == MAIN else 9))
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("合并准确率")
        ax.set_ylim(*data_ylim([p["acc"] for p in pts], step=0.05, top=0.80))
        unit_ticks(ax.yaxis, 0.05, "%.2f")
        grid(ax)
    axes[0].set_xscale("log")
    axes[0].set_xlim(30, 6000)
    axes[0].xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    axes[1].set_xlim(-0.008, 0.12)
    unit_ticks(axes[1].xaxis, 0.02, "%.2f")
    nli = next(p for p in pts if p["s"] == "nli")
    axes[1].annotate("本地运行，成本约为0", (0.0, nli["acc"]), xytext=(10, -2), textcoords="offset points",
                     color=NOTE, fontsize=FONT_SIZE - 2, va="center", ha="left")
    handles = marker_handles([p["s"] for p in pts]) + [Line2D([], [], label="帕累托前沿", **REF_LINE)]
    bottom_legend(fig, handles, ncol=4, handletextpad=0.4)
    save(fig, "fig_frontier")


def plot_failures(counts: dict) -> None:
    systems = list(counts)
    fig, ax = plt.subplots(figsize=(FIG_W, 5.0))
    widest = max(sum(c.values()) for c, _ in counts.values())
    pad = max(1.5, widest * 0.012)
    for i, s in enumerate(systems):
        y = len(systems) - 1 - i
        kinds, n = counts[s]
        left = 0
        for key, _ in FAILURE_KINDS:
            if kinds[key]:
                ax.barh(y, kinds[key], left=left, height=0.62, color=FAILURE_COLOR[key], edgecolor="white",
                        linewidth=0.8)
                if kinds[key] >= 0.05 * widest:
                    ax.text(left + kinds[key] / 2, y, str(kinds[key]), ha="center", va="center",
                            fontsize=FONT_SIZE - 1, color="white" if key != "other" else "black")
                left += kinds[key]
        ax.text(left + pad, y, f"{left} / {n}", va="center", ha="left", fontsize=FONT_SIZE - 1, color=NOTE)
    ax.set_yticks(range(len(systems)))
    ax.set_yticklabels([LABEL[s] for s in reversed(systems)])
    ax.set_xlim(0, max(1, widest) * 1.25)
    ax.set_xlabel("协议失败次数（右侧为失败数 / 请求数）")
    ax.set_ylabel("模型设置")
    grid(ax, "x")
    present = {k for c, _ in counts.values() for k, v in c.items() if v}
    handles = [Patch(facecolor=FAILURE_COLOR[k], edgecolor="white", label=l) for k, l in FAILURE_KINDS if k in present]
    bottom_legend(fig, handles, ncol=3, handlelength=2)
    save(fig, "fig_failures")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qwen-tag", default="eval", help="prediction tag for the Qwen runs, e.g. vllm")
    parser.add_argument("--data", default=None, help="folder written by merge.py (default data_all)")
    args = parser.parse_args()
    global PRED
    if args.data:
        paper_numbers_rev.set_data(Path(args.data).resolve())
    PRED = paper_numbers_rev.PRED
    apply_style()
    data = load_all(args.qwen_tag)

    pooled_cov = np.arange(5, 101) / 100
    ds_cov = np.arange(10, 101) / 100
    figures = [
        (plot_acc_dataset, (accuracy_table(data),)),
        (plot_fast_slow, (fast_slow_pairs(data),)),
        (plot_selective, (pooled_cov, selective_curves(data, pooled_cov))),
        (plot_selective_ds, (ds_cov, {ds: selective_curves(data, ds_cov, ds) for ds in DATASETS})),
        (plot_reliability, (reliability_bins(data),)),
        (plot_latency, (latency_ecdf(data),)),
        (plot_frontier, (frontier_points(data),)),
        (plot_failures, (failure_counts(data),)),
    ]
    for plot, inputs in figures:
        plot(*inputs)
        print("wrote", plot.__name__)


if __name__ == "__main__":
    main()
