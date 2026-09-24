#!/usr/bin/env python3
"""All analysis figures for the Chinese manuscript, in one shared style.

Figures (results/figs/*.png and *.pdf):
  fig_acc_dataset      per-dataset accuracy with Wilson 95% intervals
  fig_fast_slow        direct answer vs reasoning, per dataset (dumbbell)
  fig_selective        pooled selective-verification curves
  fig_selective_ds     selective-verification curves per dataset
  fig_reliability      reliability diagrams with confidence histograms
  fig_latency          empirical CDF of request latency
  fig_frontier         accuracy against latency and cost
  fig_failures         protocol failures by type

Qwen rows come from the tag given by --qwen-tag (default: vllm, the self-hosted run
with token probabilities); Qwen latency comes from its concurrency-1 sample. Systems
without label distributions are left out of the calibration and selective
figures automatically, so rerunning after the self-hosted Qwen run adds them.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import figstyle as fs
from prompts import parse_cot, parse_direct
from selective import acc_at, ranked_correct

PRED = fs.ROOT / "results" / "predictions"
SENS = fs.ROOT / "results" / "sensitivity"
JEV_IN, QIN, QOUT = 0.042e-6, 0.10e-6, 0.15e-6  # $/token list prices


# ---------------------------------------------------------------- data

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
    for system in fs.SYSTEMS:
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


def latency_rows(data, system: str) -> list[dict]:
    """Qwen latency from the concurrency-1 sample on the self-hosted server; the batch run queues."""
    if system.startswith("qwen"):
        files = sorted(SENS.glob(f"{system}__*__vllm-lat1.jsonl"))
        if files:
            return load_file(files[0])
    return data[system]


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


# ---------------------------------------------------------------- figures

def fig_acc_dataset(data) -> None:
    systems = [s for s in fs.SYSTEMS if s in data]
    fig, ax = plt.subplots(figsize=(fs.FULL_WIDTH, 3.7))
    offsets = np.linspace(-0.3, 0.3, len(systems))
    for i, ds in enumerate(fs.DATASETS):
        y0 = len(fs.DATASETS) - 1 - i
        if i % 2 == 0:
            ax.axhspan(y0 - 0.45, y0 + 0.45, color="#f5f5f3", zorder=0, linewidth=0)
        for off, s in zip(offsets, systems):
            acc, k, n = scorable_acc([r for r in data[s] if r["dataset"] == ds])
            lo, hi = wilson(k, n)
            ax.plot([lo, hi], [y0 + off] * 2, color=fs.COLOR[s], linewidth=1.1, zorder=2)
            ax.plot([acc], [y0 + off], linestyle="none", zorder=3, **fs.marker_kw(s, 4.6))
    ax.set_yticks(range(len(fs.DATASETS)))
    ax.set_yticklabels([fs.DATASET_LABEL[d] for d in reversed(fs.DATASETS)])
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(-0.55, len(fs.DATASETS) - 0.45)
    ax.set_xlim(0.4, 1.0)
    ax.set_xlabel("准确率（点为估计值，线段为Wilson 95%置信区间）")
    fs.xgrid(ax)
    ax.legend(handles=fs.legend_handles(systems), ncol=3, loc="lower center",
              bbox_to_anchor=(0.5, 1.0), columnspacing=1.4, handletextpad=0.3)
    fs.save(fig, "fig_acc_dataset")


def fig_fast_slow(data) -> None:
    pairs = [("qwen-direct-compact", "qwen-cot-compact", "零样本提示"),
             ("qwen-direct-expanded", "qwen-cot-expanded", "少样本提示")]
    pairs = [p for p in pairs if p[0] in data and p[1] in data]
    rows = fs.DATASETS + ["mean"]
    fig, axes = plt.subplots(1, len(pairs), figsize=(fs.FULL_WIDTH, 2.9), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (fast, slow, title) in zip(axes, pairs):
        fast_acc, slow_acc = [], []
        for ds in fs.DATASETS:
            fast_acc.append(scorable_acc([r for r in data[fast] if r["dataset"] == ds])[0])
            slow_acc.append(scorable_acc([r for r in data[slow] if r["dataset"] == ds])[0])
        fast_acc.append(float(np.mean(fast_acc)))
        slow_acc.append(float(np.mean(slow_acc)))
        for i, (a, b) in enumerate(zip(fast_acc, slow_acc)):
            y = len(rows) - 1 - i
            ax.plot([a, b], [y, y], color="#b5b5b5", linewidth=1.6, zorder=1, solid_capstyle="round")
            ax.plot([a], [y], linestyle="none", zorder=3, **fs.marker_kw(fast, 6))
            ax.plot([b], [y], linestyle="none", zorder=3, **fs.marker_kw(slow, 6))
            delta = (b - a) * 100
            ax.text(1.005, y, f"{delta:+.1f}".replace("-", "\u2212"), transform=ax.get_yaxis_transform(), va="center",
                    ha="left", fontsize=fs.FONT_SIZE - 0.5,
                    color=fs.INK if abs(delta) >= 1 else fs.MUTED,
                    weight="bold" if rows[i] == "mean" else "normal")
        ax.axhline(0.5, color=fs.RULE, linewidth=0.6)
        ax.set_title(title, loc="left")
        ax.set_xlim(0.45, 1.0)
        ax.set_xlabel("准确率")
        fs.xgrid(ax)
        ax.text(1.005, len(rows) - 0.4, "推理减直答\n（百分点）", transform=ax.get_yaxis_transform(),
                ha="left", va="bottom", fontsize=fs.FONT_SIZE - 1.5, color=fs.MUTED)
        ax.legend(handles=fs.legend_handles([fast, slow]), loc="upper left", handletextpad=0.3,
                  borderaxespad=0.2)
        ax.tick_params(axis="y", length=0)
        ax.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    axes[0].set_yticks(range(len(rows)))
    axes[0].set_yticklabels(["五数据集均值" if d == "mean" else fs.DATASET_LABEL[d] for d in reversed(rows)])
    axes[0].tick_params(axis="y", length=0)
    fig.tight_layout(w_pad=4.5)
    fs.save(fig, "fig_fast_slow")


def fig_selective(data) -> None:
    systems = [s for s in fs.SYSTEMS if s in data and has_dist(data[s])]
    xs = np.arange(5, 101) / 100
    fig, ax = plt.subplots(figsize=(fs.FULL_WIDTH * 0.72, 3.0))
    for s in systems:
        correct = ranked_correct(data[s])
        curve = [acc_at(correct, x) for x in xs]
        ax.plot(xs, curve, color=fs.COLOR[s], linewidth=1.6,
                linestyle="--" if s in fs.SLOW else "-", zorder=2)
        ax.plot(xs[5::20], [curve[i] for i in range(5, len(xs), 20)], linestyle="none", zorder=3,
                **fs.marker_kw(s, 5))
    for target in (0.90, 0.95):
        ax.axhline(target, color=fs.RULE, linewidth=0.7, linestyle=":", zorder=1)
        ax.text(0.99, target + 0.004, f"目标准确率 {target:.2f}", fontsize=fs.FONT_SIZE - 1.5,
                color=fs.MUTED, va="bottom", ha="right")
    ax.set_xlim(0.05, 1.0)
    ax.set_ylim(0.55, 1.0)
    ax.set_xlabel("自动判定比例（其余送人工复核）")
    ax.set_ylabel("自动判定部分的准确率")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    fs.ygrid(ax)
    ax.legend(handles=fs.legend_handles(systems), loc="lower left", handletextpad=0.3)
    fs.save(fig, "fig_selective")


def fig_selective_ds(data) -> None:
    systems = [s for s in fs.SYSTEMS if s in data and has_dist(data[s])]
    xs = np.arange(10, 101) / 100
    fig, axes = plt.subplots(2, 3, figsize=(fs.FULL_WIDTH, 3.9), sharex=True, sharey=True)
    for ax, ds in zip(axes.flat, fs.DATASETS):
        for s in systems:
            correct = ranked_correct([r for r in data[s] if r["dataset"] == ds])
            curve = [acc_at(correct, x) for x in xs]
            ax.plot(xs, curve, color=fs.COLOR[s], linewidth=1.4, linestyle="--" if s in fs.SLOW else "-")
            ax.plot([1.0], [curve[-1]], linestyle="none", **fs.marker_kw(s, 4.2))
        ax.axhline(0.90, color=fs.RULE, linewidth=0.7, linestyle=":")
        ax.set_title(fs.DATASET_LABEL[ds], loc="left")
        fs.ygrid(ax)
    legend_ax = axes.flat[-1]
    legend_ax.axis("off")
    handles = fs.legend_handles(systems) + [
        Line2D([], [], color=fs.RULE, linewidth=0.7, linestyle=":", label="目标准确率 0.90")]
    legend_ax.legend(handles=handles, loc="center", handletextpad=0.4, labelspacing=0.8)
    for ax in list(axes.flat)[:5]:
        ax.xaxis.set_tick_params(labelbottom=True)
    for ax in axes[1, :2]:
        ax.set_xlabel("自动判定比例")
    axes[0, 2].set_xlabel("自动判定比例")
    for ax in axes[:, 0]:
        ax.set_ylabel("准确率")
    axes[0, 0].set_ylim(0.4, 1.03)
    axes[0, 0].set_xlim(0.08, 1.04)
    axes[0, 0].xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    fig.tight_layout(h_pad=1.2, w_pad=1.0)
    fs.save(fig, "fig_selective_ds")


def reliability(rows: list[dict], bins: int = 10):
    pairs = sorted((max(r["distribution"].values()), int(r["prediction"] == r["gold"]))
                   for r in rows if r.get("distribution") and r.get("prediction"))
    chunks = np.array_split(np.array(pairs), bins)
    conf = [c[:, 0].mean() for c in chunks]
    acc = [c[:, 1].mean() for c in chunks]
    ece = sum(len(c) / len(pairs) * abs(c[:, 1].mean() - c[:, 0].mean()) for c in chunks)
    return conf, acc, ece, np.array([p[0] for p in pairs])


def fig_reliability(data) -> None:
    systems = [s for s in fs.SYSTEMS if s in data and has_dist(data[s])]
    ncol = 3
    nrow = math.ceil(len(systems) / ncol)
    fig = plt.figure(figsize=(fs.FULL_WIDTH, 2.85 * nrow))
    # a spacer row between panel rows keeps each title clear of the histogram above it
    ratios = [3.2, 1, 0.75] * nrow
    grid = fig.add_gridspec(3 * nrow - 1, ncol, height_ratios=ratios[:-1], hspace=0.12, wspace=0.12)
    for j, s in enumerate(systems):
        r, c = divmod(j, ncol)
        ax = fig.add_subplot(grid[3 * r, c])
        hax = fig.add_subplot(grid[3 * r + 1, c], sharex=ax)
        conf, acc, ece, allconf = reliability(data[s])
        ax.plot([0, 1], [0, 1], color=fs.RULE, linewidth=0.8, linestyle="--", zorder=1)
        ax.plot(conf, acc, color=fs.COLOR[s], linewidth=1.4, zorder=2)
        ax.plot(conf, acc, linestyle="none", zorder=3, **fs.marker_kw(s, 3.6))
        ax.set_title(fs.LABEL[s], loc="left")
        ax.text(0.04, 0.95, f"ECE = {ece:.3f}", transform=ax.transAxes, ha="left", va="top",
                fontsize=fs.FONT_SIZE - 0.5)
        ax.set_xlim(0.3, 1.02)
        ax.set_ylim(0.2, 1.0)
        fs.ygrid(ax)
        ax.tick_params(labelbottom=False)
        if c == 0:
            ax.set_ylabel("实际准确率")
        else:
            ax.tick_params(labelleft=False)
        hax.hist(allconf, bins=np.linspace(0.3, 1.0, 15), color=fs.COLOR[s], alpha=0.55,
                 edgecolor="white", linewidth=0.6)
        hax.set_yticks([])
        hax.spines["left"].set_visible(False)
        hax.set_xlabel("最高类别概率")
        if c == 0:
            hax.set_ylabel("案例数", rotation=0, ha="right", va="center")
    fig.legend(handles=[Line2D([], [], color=fs.RULE, linewidth=0.8, linestyle="--", label="完全校准")],
               loc="upper right", bbox_to_anchor=(0.99, 1.0))
    fs.save(fig, "fig_reliability")


def fig_latency(data) -> None:
    systems = [s for s in fs.SYSTEMS if s in data]
    fig, ax = plt.subplots(figsize=(fs.FULL_WIDTH * 0.72, 2.9))
    for s in systems:
        lat = np.sort([r["latency_ms"] for r in latency_rows(data, s) if r.get("latency_ms") and not r.get("error")])
        y = np.arange(1, len(lat) + 1) / len(lat)
        ax.plot(lat, y, color=fs.COLOR[s], linewidth=1.5, linestyle="--" if s in fs.SLOW else "-")
        med = float(np.median(lat))
        ax.plot([med], [0.5], linestyle="none", zorder=3, **fs.marker_kw(s, 5))
    ax.axhline(0.5, color=fs.RULE, linewidth=0.6, linestyle=":")
    ax.text(25, 0.515, "中位数", fontsize=fs.FONT_SIZE - 1.5, color=fs.MUTED)
    ax.set_xscale("log")
    ax.set_xlim(20, 30000)
    ax.set_ylim(0, 1.01)
    ax.set_xlabel("请求延迟（毫秒，对数刻度）")
    ax.set_ylabel("累计比例")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    fs.ygrid(ax)
    handles = fs.legend_handles(systems)
    for h, s in zip(handles, systems):
        h.set_linestyle("--" if s in fs.SLOW else "-")
        h.set_linewidth(1.5)
    ax.legend(handles=handles, loc="lower right", handlelength=2.4, handletextpad=0.4)
    fs.save(fig, "fig_latency")


def fig_frontier(data) -> None:
    pts = []
    for s in fs.SYSTEMS:
        if s not in data:
            continue
        rows = data[s]
        lat = [r["latency_ms"] for r in latency_rows(data, s) if r.get("latency_ms") and not r.get("error")]
        correct = sum(1 for r in rows if r.get("prediction") and r["prediction"] == r["gold"])
        cost = 0.0
        for r in rows:
            u = r.get("usage") or {}
            if u.get("input_tokens"):
                cost += u["input_tokens"] * JEV_IN
            elif u.get("prompt_tokens"):
                cost += u["prompt_tokens"] * QIN + u.get("completion_tokens", 0) * QOUT
        pts.append({"s": s, "lat": statistics.median(lat), "acc": correct / len(rows),
                    "cost": 0.0 if s == "nli" else cost / len(rows) * 1000})

    def frontier(key):
        out, best = [], -1.0
        for p in sorted(pts, key=lambda p: p[key]):
            if p["acc"] > best:
                out.append(p)
                best = p["acc"]
        return out

    fig, axes = plt.subplots(1, 2, figsize=(fs.FULL_WIDTH, 2.7), sharey=True)
    for ax, key, xlabel in [(axes[0], "lat", "请求延迟中位数（毫秒，对数刻度）"),
                            (axes[1], "cost", "每千次判断成本（美元）")]:
        fr = frontier(key)
        ax.plot([p[key] for p in fr], [p["acc"] for p in fr], color=fs.RULE, linewidth=1.0,
                linestyle="--", zorder=1)
        for p in pts:
            ax.plot([p[key]], [p["acc"]], linestyle="none", zorder=3, **fs.marker_kw(p["s"], 6.5))
        ax.set_xlabel(xlabel)
        fs.ygrid(ax)
    axes[0].set_xscale("log")
    axes[0].set_xlim(35, 5000)
    axes[0].xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    axes[1].set_xlim(-0.006, 0.115)
    axes[1].text(0.0, 0.628, "本地运行\n成本约为0", fontsize=fs.FONT_SIZE - 1.5, color=fs.MUTED,
                 ha="left", va="top")
    axes[0].set_ylabel("合并准确率（协议失败计为错误）")
    axes[0].set_ylim(0.6, 0.77)
    handles = fs.legend_handles([p["s"] for p in pts]) + [
        Line2D([], [], color=fs.RULE, linewidth=1.0, linestyle="--", label="帕累托前沿")]
    fig.legend(handles=handles, ncol=4, loc="lower center", bbox_to_anchor=(0.5, 0.98),
               columnspacing=1.2, handletextpad=0.3)
    fig.tight_layout(w_pad=2.0)
    fs.save(fig, "fig_frontier")


def widest_pad(rows) -> float:
    """Gap between a bar and its count label, scaled to the largest bar."""
    most = max(sum(1 for r in rs if failure_type(r)) for _, rs in rows)
    return max(1.5, most * 0.012)


def fig_failures(data) -> None:
    rows = [(fs.LABEL[s], data[s]) for s in fs.SYSTEMS if s in data]
    kinds = [("space", "判断空间外的答案", "#3b3b3b"),
             ("length", "推理被截断", "#8c8c8c"),
             ("other", "无法解析或请求失败", "#cfcfcf")]
    fig, ax = plt.subplots(figsize=(fs.FULL_WIDTH, 2.5))
    n_main = sum(1 for s in fs.SYSTEMS if s in data)
    widest = 0
    for i, (label, rs) in enumerate(rows):
        y = len(rows) - 1 - i
        left = 0
        counts = {k: 0 for k, *_ in kinds}
        for r in rs:
            kind = failure_type(r)
            if kind:
                counts[kind] += 1
        for key, _, color in kinds:
            if counts[key]:
                ax.barh(y, counts[key], left=left, height=0.62, color=color, edgecolor="white",
                        linewidth=0.8)
                if counts[key] >= 8:
                    ax.text(left + counts[key] / 2, y, str(counts[key]), ha="center", va="center",
                            fontsize=fs.FONT_SIZE - 1,
                            color="white" if key != "other" else fs.INK)
                left += counts[key]
        widest = max(widest, left)
        ax.text(left + widest_pad(rows), y, f"{left} / {len(rs)}", va="center", ha="left",
                fontsize=fs.FONT_SIZE - 1, color=fs.MUTED)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([label for label, _ in reversed(rows)])
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, max(1, widest) * 1.22)
    ax.set_xlabel("协议失败次数（右侧为失败数 / 请求数）")
    fs.xgrid(ax)
    present = {failure_type(r) for _, rs in rows for r in rs} - {None}
    ax.legend(handles=[Patch(facecolor=c, edgecolor="white", label=l) for k, l, c in kinds if k in present],
              ncol=3, loc="lower center", bbox_to_anchor=(0.45, 1.0), handletextpad=0.4)
    fs.save(fig, "fig_failures")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qwen-tag", default="eval", help="prediction tag for the Qwen runs, e.g. vllm")
    args = parser.parse_args()
    fs.apply()
    data = load_all(args.qwen_tag)
    for make in (fig_acc_dataset, fig_fast_slow, fig_selective, fig_selective_ds,
                 fig_reliability, fig_latency, fig_frontier, fig_failures):
        make(data)
        print("wrote", make.__name__)


if __name__ == "__main__":
    main()
