#!/usr/bin/env python3
"""Figure 1 of the Chinese manuscript: the claim-verification step in a trustworthy generation system.

A service pipeline (question -> retrieval -> generation -> claim split) feeds the
verification gate, which triages each claim into release / human review / block
by its label distribution. The candidate verifier implementations sit inside the
gate; the five evaluation dimensions sit underneath. Writes
results/figs/fig_framework.png and .pdf in the shared figure style.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle as fs  # noqa: E402

fs.apply()
OUT = fs.FIGS / "fig_framework.png"

INK = "#222222"
MUTED = "#666666"
LINE = "#555555"
GATE = "#2a78d6"
GATE_FILL = "#eaf2fc"
RELEASE, REVIEW, BLOCK = "#1baf7a", "#eda100", "#eb6834"


def box(ax, x, y, w, h, text, edge=LINE, fill="white", size=7, weight="normal", lw=1.0, radius=0.08):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={radius}",
                                linewidth=lw, edgecolor=edge, facecolor=fill, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size,
            color=INK, weight=weight, zorder=3)


def arrow(ax, start, end, color=LINE, style="-|>", ls="-", rad=0.0, lw=1.1):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle=style, mutation_scale=10, color=color,
                                 linewidth=lw, linestyle=ls, shrinkA=0, shrinkB=0,
                                 connectionstyle=f"arc3,rad={rad}", zorder=1))


def main() -> None:
    fig, ax = plt.subplots(figsize=(fs.FULL_WIDTH, fs.FULL_WIDTH * 3.6 / 7.4))
    ax.set_xlim(0, 11.4)
    ax.set_ylim(0.5, 5.75)
    ax.axis("off")

    # service pipeline
    row_y, bh, bw = 4.3, 0.6, 1.1
    steps = [("用户问题", 0.1), ("检索证据", 1.45), ("生成答案", 2.8), ("声明拆分", 4.15)]
    for label, x in steps:
        box(ax, x, row_y, bw, bh, label)
    for (_, x1), (_, x2) in zip(steps, steps[1:]):
        arrow(ax, (x1 + bw, row_y + bh / 2), (x2, row_y + bh / 2))

    # the gate
    gx, gy, gw, gh = 5.6, 2.2, 2.15, 3.3
    ax.add_patch(FancyBboxPatch((gx, gy), gw, gh, boxstyle="round,pad=0,rounding_size=0.12",
                                linewidth=1.6, edgecolor=GATE, facecolor=GATE_FILL, zorder=1))
    ax.text(gx + gw / 2, gy + gh - 0.32, "声明验证", ha="center", va="center", fontsize=9,
            weight="bold", color=GATE)
    arrow(ax, (4.15 + bw, row_y + bh / 2), (gx, row_y + bh / 2))
    ax.text(gx + gw / 2, 4.88, "输出三类判断的概率", ha="center", va="center", fontsize=6.3, color=INK)
    ax.plot([gx + 0.25, gx + gw - 0.25], [4.4, 4.4], color=GATE, linewidth=0.5, alpha=0.5)
    ax.text(gx + gw / 2, 4.12, "候选验证模型", ha="center", va="center", fontsize=6.3, color=MUTED)
    for i, label in enumerate(["快思考模型", "大语言模型", "NLI模型"]):
        box(ax, gx + 0.22, 3.55 - i * 0.45, gw - 0.44, 0.32, label, edge=GATE, size=6.5, lw=0.8,
            radius=0.05)

    # triage outcomes
    ox, ow, oh = 8.85, 1.15, 0.55
    outcomes = [
        (4.6, "自动放行", RELEASE, "高置信支持"),
        (3.55, "人工复核", REVIEW, "置信度不足"),
        (2.5, "拦截", BLOCK, "高置信否定"),
    ]
    for y, label, color, cond in outcomes:
        box(ax, ox, y - oh / 2, ow, oh, label, edge=color, fill="white", size=7.5, lw=1.3)
        arrow(ax, (gx + gw, y), (ox, y), color=color)
        ax.text((gx + gw + ox) / 2, y + 0.08, cond, ha="center", va="bottom", fontsize=6, color=MUTED)

    # delivery and loops
    dx, dw = 10.3, 1.0
    box(ax, dx, 4.6 - oh / 2, dw, oh, "交付用户", size=7)
    arrow(ax, (ox + ow, 4.6), (dx, 4.6), color=RELEASE)
    ax.plot([ox + ow, dx + dw / 2], [3.55, 3.55], color=REVIEW, linewidth=1.1, zorder=1)
    arrow(ax, (dx + dw / 2, 3.55), (dx + dw / 2, 4.6 - oh / 2), color=REVIEW)
    ax.text(dx + dw / 2 + 0.08, 3.95, "核实后\n交付", ha="left", va="center", fontsize=6, color=MUTED)
    loop_y = 1.75
    bx = ox + ow / 2
    ax.plot([bx, bx, 3.35], [2.5 - oh / 2, loop_y, loop_y], color=BLOCK, linewidth=1.1, linestyle="--", zorder=1)
    arrow(ax, (3.35, loop_y), (3.35, row_y), color=BLOCK, ls="--")
    ax.text(4.4, loop_y + 0.08, "重新生成或拒答", ha="center", va="bottom", fontsize=6.3, color=MUTED)

    # evaluation dimensions
    ey, eh = 0.62, 0.72
    ax.add_patch(FancyBboxPatch((0.1, ey), 11.2, eh, boxstyle="round,pad=0,rounding_size=0.08",
                                linewidth=0.8, edgecolor="#bbbbbb", facecolor="#f6f6f4", zorder=1))
    ax.text(0.3, ey + eh / 2, "评价\n维度", ha="left", va="center", fontsize=7.5, weight="bold",
            color=INK, linespacing=1.3)
    dims = ["准确率", "校准", "可分流性", "延迟与成本", "协议可靠性"]
    for i, label in enumerate(dims):
        x = 1.6 + i * 1.94
        weight = "bold" if label == "可分流性" else "normal"
        edge = GATE if label == "可分流性" else "#999999"
        box(ax, x, ey + 0.14, 1.65, eh - 0.28, label, edge=edge, size=7, weight=weight, lw=1.0)

        fs.save(fig, OUT.stem)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
