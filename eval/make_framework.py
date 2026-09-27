#!/usr/bin/env python3
"""Figure 1 of the Chinese manuscript: the fast/slow evaluation framework for claim verification.

The generation pipeline (left column) hands claims to the verification step. Inside
it the same claim and evidence reach a three-way label distribution along two
answering modes: fast thinking (no intermediate text; the decision-oriented model,
direct LLM answers, the NLI model) and slow thinking (reasoning text first). The
distribution triages each claim into release / human review / block. Underneath,
the five evaluation dimensions split into judgement quality and running cost, the
two sides of the fast/slow trade-off. Writes results/figs/fig_framework.png and .pdf.
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
FAST, FAST_FILL = "#2a78d6", "#eaf2fc"
SLOW, SLOW_FILL = "#7b52c7", "#f1ecfa"
PANEL_FILL = "#f8f8f6"
RELEASE, REVIEW, BLOCK = "#1baf7a", "#eda100", "#eb6834"


def box(ax, x, y, w, h, text, edge=LINE, fill="white", size=7, weight="normal", lw=1.0, radius=0.08,
        color=INK):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={radius}",
                                linewidth=lw, edgecolor=edge, facecolor=fill, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size,
            color=color, weight=weight, zorder=3, linespacing=1.25)


def arrow(ax, start, end, color=LINE, style="-|>", ls="-", lw=1.1):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle=style, mutation_scale=10, color=color,
                                 linewidth=lw, linestyle=ls, shrinkA=0, shrinkB=0, zorder=1))


def main() -> None:
    fig, ax = plt.subplots(figsize=(fs.FULL_WIDTH, fs.FULL_WIDTH * 7.6 / 11.3))
    ax.set_xlim(-0.1, 11.2)
    ax.set_ylim(0.2, 7.8)
    ax.axis("off")

    # generation pipeline, top to bottom
    px, pw, ph = 0.15, 1.4, 0.6
    steps = [("用户问题", 6.55), ("检索证据", 5.55), ("生成答案", 4.55), ("声明拆分", 3.55)]
    for label, y in steps:
        box(ax, px, y, pw, ph, label)
    for (_, y1), (_, y2) in zip(steps, steps[1:]):
        arrow(ax, (px + pw / 2, y1), (px + pw / 2, y2 + ph))

    # verification step: one input, two answering modes, one label distribution
    vx, vy, vw, vh = 2.05, 2.5, 7.0, 5.1
    ax.add_patch(FancyBboxPatch((vx, vy), vw, vh, boxstyle="round,pad=0,rounding_size=0.12",
                                linewidth=1.3, edgecolor=LINE, facecolor=PANEL_FILL, zorder=0))
    ax.text(vx + vw / 2, 7.25, "声明验证", ha="center", va="center", fontsize=9, weight="bold", color=INK)
    ax.text(vx + vw / 2, 6.88, "同一声明与证据，同一判断空间，两种作答方式", ha="center", va="center",
            fontsize=6.5, color=MUTED)

    lo, hi = 3.2, 6.35
    ix, iw, ox_in, ow_in = 2.3, 0.95, 7.75, 1.1
    box(ax, ix, lo, iw, hi - lo, "声明\n＋\n证据", size=7.5)
    box(ax, ox_in, lo, ow_in, hi - lo, "三类判断\n的概率\n\n支持\n反驳\n证据不足", size=7)
    arrow(ax, (px + pw, 3.85), (ix, 3.85))

    # fast lane
    fy = 5.55
    arrow(ax, (ix + iw, fy), (ox_in, fy), color=FAST, lw=1.6)
    ax.text((ix + iw + ox_in) / 2, fy + 0.2, "快思考：不生成中间文本，直接给出判断", ha="center", va="bottom",
            fontsize=7, color=FAST, weight="bold")
    cx = ix + iw + 0.25
    for label, cw in (("决策化模型JEV", 1.6), ("Qwen直答", 1.1), ("NLI模型", 1.1)):
        box(ax, cx, 4.72, cw, 0.42, label, edge=FAST, fill=FAST_FILL, size=6.5, lw=0.8, radius=0.05)
        cx += cw + 0.12

    # slow lane
    sy = 3.7
    rx, rw = 4.85, 1.3  # centred under the lanes
    arrow(ax, (ix + iw, sy), (rx, sy), color=SLOW, lw=1.6)
    box(ax, rx, sy - 0.26, rw, 0.52, "推理文本", edge=SLOW, fill="white", size=7, lw=1.2)
    arrow(ax, (rx + rw, sy), (ox_in, sy), color=SLOW, lw=1.6)
    ax.text((ix + iw + ox_in) / 2, sy + 0.4, "慢思考：先写推理，末行给出判断", ha="center", va="bottom",
            fontsize=7, color=SLOW, weight="bold")
    box(ax, rx, 2.72, rw, 0.42, "Qwen推理", edge=SLOW, fill=SLOW_FILL, size=6.5, lw=0.8, radius=0.05)

    # triage by the label distribution
    ox, ow, oh = 9.55, 1.45, 0.72
    outcomes = [(5.75, "自动放行", "高置信支持", RELEASE),
                (4.6, "人工复核", "置信度不足", REVIEW),
                (3.45, "拦截", "高置信否定", BLOCK)]
    for y, label, cond, color in outcomes:
        ax.add_patch(FancyBboxPatch((ox, y - oh / 2), ow, oh, boxstyle="round,pad=0,rounding_size=0.08",
                                    linewidth=1.3, edgecolor=color, facecolor="white", zorder=2))
        ax.text(ox + ow / 2, y + 0.11, label, ha="center", va="center", fontsize=7.5, color=INK, zorder=3)
        ax.text(ox + ow / 2, y - 0.19, cond, ha="center", va="center", fontsize=5.8, color=MUTED, zorder=3)
        arrow(ax, (ox_in + ow_in, y), (ox, y), color=color)

    # a blocked claim goes back to generation
    ly = 2.25
    bx = ox + ow / 2
    ax.plot([bx, bx, 0.0, 0.0], [3.45 - oh / 2, ly, ly, 4.85], color=BLOCK, linewidth=1.0,
            linestyle="--", zorder=1)
    arrow(ax, (0.0, 4.85), (px, 4.85), color=BLOCK, ls="--", lw=1.0)
    ax.text(5.5, ly - 0.06, "重新生成或拒答", ha="center", va="top", fontsize=6.3, color=MUTED)

    # evaluation dimensions: judgement quality against running cost
    gy, gh = 0.35, 1.45
    groups = [(0.15, 5.4, "判断质量", ["准确率", "校准", "可分流性"], 1.55),
              (6.65, 4.35, "运行代价", ["延迟与成本", "协议可靠性"], 1.95)]
    for x, w, title, dims, cw in groups:
        ax.add_patch(FancyBboxPatch((x, gy), w, gh, boxstyle="round,pad=0,rounding_size=0.08",
                                    linewidth=0.8, edgecolor="#bbbbbb", facecolor="#f3f3f1", zorder=0))
        ax.text(x + w / 2, gy + gh - 0.27, title, ha="center", va="center", fontsize=7.5,
                weight="bold", color=INK)
        gap = (w - len(dims) * cw) / (len(dims) + 1)
        for i, label in enumerate(dims):
            box(ax, x + gap + i * (cw + gap), gy + 0.17, cw, 0.58, label, edge="#999999", size=7)
    arrow(ax, (5.62, 0.8), (6.63, 0.8), style="<|-|>", color=INK, lw=1.0)
    ax.text(6.12, 1.02, "权衡", ha="center", va="bottom", fontsize=7.5, weight="bold", color=INK)

    fs.save(fig, OUT.stem)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
