"""Shared figure style for the Chinese manuscript.

Chinese journals set figure text in a sans face: 黑体 for Chinese, Arial or
Helvetica for Latin letters and numbers. Here Arial is primary and Noto Sans CJK
SC (思源黑体) fills in the Chinese glyphs. One color and one marker per system,
used in every figure: color carries identity, a filled marker means fast
thinking and a hollow marker means slow thinking.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
FIGS = ROOT / "results" / "figs"

FULL_WIDTH = 6.1   # inches, about 15.5 cm of text width
FONT_SIZE = 8

INK = "#1f1f1f"
MUTED = "#6b6b6b"
GRID = "#e4e4e4"
RULE = "#9a9a9a"

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
# fixed categorical slots, validated for CVD separation (see dataviz palette check)
COLOR = {
    "jev": "#2a78d6",
    "nli": "#eb6834",
    "qwen-direct-compact": "#1baf7a",
    "qwen-direct-expanded": "#eda100",
    "qwen-cot-compact": "#e87ba4",
    "qwen-cot-expanded": "#008300",
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

DATASETS = ["fever", "scifact", "hover", "vitaminc", "climate_fever"]
DATASET_LABEL = {
    "fever": "FEVER",
    "scifact": "SciFact",
    "hover": "HoVer",
    "vitaminc": "VitaminC",
    "climate_fever": "Climate-FEVER",
}


CJK_FONT_FILES = [
    Path.home() / "Library" / "Fonts" / "NotoSansCJKsc-Regular.otf",
    Path.home() / "Library" / "Fonts" / "NotoSansCJKsc-Bold.otf",
]


def apply() -> None:
    from matplotlib import font_manager
    for path in CJK_FONT_FILES:  # user-installed fonts are missing from matplotlib's cache
        if path.exists():
            font_manager.fontManager.addfont(str(path))
    plt.rcParams.update({
        # an explicit family list enables per-glyph fallback: Latin from Arial, Chinese from 思源黑体
        "font.family": ["Arial", "Noto Sans CJK SC"],
        "font.size": FONT_SIZE,
        "axes.titlesize": FONT_SIZE + 0.5,
        "axes.titleweight": "bold",
        "axes.labelsize": FONT_SIZE,
        "axes.labelcolor": INK,
        "axes.edgecolor": RULE,
        "axes.linewidth": 0.6,
        "axes.unicode_minus": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.labelsize": FONT_SIZE - 0.5,
        "ytick.labelsize": FONT_SIZE - 0.5,
        "xtick.color": INK,
        "ytick.color": INK,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "legend.fontsize": FONT_SIZE - 0.5,
        "legend.frameon": False,
        "text.color": INK,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
        "pdf.fonttype": 42,
    })


def ygrid(ax) -> None:
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def xgrid(ax) -> None:
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def marker_kw(system: str, size: float = 5.5) -> dict:
    """Filled marker for fast thinking, hollow for slow thinking."""
    color = COLOR[system]
    return {
        "marker": MARKER[system],
        "markersize": size,
        "markeredgecolor": color,
        "markeredgewidth": 1.1,
        "markerfacecolor": "white" if system in SLOW else color,
        "color": color,
    }


def legend_handles(systems) -> list:
    from matplotlib.lines import Line2D
    return [Line2D([], [], linestyle="none", label=LABEL[s], **marker_kw(s)) for s in systems]


def save(fig, name: str) -> Path:
    FIGS.mkdir(parents=True, exist_ok=True)
    path = FIGS / f"{name}.png"
    fig.savefig(path)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)
    return path
