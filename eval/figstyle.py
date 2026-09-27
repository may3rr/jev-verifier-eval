"""Shared figure settings for the Chinese manuscript.

Chinese text in every figure is set in 宋体 (Songti SC). The framework diagram
(make_framework.py) pairs it with Times New Roman at 8pt through apply(); the
data figures (make_figs.py) keep their own style block and only borrow
register_fonts() and save paths from here.
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
RULE = "#9a9a9a"

SONG = "Songti SC"
SONG_TTC = Path("/System/Library/Fonts/Supplemental/Songti.ttc")
TIMES_FILES = [
    Path("/System/Library/Fonts/Supplemental/Times New Roman.ttf"),
    Path("/System/Library/Fonts/Supplemental/Times New Roman Bold.ttf"),
    Path("/System/Library/Fonts/Supplemental/Times New Roman Italic.ttf"),
    Path("/System/Library/Fonts/Supplemental/Times New Roman Bold Italic.ttf"),
]


def register_fonts() -> None:
    """Make 宋体 regular and bold available to matplotlib.

    matplotlib reads only the first face of a .ttc, which in Songti.ttc is the
    Black weight, so the regular and bold faces are extracted once into the
    matplotlib cache directory and registered from there.
    """
    from fontTools.ttLib import TTCollection
    from matplotlib import font_manager
    cache = Path(matplotlib.get_cachedir())
    for weight in ("Regular", "Bold"):
        path = cache / f"SongtiSC-{weight}.ttf"
        if not path.exists():
            for font in TTCollection(str(SONG_TTC)).fonts:
                if font["name"].getDebugName(4) == f"{SONG} {weight}":
                    font.save(str(path))
                    break
        font_manager.fontManager.addfont(str(path))
    for path in TIMES_FILES:  # system Supplemental fonts are missing from matplotlib's cache
        if path.exists():
            font_manager.fontManager.addfont(str(path))


def apply() -> None:
    """Style for the framework diagram."""
    register_fonts()
    plt.rcParams.update({
        # an explicit family list enables per-glyph fallback: Latin from Times New Roman, Chinese from 宋体
        "font.family": ["Times New Roman", SONG],
        "font.size": FONT_SIZE,
        "axes.labelcolor": INK,
        "axes.edgecolor": RULE,
        "axes.linewidth": 0.6,
        "axes.unicode_minus": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": INK,
        "ytick.color": INK,
        "legend.frameon": False,
        "text.color": INK,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
        "pdf.fonttype": 42,
    })


def save(fig, name: str) -> Path:
    FIGS.mkdir(parents=True, exist_ok=True)
    path = FIGS / f"{name}.png"
    fig.savefig(path)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)
    return path
