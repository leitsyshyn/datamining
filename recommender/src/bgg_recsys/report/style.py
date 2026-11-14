"""The chart theme, a few drawing primitives, and where figures are saved."""

from __future__ import annotations

from pathlib import Path

import matplotlib.patheffects as pe
import numpy as np
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SEQ = LinearSegmentedColormap.from_list("seq", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
DIV = LinearSegmentedColormap.from_list("div", ["#1c5cab", "#86b6ef", "#f0efec", "#f2a3a2", "#b53434"])
GROUP_COLORS = {"non-personalised": SERIES[1], "rating-prediction CF": SERIES[0], "top-N CF": SERIES[2]}

_figure_dir = Path("figures")


def use(figure_dir: Path) -> None:
    """Apply the theme to matplotlib and seaborn, and set where `save` writes figures."""
    global _figure_dir
    _figure_dir = figure_dir
    figure_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", palette=SERIES, rc={
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK2, "axes.titlecolor": INK, "text.color": INK,
        "xtick.color": INK2, "ytick.color": INK2, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True, "lines.linewidth": 2,
        "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold", "legend.frameon": False, "figure.dpi": 110,
    })


def save(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(_figure_dir / f"{name}.png", dpi=140, bbox_inches="tight")


def percent_axis(axis) -> None:
    axis.set_major_formatter(lambda v, _: f"{v:.0%}")


def label_points(ax, xs, ys, labels, min_gap_px: float = 13, dx_pt: float = 5) -> None:
    """Direct labels next to points, nudged vertically so nearby labels don't collide."""
    fig = ax.figure
    fig.canvas.draw()
    pts = ax.transData.transform(np.column_stack([xs, ys]))
    placed: list[tuple[float, float]] = []
    for k in np.argsort(pts[:, 1]):
        x, y = pts[k]
        for px, py in placed:
            if abs(px - x) < 150 and abs(y - py) < min_gap_px:
                y = py + min_gap_px
        placed.append((x, y))
        ax.annotate(labels[k], (xs[k], ys[k]), xytext=(dx_pt, (y - pts[k][1]) * 72 / fig.dpi), textcoords="offset points",
                    fontsize=8.5, color=INK2, va="center", path_effects=[pe.withStroke(linewidth=3, foreground=SURFACE)])


def group_legend(ax, marker: bool = False, **kw) -> None:
    """Legend for the three model groups of the ranking charts."""
    handles = [Line2D([], [], marker="o", linestyle="", markersize=7, color=c) if marker else Rectangle((0, 0), 1, 1, color=c)
               for c in GROUP_COLORS.values()]
    ax.legend(handles, list(GROUP_COLORS), **kw)
