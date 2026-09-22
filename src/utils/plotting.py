"""Figure helpers.

Every figure in ``reports/figures`` is produced through ``save_figure`` so that
all of them share one style, one size convention and one caption mechanism.
The caption is the research question the figure answers (spec §57): a figure
without a question is not included.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: figures are written to disk, never shown
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# Colour-blind-safe qualitative palette (Okabe-Ito), extended for 15 assets.
PALETTE = [
    "#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00",
    "#56B4E9", "#7F7F7F", "#332288", "#88CCEE", "#44AA99",
    "#117733", "#999933", "#DDCC77", "#CC6677", "#882255",
]

ASSET_CLASS_COLOURS = {
    "equity": "#0072B2",
    "rates": "#009E73",
    "fixed_income": "#44AA99",
    "credit": "#E69F00",
    "commodity": "#D55E00",
    "real_estate": "#CC79A7",
}

_STYLE = {
    "figure.figsize": (11.0, 6.0),
    "figure.dpi": 110,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "font.size": 10.5,
    "axes.titlesize": 12.5,
    "axes.titleweight": "bold",
    "axes.labelsize": 10.5,
    "axes.grid": True,
    "axes.axisbelow": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "grid.alpha": 0.25,
    "grid.linestyle": "-",
    "grid.linewidth": 0.6,
    "legend.frameon": False,
    "legend.fontsize": 9.0,
    "lines.linewidth": 1.4,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
}


def apply_style() -> None:
    plt.rcParams.update(_STYLE)
    plt.rcParams["axes.prop_cycle"] = plt.cycler(color=PALETTE)


apply_style()

FIGURE_INDEX: list[dict[str, str]] = []


def new_axes(nrows: int = 1, ncols: int = 1, figsize=None, **kwargs):
    apply_style()
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize or _STYLE["figure.figsize"], **kwargs)
    return fig, axes


def save_figure(
    fig,
    path: str | Path,
    question: str = "",
    number: int | None = None,
    close: bool = True,
) -> Path:
    """Write a figure and record the research question it answers."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    FIGURE_INDEX.append(
        {"number": "" if number is None else str(number), "file": path.name, "question": question}
    )
    if question:
        caption = path.with_suffix(".txt")
        label = f"Figure {number}. " if number is not None else ""
        caption.write_text(f"{label}{question}\n", encoding="utf-8")
    if close:
        plt.close(fig)
    return path


def write_figure_index(path: str | Path) -> Path:
    """Markdown index of every figure written in this process."""
    path = Path(path)
    lines = ["| # | File | Research question |", "|---|------|-------------------|"]
    for row in FIGURE_INDEX:
        lines.append(f"| {row['number']} | `{row['file']}` | {row['question']} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


# --- reusable chart primitives ------------------------------------------
def plot_lines(ax, frame: pd.DataFrame, title="", ylabel="", legend_cols=2, logy=False):
    for i, column in enumerate(frame.columns):
        ax.plot(frame.index, frame[column].to_numpy(), label=str(column), color=PALETTE[i % len(PALETTE)])
    if logy:
        ax.set_yscale("log")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    if len(frame.columns) > 1:
        ax.legend(ncol=legend_cols, loc="upper left")
    return ax


def plot_drawdown(ax, curve: pd.Series, title="Drawdown", color="#B03A2E"):
    running_max = curve.cummax()
    dd = curve / running_max - 1.0
    ax.fill_between(dd.index, dd.to_numpy(), 0.0, color=color, alpha=0.35)
    ax.plot(dd.index, dd.to_numpy(), color=color, linewidth=1.0)
    ax.set_title(title)
    ax.set_ylabel("Drawdown")
    return ax


def plot_heatmap(ax, matrix: pd.DataFrame, title="", cmap="RdBu_r", vmin=-1.0, vmax=1.0, fmt="{:.2f}", annotate=True):
    data = matrix.to_numpy(dtype=float)
    image = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(matrix.shape[1]), [str(c) for c in matrix.columns], rotation=90)
    ax.set_yticks(range(matrix.shape[0]), [str(i) for i in matrix.index])
    if annotate and data.size <= 400:
        mid = (vmin + vmax) / 2.0
        span = max(vmax - vmin, 1e-12)
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                value = data[i, j]
                if not np.isfinite(value):
                    continue
                shade = "white" if abs(value - mid) > 0.38 * span else "black"
                ax.text(j, i, fmt.format(value), ha="center", va="center", fontsize=7.0, color=shade)
    ax.set_title(title)
    ax.grid(False)
    return image


def bar_with_values(ax, series: pd.Series, title="", ylabel="", fmt="{:.2f}", color=None, rotation=45):
    values = series.to_numpy(dtype=float)
    colors = color or [PALETTE[i % len(PALETTE)] for i in range(len(series))]
    ax.bar(range(len(series)), values, color=colors)
    ax.set_xticks(range(len(series)), [str(i) for i in series.index], rotation=rotation, ha="right" if rotation else "center")
    span = np.nanmax(np.abs(values)) if len(values) else 1.0
    for i, value in enumerate(values):
        if not np.isfinite(value):
            continue
        offset = 0.02 * span * (1 if value >= 0 else -1)
        ax.text(i, value + offset, fmt.format(value), ha="center",
                va="bottom" if value >= 0 else "top", fontsize=8.0)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    return ax
