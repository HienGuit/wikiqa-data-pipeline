"""Shared plotting helpers for EDA notebooks."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

PALETTE = "tab10"
DPI = 300
FIGSIZE_DEFAULT = (10, 5)
FIGSIZE_WIDE = (14, 5)
COLOR_REASONING = {
    "extraction": "#2B6CB0",
    "bridge": "#D69E2E",
    "multi-sentence": "#2F855A",
}
COLOR_QUALITY = {
    "weak": "#C53030",
    "usable": "#D69E2E",
    "strong": "#2F855A",
}
COLOR_DIFFICULTY = {
    "easy": "#2C7FB8",
    "medium": "#F28E2B",
    "hard": "#7A5195",
}


def apply_publication_style() -> None:
    """Apply a clean, publication-oriented Matplotlib style."""

    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "axes.edgecolor": "#C9D1D9",
            "axes.linewidth": 0.8,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 9,
            "figure.titlesize": 15,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def save_fig(fig, fig_dir: Path, filename: str, *, close: bool = False, show: bool = False) -> None:
    """Save a figure as PNG and PDF, creating the target directory if needed."""

    fig_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()

    png_path = fig_dir / filename
    fig.savefig(png_path, dpi=DPI, bbox_inches="tight")

    if filename.endswith(".png"):
        pdf_path = fig_dir / filename.replace(".png", ".pdf")
        fig.savefig(pdf_path, bbox_inches="tight")

    if show:
        plt.show()
    if close:
        plt.close(fig)
