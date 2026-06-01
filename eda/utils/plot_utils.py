"""Shared plotting helpers for EDA notebooks."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


PALETTE = "tab10"
DPI = 150
FIGSIZE_DEFAULT = (10, 5)
FIGSIZE_WIDE = (14, 5)


def save_fig(fig, fig_dir: Path, filename: str, *, close: bool = False) -> None:
    """Save a figure, creating the target directory if needed."""

    fig_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(fig_dir / filename, dpi=DPI, bbox_inches="tight")
    plt.show()
    if close:
        plt.close(fig)
