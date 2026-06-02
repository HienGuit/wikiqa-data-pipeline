"""Create a compact visualization for full-population Gemini vs DeepSeek agreement."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import QA_REPORTS_DIR, ensure_dirs  # noqa: E402

REPORT_JSON = QA_REPORTS_DIR / "full_llm_agreement_report.json"
FIGURE_PATH = QA_REPORTS_DIR / "full_llm_agreement_overview.png"
VIS_REPORT_PATH = QA_REPORTS_DIR / "full_llm_agreement_visualization_report.json"

DIMENSION_ORDER = ["quality_band", "difficulty_band", "inferential_validity_band"]
DISPLAY_LABELS = {
    "quality_band": "Quality",
    "difficulty_band": "Difficulty",
    "inferential_validity_band": "Inferential Validity",
}
BAR_COLORS = {
    "quality_band": "#4C78A8",
    "difficulty_band": "#59A14F",
    "inferential_validity_band": "#F28E2B",
}


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 10,
            "legend.fontsize": 9,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def main() -> None:
    ensure_dirs()
    configure_style()
    report = json.loads(REPORT_JSON.read_text(encoding="utf-8"))

    labels = [DISPLAY_LABELS[key] for key in DIMENSION_ORDER]
    agreement = [report["dimensions"][key]["observed_agreement_percent"] for key in DIMENSION_ORDER]
    expected = [report["dimensions"][key]["expected_agreement_percent"] for key in DIMENSION_ORDER]
    kappa = [report["dimensions"][key]["cohen_kappa"] for key in DIMENSION_ORDER]
    rows = [report["dimensions"][key]["rows"] for key in DIMENSION_ORDER]
    colors = [BAR_COLORS[key] for key in DIMENSION_ORDER]
    display_labels = [f"{DISPLAY_LABELS[key]} (n={rows[idx]:,})" for idx, key in enumerate(DIMENSION_ORDER)]

    y = np.arange(len(labels))
    fig = plt.figure(figsize=(12.6, 5.8), constrained_layout=True)
    grid = fig.add_gridspec(3, 2, height_ratios=[1.2, 12.5, 2.2], width_ratios=[1.15, 0.85])
    title_ax = fig.add_subplot(grid[0, :])
    ax_left = fig.add_subplot(grid[1, 0])
    ax_right = fig.add_subplot(grid[1, 1])
    caption_ax = fig.add_subplot(grid[2, :])

    title_ax.axis("off")
    caption_ax.axis("off")

    title_ax.text(
        0.5,
        0.56,
        "Full-Population Agreement Between Gemini and DeepSeek",
        ha="center",
        va="center",
        fontsize=14,
        fontweight="bold",
        color="#111111",
    )

    # Panel A: observed vs expected agreement
    ax_left.set_title("(a) Observed vs Expected Agreement", pad=10)
    ax_left.set_xlabel("Agreement (%)")
    ax_left.set_xlim(0, 100)
    ax_left.set_yticks(y, labels=display_labels)
    ax_left.grid(axis="x", alpha=0.25)
    ax_left.set_axisbelow(True)
    for idx, (obs, exp, color) in enumerate(zip(agreement, expected, colors)):
        ax_left.hlines(idx, exp, obs, color=color, linewidth=4, alpha=0.9)
        ax_left.scatter(exp, idx, s=64, color="white", edgecolor=color, linewidth=1.6, zorder=3)
        ax_left.scatter(obs, idx, s=78, color=color, edgecolor="white", linewidth=0.9, zorder=4)
        ax_left.text(
            min(obs + 1.4, 98.8),
            idx,
            f"{obs:.1f}%",
            va="center",
            fontsize=9,
            fontweight="bold",
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.2, "alpha": 0.92},
        )
        ax_left.text(
            max(exp - 1.4, 1.8),
            idx - 0.18,
            f"E={exp:.1f}%",
            ha="right",
            va="center",
            fontsize=8,
            color="#555555",
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.15, "alpha": 0.85},
        )

    # Panel B: kappa
    ax_right.set_title("(b) Cohen's Kappa", pad=10)
    ax_right.set_xlabel("Kappa")
    ax_right.set_xlim(0, 0.42)
    ax_right.set_yticks(y, labels=[""] * len(display_labels))
    ax_right.grid(axis="x", alpha=0.25)
    ax_right.set_axisbelow(True)
    ax_right.axvspan(0.00, 0.20, color="#f3d7cf", alpha=0.5, zorder=0)
    ax_right.axvspan(0.20, 0.40, color="#f6e7bb", alpha=0.45, zorder=0)
    ax_right.axvspan(0.40, 0.60, color="#dcecc8", alpha=0.40, zorder=0)
    for idx, (value, color) in enumerate(zip(kappa, colors)):
        ax_right.hlines(idx, 0, value, color=color, linewidth=4, alpha=0.9)
        ax_right.scatter(value, idx, s=80, color=color, edgecolor="white", linewidth=0.9, zorder=3)
        ax_right.text(
            min(value + 0.014, 0.395),
            idx,
            f"{value:.3f}",
            va="center",
            fontsize=9,
            fontweight="bold",
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.2, "alpha": 0.92},
        )

    caption_ax.text(
        0.5,
        0.56,
        "Observed agreement is reported alongside expected agreement because class imbalance inflates raw agreement, especially for difficulty. "
        "Inferential validity is computed only on shared multi-sentence samples.",
        ha="center",
        va="center",
        fontsize=9,
        color="#444444",
        wrap=True,
    )
    fig.savefig(FIGURE_PATH, dpi=260, bbox_inches="tight")
    plt.close(fig)

    vis_report = {
        "source_report": str(REPORT_JSON),
        "figure": str(FIGURE_PATH),
        "dimension_order": DIMENSION_ORDER,
        "rows": {DISPLAY_LABELS[key]: rows[idx] for idx, key in enumerate(DIMENSION_ORDER)},
    }
    VIS_REPORT_PATH.write_text(json.dumps(vis_report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(vis_report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
