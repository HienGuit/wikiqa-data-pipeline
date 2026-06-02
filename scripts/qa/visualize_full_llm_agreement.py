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
    kappa = [report["dimensions"][key]["cohen_kappa"] for key in DIMENSION_ORDER]
    rows = [report["dimensions"][key]["rows"] for key in DIMENSION_ORDER]
    colors = [BAR_COLORS[key] for key in DIMENSION_ORDER]

    y = np.arange(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.8), constrained_layout=True)

    axes[0].barh(y, agreement, color=colors, edgecolor="white", linewidth=1.0)
    axes[0].set_title("Observed Agreement")
    axes[0].set_xlabel("Percent")
    axes[0].set_xlim(0, 100)
    axes[0].set_yticks(y, labels=labels)
    axes[0].grid(axis="x", alpha=0.25)
    axes[0].set_axisbelow(True)
    for idx, value in enumerate(agreement):
        axes[0].text(min(value + 1.2, 98.5), idx, f"{value:.1f}%", va="center", fontsize=9, fontweight="bold")

    axes[1].barh(y, kappa, color=colors, edgecolor="white", linewidth=1.0)
    axes[1].set_title("Cohen's Kappa")
    axes[1].set_xlabel("Kappa")
    axes[1].set_xlim(0, max(0.4, max(kappa) + 0.08))
    axes[1].set_yticks(y, labels=[""] * len(labels))
    axes[1].grid(axis="x", alpha=0.25)
    axes[1].set_axisbelow(True)
    for idx, value in enumerate(kappa):
        axes[1].text(value + 0.01, idx, f"{value:.3f}", va="center", fontsize=9, fontweight="bold")

    fig.suptitle("Full-Population Agreement Between Gemini and DeepSeek", fontsize=14, fontweight="bold")
    fig.text(
        0.5,
        0.02,
        "Agreement is computed on all shared judged rows; inferential validity is restricted to shared multi-sentence samples.",
        ha="center",
        fontsize=9,
        color="#444444",
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
