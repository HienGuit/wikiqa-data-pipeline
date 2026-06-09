"""Shared agreement computation and visualization utilities for QA human verification."""

from __future__ import annotations

import math
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from src.qa.human_verification import load_json, write_json, write_text

ROOT = Path(__file__).resolve().parents[2]

PAIR_SPECS = [
    ("annotator1", "annotator2"),
    ("annotator1", "gemini_annotation"),
    ("annotator2", "gemini_annotation"),
]

ACTOR_ORDER = ["annotator1", "annotator2", "gemini_annotation"]
ACTOR_LABELS = {
    "annotator1": "A1",
    "annotator2": "A2",
    "gemini_annotation": "Gemini",
}
DIMENSION_TITLES = {
    "quality_band": "Task 1: Quality Band",
    "difficulty_band": "Task 1: Difficulty Band",
    "inferential_validity_band": "Task 2: Inferential Validity Band",
}
HEATMAP_CMAP = LinearSegmentedColormap.from_list(
    "kappa_academic",
    [
        (0.00, "#7f1d3a"),
        (0.20, "#c74f4f"),
        (0.40, "#e7b04b"),
        (0.60, "#b8cf6a"),
        (0.80, "#4d9964"),
        (1.00, "#1f5d4e"),
    ],
)


def _repo_rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path)


def get_task1_quality(source: str, row: Dict[str, Any]) -> str:
    if source.startswith("annotator"):
        return str(row[source].get("human_quality_band", "")).strip().lower()
    return str(row[source].get("quality_band_ref", "")).strip().lower()


def get_task1_difficulty(source: str, row: Dict[str, Any]) -> str:
    if source.startswith("annotator"):
        return str(row[source].get("human_difficulty_band", "")).strip().lower()
    return str(row[source].get("difficulty_band_ref", "")).strip().lower()


def get_task2_inferential(source: str, row: Dict[str, Any]) -> str:
    if source.startswith("annotator"):
        return str(row[source].get("human_inferential_validity_band", "")).strip().lower()
    return str(row[source].get("inferential_validity_band_ref", "")).strip().lower()


DIMENSIONS = [
    {
        "task_file": "task1.json",
        "dimension_id": "quality_band",
        "task_name": "task1_quality_difficulty",
        "getter": get_task1_quality,
        "labels": ["weak", "usable", "strong"],
    },
    {
        "task_file": "task1.json",
        "dimension_id": "difficulty_band",
        "task_name": "task1_quality_difficulty",
        "getter": get_task1_difficulty,
        "labels": ["easy", "medium", "hard"],
    },
    {
        "task_file": "task2.json",
        "dimension_id": "inferential_validity_band",
        "task_name": "task2_inferential_validity",
        "getter": get_task2_inferential,
        "labels": ["weak", "usable", "strong"],
    },
]


def interpret_kappa(value: float) -> str:
    if value < 0:
        return "less than chance"
    if value <= 0.20:
        return "slight agreement"
    if value <= 0.40:
        return "fair agreement"
    if value <= 0.60:
        return "moderate agreement"
    if value <= 0.80:
        return "substantial agreement"
    return "almost perfect agreement"


def interpret_percent(percent: float) -> str:
    if percent < 50:
        return "low agreement"
    if percent < 65:
        return "moderate agreement"
    if percent < 80:
        return "strong agreement"
    if percent < 90:
        return "very strong agreement"
    return "near-complete agreement"


def compute_cohen_kappa(values_a: Sequence[str], values_b: Sequence[str], labels: Sequence[str]) -> Dict[str, Any]:
    if len(values_a) != len(values_b):
        raise ValueError("Sequences must have the same length.")
    n = len(values_a)
    if n == 0:
        raise ValueError("Cannot compute agreement on zero rows.")

    agreements = sum(1 for a, b in zip(values_a, values_b) if a == b)
    observed = agreements / n

    counts_a = Counter(values_a)
    counts_b = Counter(values_b)
    all_labels = list(dict.fromkeys([*labels, *counts_a.keys(), *counts_b.keys()]))
    expected = sum((counts_a[label] / n) * (counts_b[label] / n) for label in all_labels)

    if math.isclose(1 - expected, 0.0):
        kappa = 1.0 if math.isclose(observed, 1.0) else 0.0
    else:
        kappa = (observed - expected) / (1 - expected)

    return {
        "n": n,
        "observed_agreement": observed,
        "observed_agreement_percent": round(observed * 100, 2),
        "expected_agreement": expected,
        "expected_agreement_percent": round(expected * 100, 2),
        "cohen_kappa": round(kappa, 4),
        "agreement_count": agreements,
        "distribution_a": dict(sorted(counts_a.items())),
        "distribution_b": dict(sorted(counts_b.items())),
        "percent_interpretation": interpret_percent(observed * 100),
        "kappa_interpretation": interpret_kappa(kappa),
    }


def build_pair_summary(
    *,
    rows: List[Dict[str, Any]],
    source_a: str,
    source_b: str,
    getter: Callable[[str, Dict[str, Any]], str],
    labels: Sequence[str],
) -> Dict[str, Any]:
    values_a = [getter(source_a, row) for row in rows]
    values_b = [getter(source_b, row) for row in rows]
    return compute_cohen_kappa(values_a, values_b, labels)


def compute_bundle_iaa(bundle_dir: Path) -> Dict[str, Any]:
    tasks_dir = bundle_dir / "tasks"
    manifest_path = bundle_dir / "manifest.json"
    task_cache = {
        "task1.json": load_json(tasks_dir / "task1.json"),
        "task2.json": load_json(tasks_dir / "task2.json"),
    }
    for rows in task_cache.values():
        for row in rows:
            if "gemini_annotation" not in row and "gemini_key" in row:
                row["gemini_annotation"] = row["gemini_key"]

    dimensions_report: List[Dict[str, Any]] = []
    matrix: Dict[str, Dict[str, Dict[str, Any]]] = {}

    for dim in DIMENSIONS:
        rows = task_cache[dim["task_file"]]
        pair_results = []
        matrix_key = f"{dim['task_name']}::{dim['dimension_id']}"
        matrix[matrix_key] = {}
        for source_a, source_b in PAIR_SPECS:
            result = build_pair_summary(
                rows=rows,
                source_a=source_a,
                source_b=source_b,
                getter=dim["getter"],
                labels=dim["labels"],
            )
            pair_label = f"{source_a}__vs__{source_b}"
            entry = {"pair": pair_label, "left": source_a, "right": source_b, **result}
            pair_results.append(entry)
            matrix[matrix_key][pair_label] = entry

        dimensions_report.append(
            {
                "task_file": dim["task_file"],
                "task_name": dim["task_name"],
                "dimension": dim["dimension_id"],
                "rows": len(rows),
                "pairs": pair_results,
            }
        )

    return {
        "bundle_dir": _repo_rel(bundle_dir),
        "manifest_path": _repo_rel(manifest_path),
        "pair_definitions": [f"{a}__vs__{b}" for a, b in PAIR_SPECS],
        "dimensions": dimensions_report,
        "matrix": matrix,
    }


def render_iaa_markdown(summary: Dict[str, Any]) -> str:
    lines = ["# IAA Summary", "", f"Bundle: `{summary['bundle_dir']}`", ""]
    for dimension_block in summary["dimensions"]:
        lines.extend(
            [
                f"## {dimension_block['task_name']} / {dimension_block['dimension']}",
                "",
                f"Rows: `{dimension_block['rows']}`",
                "",
                "| Pair | % Agreement | Interpretation | Cohen's kappa | Interpretation |",
                "|---|---:|---|---:|---|",
            ]
        )
        for pair in dimension_block["pairs"]:
            lines.append(
                "| "
                + f"{pair['pair']} | {pair['observed_agreement_percent']:.2f}% | {pair['percent_interpretation']} | "
                + f"{pair['cohen_kappa']:.4f} | {pair['kappa_interpretation']} |"
            )
        lines.append("")
    return "\n".join(lines)


def write_iaa_outputs(summary: Dict[str, Any], *, json_output: Path, markdown_output: Path) -> None:
    write_json(json_output, summary)
    write_text(markdown_output, render_iaa_markdown(summary))


def configure_iaa_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "figure.titlesize": 15,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def pair_to_matrix(summary_pairs: List[Dict[str, Any]]) -> np.ndarray:
    matrix = np.full((len(ACTOR_ORDER), len(ACTOR_ORDER)), np.nan, dtype=float)
    index = {actor: idx for idx, actor in enumerate(ACTOR_ORDER)}
    for pair in summary_pairs:
        i = index[pair["left"]]
        j = index[pair["right"]]
        matrix[i, j] = pair["cohen_kappa"]
        matrix[j, i] = pair["cohen_kappa"]
    return matrix


def upper_triangle_matrix(matrix: np.ndarray) -> np.ndarray:
    tri = matrix.copy()
    for i in range(tri.shape[0]):
        for j in range(tri.shape[1]):
            if j < i:
                tri[i, j] = np.nan
    return tri


def draw_heatmap(ax: plt.Axes, matrix: np.ndarray, title: str):
    display = upper_triangle_matrix(matrix)
    masked = np.ma.masked_invalid(display)
    cmap = HEATMAP_CMAP.copy()
    cmap.set_bad("#f5f2ea")
    im = ax.imshow(masked, vmin=0.0, vmax=1.0, cmap=cmap, aspect="equal")

    labels = [ACTOR_LABELS[item] for item in ACTOR_ORDER]
    ax.set_xticks(np.arange(len(labels)), labels=labels)
    ax.set_yticks(np.arange(len(labels)), labels=labels)
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False, length=0)
    ax.set_title(title, pad=12, fontweight="bold")

    for i in range(display.shape[0]):
        for j in range(display.shape[1]):
            value = display[i, j]
            if np.isnan(value):
                text = "-" if i == j else ""
                color = "#666666"
            else:
                text = f"{value:.2f}"
                color = "white" if value >= 0.55 else "#17202a"
            if text:
                ax.text(j, i, text, ha="center", va="center", color=color, fontsize=10, fontweight="bold")

    ax.set_xticks(np.arange(-0.5, len(labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(labels), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2.5)
    ax.tick_params(which="minor", bottom=False, left=False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    return im


def build_confusion_matrix(rows: List[Dict[str, Any]], field: str, labels: List[str]) -> np.ndarray:
    index = {label: idx for idx, label in enumerate(labels)}
    matrix = np.zeros((len(labels), len(labels)), dtype=int)
    for row in rows:
        left = row["annotator1"][field]
        right = row["annotator2"][field]
        matrix[index[left], index[right]] += 1
    return matrix


def draw_confusion(ax: plt.Axes, matrix: np.ndarray, labels: List[str], title: str):
    row_totals = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(matrix, row_totals, out=np.zeros_like(matrix, dtype=float), where=row_totals != 0)
    im = ax.imshow(normalized, cmap="Blues", vmin=0.0, vmax=1.0, aspect="equal")

    ax.set_xticks(np.arange(len(labels)), labels=labels)
    ax.set_yticks(np.arange(len(labels)), labels=labels)
    ax.set_xlabel("Annotator 2", labelpad=8)
    ax.set_ylabel("Annotator 1", labelpad=8)
    ax.set_title(title, pad=10, fontweight="bold")

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            count = matrix[i, j]
            pct = normalized[i, j] * 100
            color = "white" if normalized[i, j] >= 0.55 else "#17202a"
            ax.text(j, i - 0.05, f"{count}", ha="center", va="center", color=color, fontsize=12, fontweight="bold")
            ax.text(j, i + 0.23, f"{pct:.1f}%", ha="center", va="center", color=color, fontsize=8)

    ax.set_xticks(np.arange(-0.5, len(labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(labels), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2.5)
    ax.tick_params(which="minor", bottom=False, left=False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    return im


def create_heatmap_figure(summary: Dict[str, Any], output_path: Path) -> None:
    blocks = summary["dimensions"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.8), constrained_layout=True)
    ims = []
    for ax, block in zip(axes, blocks):
        title = DIMENSION_TITLES.get(block["dimension"], block["dimension"])
        ims.append(draw_heatmap(ax, pair_to_matrix(block["pairs"]), title))

    cbar = fig.colorbar(ims[-1], ax=axes, shrink=0.88, pad=0.02)
    cbar.set_label("Cohen's kappa", rotation=90, labelpad=12)
    cbar.set_ticks([0.0, 0.2, 0.4, 0.6, 0.8])
    cbar.set_ticklabels(["0.0\nPoor", "0.2\nSlight", "0.4\nFair", "0.6\nModerate", "0.8+\nSubstantial"])
    fig.suptitle("Human Verification and External Gemini Agreement", fontweight="bold")
    fig.text(
        0.5,
        0.02,
        "Upper-triangular heatmaps summarize Cohen's kappa for Annotator 1, Annotator 2, and the external Gemini annotator across the three evaluation dimensions.",
        ha="center",
        fontsize=10,
        color="#444444",
    )
    fig.savefig(output_path, dpi=260, bbox_inches="tight")
    plt.close(fig)


def create_task1_confusion(task1_rows: List[Dict[str, Any]], output_path: Path) -> None:
    quality_labels = ["weak", "usable", "strong"]
    difficulty_labels = ["easy", "medium", "hard"]
    quality_matrix = build_confusion_matrix(task1_rows, "human_quality_band", quality_labels)
    difficulty_matrix = build_confusion_matrix(task1_rows, "human_difficulty_band", difficulty_labels)

    fig = plt.figure(figsize=(13.8, 7.8), constrained_layout=True)
    grid = fig.add_gridspec(3, 2, height_ratios=[1.2, 15, 2.6], width_ratios=[1, 1])
    title_ax = fig.add_subplot(grid[0, :])
    axes = [fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1])]
    caption_ax = fig.add_subplot(grid[2, :])
    title_ax.axis("off")
    caption_ax.axis("off")

    draw_confusion(axes[0], quality_matrix, quality_labels, "Quality Band")
    im = draw_confusion(axes[1], difficulty_matrix, difficulty_labels, "Difficulty Band")
    cbar = fig.colorbar(im, ax=axes, shrink=0.9, pad=0.03)
    cbar.set_label("Row-normalized proportion", rotation=90, labelpad=12)
    title_ax.text(
        0.5,
        0.55,
        "Task 1: Annotator 1 vs Annotator 2 Confusion Matrices",
        ha="center",
        va="center",
        fontsize=15,
        fontweight="bold",
        color="#111111",
    )
    caption_ax.text(
        0.5,
        0.55,
        "Diagonal mass indicates agreement; off-diagonal cells reveal whether disagreements are local boundary shifts or larger categorical jumps.",
        ha="center",
        va="center",
        fontsize=10,
        color="#444444",
        wrap=True,
    )
    fig.savefig(output_path, dpi=260)
    plt.close(fig)


def create_task2_confusion(task2_rows: List[Dict[str, Any]], output_path: Path) -> None:
    labels = ["weak", "usable", "strong"]
    matrix = build_confusion_matrix(task2_rows, "human_inferential_validity_band", labels)

    fig = plt.figure(figsize=(7.4, 7.6), constrained_layout=True)
    grid = fig.add_gridspec(3, 1, height_ratios=[1.2, 15, 2.8])
    title_ax = fig.add_subplot(grid[0, 0])
    ax = fig.add_subplot(grid[1, 0])
    caption_ax = fig.add_subplot(grid[2, 0])
    title_ax.axis("off")
    caption_ax.axis("off")

    im = draw_confusion(ax, matrix, labels, "Inferential Validity Band")
    cbar = fig.colorbar(im, ax=ax, shrink=0.92, pad=0.03)
    cbar.set_label("Row-normalized proportion", rotation=90, labelpad=12)
    title_ax.text(
        0.5,
        0.55,
        "Task 2: Annotator 1 vs Annotator 2 Confusion Matrix",
        ha="center",
        va="center",
        fontsize=15,
        fontweight="bold",
        color="#111111",
    )
    caption_ax.text(
        0.5,
        0.55,
        "This matrix is most informative when disagreement concentrates near adjacent labels such as weak <-> usable.",
        ha="center",
        va="center",
        fontsize=10,
        color="#444444",
        wrap=True,
    )
    fig.savefig(output_path, dpi=260)
    plt.close(fig)


def build_visualization_report(bundle_dir: Path) -> Dict[str, str]:
    report_dir = bundle_dir / "reports"
    return {
        "bundle_dir": _repo_rel(bundle_dir),
        "heatmap": _repo_rel(report_dir / "iaa_kappa_heatmap_matrix.png"),
        "task1_confusion": _repo_rel(report_dir / "task1_a1_vs_a2_confusion.png"),
        "task2_confusion": _repo_rel(report_dir / "task2_a1_vs_a2_confusion.png"),
    }
