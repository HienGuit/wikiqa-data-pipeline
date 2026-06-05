"""Build feature-engineering EDA figures and reports."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eda.utils.plot_utils import COLOR_REASONING, apply_publication_style, save_fig  # noqa: E402
from src.config import (  # noqa: E402
    FEATURE_ENGINEERING_FIGURES_DIR,
    FEATURE_MATRIX_EDA_REPORT,
    FEATURE_MATRIX_FULL,
    ensure_dirs,
)


def repo_rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build EDA for the feature matrix.")
    parser.add_argument("--input", default=str(FEATURE_MATRIX_FULL))
    parser.add_argument("--output-dir", default=str(FEATURE_ENGINEERING_FIGURES_DIR))
    parser.add_argument("--report", default=str(FEATURE_MATRIX_EDA_REPORT))
    return parser.parse_args()


def bar_distribution(series: pd.Series, *, figure_dir: Path, title: str, xlabel: str, filename: str) -> str:
    counts = series.fillna("missing").value_counts()
    fig, ax = plt.subplots(figsize=(9.6, 5.2))
    bars = ax.bar(counts.index.astype(str), counts.values, color="#4C78A8", edgecolor="white", linewidth=0.8)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Number of samples")
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    ax.set_xticks(np.arange(len(counts.index)), labels=counts.index.astype(str), rotation=25, ha="right")
    for bar, value in zip(bars, counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{int(value)}", ha="center", va="bottom", fontsize=8
        )
    save_fig(fig, figure_dir, filename, close=True)
    return filename


def draw_knowledge_distribution(df: pd.DataFrame, figure_dir: Path) -> str:
    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    values = df["knowledge_difficulty"].dropna()
    ax.hist(values, bins=30, color="#2B6CB0", edgecolor="white")
    ax.set_title("Knowledge Difficulty Distribution")
    ax.set_xlabel("knowledge_difficulty")
    ax.set_ylabel("Number of samples")
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    filename = "01_knowledge_difficulty_distribution.png"
    save_fig(fig, figure_dir, filename, close=True)
    return filename


def draw_structural_boxplots(df: pd.DataFrame, figure_dir: Path) -> str:
    order = ["extraction", "bridge", "multi-sentence"]
    features = [
        ("q_length", "Question length"),
        ("a_length", "Answer length"),
        ("ctx_length", "Context length"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 5.0))
    for ax, (feature, title) in zip(axes, features):
        data = [df.loc[df["final_reasoning_bucket"] == bucket, feature] for bucket in order]
        bp = ax.boxplot(data, tick_labels=order, patch_artist=True, showfliers=False)
        for patch, bucket in zip(bp["boxes"], order):
            patch.set_facecolor(COLOR_REASONING[bucket])
            patch.set_alpha(0.8)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)
    filename = "05_structural_boxplots_by_reasoning.png"
    save_fig(fig, figure_dir, filename, close=True)
    return filename


def draw_correlation_heatmap(df: pd.DataFrame, figure_dir: Path) -> str:
    numeric = df.select_dtypes(include=["number"]).copy()
    numeric = numeric.drop(columns=[column for column in ["row_id"] if column in numeric.columns])
    corr = numeric.corr(numeric_only=True)
    mask = np.tril(np.ones_like(corr.values, dtype=bool), k=-1)
    masked = np.ma.array(corr.values, mask=mask)
    cmap = plt.get_cmap("coolwarm").copy()
    cmap.set_bad(color="white")

    fig, ax = plt.subplots(figsize=(12.0, 10.0))
    im = ax.imshow(masked, cmap=cmap, vmin=-1, vmax=1)
    ax.set_title("Feature Correlation Heatmap (Upper Triangle)")
    ax.set_xticks(np.arange(len(corr.columns)), labels=corr.columns, rotation=90)
    ax.set_yticks(np.arange(len(corr.index)), labels=corr.index)
    cbar = fig.colorbar(im, ax=ax, pad=0.01)
    cbar.set_label("Pearson correlation")
    filename = "07_feature_correlation_heatmap.png"
    save_fig(fig, figure_dir, filename, close=True)
    return filename


def build_multicollinearity_table(df: pd.DataFrame, table_dir: Path) -> tuple[str, str]:
    numeric = df.select_dtypes(include=["number"]).copy()
    numeric = numeric.drop(columns=[column for column in ["row_id"] if column in numeric.columns])
    corr = numeric.corr(numeric_only=True).abs()
    rows = []
    columns = list(corr.columns)
    for i, left in enumerate(columns):
        for right in columns[i + 1 :]:
            value = corr.loc[left, right]
            if pd.notna(value) and value >= 0.8:
                rows.append({"feature_left": left, "feature_right": right, "abs_corr": round(float(value), 4)})
    result = (
        pd.DataFrame(rows).sort_values(by="abs_corr", ascending=False)
        if rows
        else pd.DataFrame(columns=["feature_left", "feature_right", "abs_corr"])
    )
    csv_name = "multicollinearity_pairs.csv"
    md_name = "multicollinearity_pairs.md"
    table_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(table_dir / csv_name, index=False, encoding="utf-8")
    (table_dir / md_name).write_text(result.to_markdown(index=False), encoding="utf-8")
    return csv_name, md_name


def build_summary(df: pd.DataFrame, figures: list[str], multicol_csv: str, summary_path: Path) -> None:
    question_type_top = df["question_type"].value_counts().head(3).to_dict()
    answer_type_top = df["answer_type"].value_counts().head(3).to_dict()
    popularity_source = df["popularity_source"].value_counts().to_dict()
    lines = [
        "# Feature Engineering EDA Summary",
        "",
        f"- Feature matrix rows: `{len(df):,}`",
        f"- Feature matrix columns: `{len(df.columns):,}`",
        f"- Non-null knowledge difficulty rows: `{int(df['knowledge_difficulty'].notna().sum()):,}`",
        f"- Top question types: `{question_type_top}`",
        f"- Top answer types: `{answer_type_top}`",
        f"- Popularity source mix: `{popularity_source}`",
        "- Full-matrix phase-1 knowledge signals available before feature selection: `page_views_rank`, `site_links_rank`, `wiki_count_rank`, `statements_rank`, `references_rank`, and `knowledge_difficulty`.",
        "- After multicollinearity-based feature selection, the retained knowledge signals are `page_views_rank`, `wiki_count_rank`, `statements_rank`, and `knowledge_difficulty`.",
        "- Excluded from phase 1: `wiki_level` and `linked_entities` due to insufficient provenance or API stability.",
        f"- Multicollinearity table: `{multicol_csv}`",
        "- The correlation heatmap is retained as the main visual evidence for feature-pruning decisions.",
        "",
        "## Figures",
    ]
    lines.extend([f"- `{figure}`" for figure in figures])
    summary_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()
    ensure_dirs()
    apply_publication_style()
    output_dir = Path(args.output_dir)
    figure_dir = output_dir / "figures"
    table_dir = output_dir / "tables"
    summary_md = output_dir / "feature_engineering_eda_summary.md"
    figure_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    legacy_scatter = figure_dir / "06_ctx_length_vs_knowledge_difficulty.png"
    if legacy_scatter.exists():
        legacy_scatter.unlink()

    df = pd.read_csv(args.input)
    figures = [
        draw_knowledge_distribution(df, figure_dir),
        bar_distribution(
            df["answer_type"],
            figure_dir=figure_dir,
            title="Answer Type Distribution",
            xlabel="Answer type",
            filename="02_answer_type_distribution.png",
        ),
        bar_distribution(
            df["popularity_source"],
            figure_dir=figure_dir,
            title="Popularity Source Distribution",
            xlabel="Popularity source",
            filename="03_popularity_source_distribution.png",
        ),
        bar_distribution(
            df["question_type"],
            figure_dir=figure_dir,
            title="Question Type Distribution",
            xlabel="Question type",
            filename="04_question_type_distribution.png",
        ),
        draw_structural_boxplots(df, figure_dir),
        draw_correlation_heatmap(df, figure_dir),
    ]
    multicol_csv, multicol_md = build_multicollinearity_table(df, table_dir)
    build_summary(df, figures, multicol_csv, summary_md)

    report = {
        "input": repo_rel(Path(args.input)),
        "figure_dir": repo_rel(figure_dir),
        "table_dir": repo_rel(table_dir),
        "figures": figures,
        "vector_figures": [name.replace(".png", ".pdf") for name in figures],
        "tables": {"csv": multicol_csv, "markdown": multicol_md},
        "summary_markdown": summary_md.name,
        "notes": {
            "difficulty_proxy_used": False,
            "slicing_dimensions": ["final_reasoning_bucket", "quality_band"],
            "removed_figures": ["06_ctx_length_vs_knowledge_difficulty.png"],
            "wiki_level_note": "Wiki level is excluded from the active phase-1 feature set.",
            "linked_entities_note": "Linked-entity connectivity was removed from the final phase-1 feature set because API coverage was not stable enough.",
            "full_matrix_phase1_knowledge_signals": [
                "page_views_rank",
                "site_links_rank",
                "wiki_count_rank",
                "statements_rank",
                "references_rank",
                "knowledge_difficulty",
            ],
            "retained_final_matrix_knowledge_signals": [
                "page_views_rank",
                "wiki_count_rank",
                "statements_rank",
                "knowledge_difficulty",
            ],
        },
    }
    Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
