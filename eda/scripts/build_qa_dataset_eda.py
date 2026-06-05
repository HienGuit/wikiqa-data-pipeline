"""Build publication-ready EDA figures and tables for the QA dataset."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eda.utils.loader import load_qa_dataset
from eda.utils.plot_utils import (  # noqa: E402
    COLOR_QUALITY,
    COLOR_REASONING,
    apply_publication_style,
    save_fig,
)
from src.config import (  # noqa: E402
    FIGURES_DIR,
    QA_CANONICAL_JUDGED_CONTEXT_CLEANED,
    QA_THREE_WAY_ANALYSIS,
    QA_THREE_WAY_READY,
    ensure_dirs,
)

OUTPUT_DIR = FIGURES_DIR / "02_qa_dataset_eda"
FIGURE_DIR = OUTPUT_DIR / "figures"
TABLE_DIR = OUTPUT_DIR / "tables"
REPORT_PATH = OUTPUT_DIR / "eda_report.json"
SUMMARY_MD = OUTPUT_DIR / "eda_summary.md"


def repo_rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path)


def percent_table(df: pd.DataFrame, index: str, columns: str, *, column_order: list[str]) -> pd.DataFrame:
    table = pd.crosstab(df[index], df[columns], normalize="index")
    return table.reindex(columns=column_order, fill_value=0.0)


def count_table(df: pd.DataFrame, index: str, columns: str, *, column_order: list[str]) -> pd.DataFrame:
    table = pd.crosstab(df[index], df[columns])
    return table.reindex(columns=column_order, fill_value=0)


def draw_reasoning_distribution(df: pd.DataFrame) -> str:
    order = ["extraction", "bridge", "multi-sentence"]
    counts = df["final_reasoning_bucket"].value_counts().reindex(order)
    total = int(counts.sum())

    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    bars = ax.bar(
        counts.index,
        counts.values,
        color=[COLOR_REASONING[item] for item in counts.index],
        edgecolor="white",
        linewidth=1.2,
    )
    ax.set_title("Reasoning Bucket Distribution in the Final Release Dataset", pad=14)
    ax.set_ylabel("Number of QA pairs")
    ax.set_xlabel("")
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)

    ax.set_ylim(0, float(counts.max()) * 1.24)

    for bar, value in zip(bars, counts.values):
        pct = value / total * 100
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + total * 0.016,
            f"{value:,}\n({pct:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    filename = "01_reasoning_bucket_distribution.png"
    save_fig(fig, FIGURE_DIR, filename, close=True)
    return filename


def draw_domain_distribution(df: pd.DataFrame) -> str:
    counts = df["domain"].value_counts().sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    bars = ax.barh(counts.index, counts.values, color="#4C78A8", edgecolor="white", linewidth=1.0)
    ax.set_title("Domain Distribution in the Final Release Dataset")
    ax.set_xlabel("Number of QA pairs")
    ax.set_ylabel("")
    ax.grid(axis="x", alpha=0.25)
    ax.set_axisbelow(True)

    for bar, value in zip(bars, counts.values):
        ax.text(value + max(counts.values) * 0.01, bar.get_y() + bar.get_height() / 2, f"{value:,}", va="center")

    filename = "02_domain_distribution.png"
    save_fig(fig, FIGURE_DIR, filename, close=True)
    return filename


def draw_reasoning_difficulty_heatmap(df: pd.DataFrame) -> str:
    reasoning_order = ["extraction", "bridge", "multi-sentence"]
    difficulty_order = ["easy", "medium", "hard"]
    pct = percent_table(df, "final_reasoning_bucket", "difficulty_band", column_order=difficulty_order).reindex(
        reasoning_order
    )
    counts = count_table(df, "final_reasoning_bucket", "difficulty_band", column_order=difficulty_order).reindex(
        reasoning_order
    )

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    im = ax.imshow(pct.values, cmap="YlGnBu", vmin=0.0, vmax=max(0.6, float(pct.values.max())))
    ax.set_title("Difficulty Composition by Reasoning Bucket")
    ax.set_xticks(np.arange(len(difficulty_order)), labels=difficulty_order)
    ax.set_yticks(np.arange(len(reasoning_order)), labels=reasoning_order)
    ax.set_xlabel("Difficulty band")
    ax.set_ylabel("Reasoning bucket")

    for i in range(pct.shape[0]):
        for j in range(pct.shape[1]):
            p = pct.iloc[i, j] * 100
            n = int(counts.iloc[i, j])
            color = "white" if pct.iloc[i, j] >= 0.35 else "#17202a"
            ax.text(j, i, f"{p:.1f}%\n(n={n})", ha="center", va="center", color=color, fontsize=10, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("Row-normalized proportion")
    filename = "03_reasoning_difficulty_heatmap.png"
    save_fig(fig, FIGURE_DIR, filename, close=True)
    return filename


def draw_stacked_bar(
    table: pd.DataFrame,
    *,
    color_map: dict[str, str],
    title: str,
    xlabel: str,
    ylabel: str,
    filename: str,
    legend_title: str,
    rotate_x: int = 0,
    figsize: tuple[float, float] = (8.6, 5.2),
    legend_loc: str = "upper center",
    legend_bbox: tuple[float, float] = (0.5, -0.14),
    legend_ncol: int | None = None,
    title_pad: float = 10.0,
    top_margin: float | None = None,
) -> str:
    fig, ax = plt.subplots(figsize=figsize)
    bottom = np.zeros(len(table))
    x = np.arange(len(table.index))

    for column in table.columns:
        values = table[column].values * 100
        ax.bar(
            x,
            values,
            bottom=bottom,
            label=column,
            color=color_map.get(column, "#999999"),
            edgecolor="white",
            linewidth=0.8,
        )
        bottom += values

    ax.set_title(title, pad=title_pad)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.set_xticks(x, labels=table.index, rotation=rotate_x, ha="right" if rotate_x else "center")
    ax.set_ylim(0, 100)
    ax.set_yticks(np.arange(0, 101, 20))
    ax.set_yticklabels([f"{tick}%" for tick in np.arange(0, 101, 20)])
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend(
        title=legend_title,
        ncol=legend_ncol or min(3, len(table.columns)),
        frameon=False,
        loc=legend_loc,
        bbox_to_anchor=legend_bbox,
    )

    if top_margin is not None:
        fig.subplots_adjust(top=top_margin)

    save_fig(fig, FIGURE_DIR, filename, close=True)
    return filename


def build_length_statistics(df: pd.DataFrame) -> tuple[str, str]:
    rows = []
    for field in ("question", "answer", "context"):
        token = df[f"{field}_token_len"]
        char = df[f"{field}_char_len"]
        rows.append(
            {
                "field": field,
                "mean_tokens": round(float(token.mean()), 2),
                "median_tokens": round(float(token.median()), 2),
                "std_tokens": round(float(token.std(ddof=0)), 2),
                "p10_tokens": round(float(token.quantile(0.10)), 2),
                "p90_tokens": round(float(token.quantile(0.90)), 2),
                "min_tokens": int(token.min()),
                "max_tokens": int(token.max()),
                "mean_chars": round(float(char.mean()), 2),
                "median_chars": round(float(char.median()), 2),
                "std_chars": round(float(char.std(ddof=0)), 2),
                "p10_chars": round(float(char.quantile(0.10)), 2),
                "p90_chars": round(float(char.quantile(0.90)), 2),
                "min_chars": int(char.min()),
                "max_chars": int(char.max()),
            }
        )

    stats_df = pd.DataFrame(rows)
    csv_name = "text_length_statistics.csv"
    md_name = "text_length_statistics.md"
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    stats_df.to_csv(TABLE_DIR / csv_name, index=False, encoding="utf-8")
    (TABLE_DIR / md_name).write_text(stats_df.to_markdown(index=False), encoding="utf-8")
    return csv_name, md_name


def draw_length_boxplots(df: pd.DataFrame) -> str:
    order = ["extraction", "bridge", "multi-sentence"]
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.2))

    question_data = [df.loc[df["final_reasoning_bucket"] == item, "question_token_len"] for item in order]
    context_data = [df.loc[df["final_reasoning_bucket"] == item, "context_token_len"] for item in order]

    for ax, data, title, ylabel in [
        (axes[0], question_data, "Question Length by Reasoning Bucket", "Question length (tokens)"),
        (axes[1], context_data, "Context Length by Reasoning Bucket", "Context length (tokens)"),
    ]:
        bp = ax.boxplot(data, tick_labels=order, patch_artist=True, showfliers=False)
        for patch, bucket in zip(bp["boxes"], order):
            patch.set_facecolor(COLOR_REASONING[bucket])
            patch.set_alpha(0.78)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)

    filename = "07_length_by_reasoning_boxplots.png"
    save_fig(fig, FIGURE_DIR, filename, close=True)
    return filename


def build_summary_markdown(
    release_df: pd.DataFrame,
    judged_df: pd.DataFrame,
    figures: list[str],
    table_md: str,
) -> None:
    reason_share = (
        release_df["final_reasoning_bucket"]
        .value_counts(normalize=True)
        .reindex(["extraction", "bridge", "multi-sentence"])
        * 100
    )
    quality_reasoning = percent_table(
        judged_df,
        "reasoning_type",
        "quality_band",
        column_order=["weak", "usable", "strong"],
    ).reindex(["extraction", "multi-sentence"])
    weak_extraction = quality_reasoning.loc["extraction", "weak"] * 100
    weak_multi = quality_reasoning.loc["multi-sentence", "weak"] * 100
    medium_hard = percent_table(
        release_df,
        "final_reasoning_bucket",
        "difficulty_band",
        column_order=["easy", "medium", "hard"],
    ).reindex(["extraction", "bridge", "multi-sentence"])
    domain_quality = percent_table(judged_df, "domain", "quality_band", column_order=["weak", "usable", "strong"])
    weakest_domain = domain_quality["weak"].sort_values(ascending=False).index[0]

    lines = [
        "# QA Dataset EDA Summary",
        "",
        "## Dataset Scope",
        f"- Public final release dataset: `{len(release_df):,}` QA pairs mirrored from the analysis source `qa_pairs_three_way_analysis.jsonl` into `qa_pairs_three_way_ready.jsonl`.",
        f"- External-Gemini diagnostic dataset: `{len(judged_df):,}` QA pairs from `qa_pairs_canonical_judged_context_cleaned.jsonl`.",
        "",
        "## Key Findings",
        f"- Extraction remains the dominant bucket ({reason_share['extraction']:.1f}%), while bridge ({reason_share['bridge']:.1f}%) captures the transitional reasoning region between literal extraction and fully multi-sentence inference ({reason_share['multi-sentence']:.1f}%).",
        f"- Multi-sentence QA has a markedly higher weak-quality share than extraction in the full Gemini-annotated pool ({weak_multi:.1f}% vs {weak_extraction:.1f}%), supporting the decision to apply stricter human verification to inferential content.",
        f"- Within this final three-way release, the combined medium+hard share is {((medium_hard.loc['extraction', 'medium'] + medium_hard.loc['extraction', 'hard']) * 100):.1f}% for extraction, {((medium_hard.loc['bridge', 'medium'] + medium_hard.loc['bridge', 'hard']) * 100):.1f}% for bridge, and {((medium_hard.loc['multi-sentence', 'medium'] + medium_hard.loc['multi-sentence', 'hard']) * 100):.1f}% for multi-sentence. This reflects the current release composition rather than a universal claim about all inferential QA.",
        f"- `{weakest_domain}` shows the highest weak-quality proportion in the full Gemini-annotated pool and should therefore be discussed explicitly in the limitations section.",
        "",
        "## Outputs",
    ]
    lines.extend([f"- Figure: `{name}`" for name in figures])
    lines.append(f"- Table: `{table_md}`")
    SUMMARY_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    apply_publication_style()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    legacy_answer_type_figure = FIGURE_DIR / "08_answer_type_difficulty_grouped.png"
    if legacy_answer_type_figure.exists():
        legacy_answer_type_figure.unlink()

    release_df = load_qa_dataset(
        QA_THREE_WAY_ANALYSIS,
        required_columns={"final_reasoning_bucket", "quality_band", "difficulty_band", "inferential_validity_band"},
    )
    judged_df = load_qa_dataset(
        QA_CANONICAL_JUDGED_CONTEXT_CLEANED,
        required_columns={"reasoning_type", "quality_band", "difficulty_band", "inferential_validity_band"},
    )
    figures: list[str] = []
    figures.append(draw_reasoning_distribution(release_df))
    figures.append(draw_domain_distribution(release_df))
    figures.append(draw_reasoning_difficulty_heatmap(release_df))

    domain_reasoning = percent_table(
        release_df,
        "domain",
        "final_reasoning_bucket",
        column_order=["extraction", "bridge", "multi-sentence"],
    )
    figures.append(
        draw_stacked_bar(
            domain_reasoning,
            color_map=COLOR_REASONING,
            title="Reasoning Composition by Domain",
            xlabel="Domain",
            ylabel="Proportion of QA pairs",
            filename="04_domain_reasoning_stacked.png",
            legend_title="Reasoning bucket",
            rotate_x=25,
            figsize=(10.2, 5.3),
            legend_loc="lower center",
            legend_bbox=(0.5, 1.12),
            legend_ncol=3,
            title_pad=22,
            top_margin=0.82,
        )
    )

    quality_reasoning = percent_table(
        judged_df,
        "reasoning_type",
        "quality_band",
        column_order=["weak", "usable", "strong"],
    ).reindex(["extraction", "multi-sentence"])
    figures.append(
        draw_stacked_bar(
            quality_reasoning,
            color_map=COLOR_QUALITY,
            title="Quality Band by Reasoning Type (Full Gemini-Annotated Pool)",
            xlabel="Reasoning type",
            ylabel="Proportion of QA pairs",
            filename="05_quality_reasoning_stacked.png",
            legend_title="Quality band",
            figsize=(8.6, 5.0),
        )
    )

    quality_domain = percent_table(
        judged_df,
        "domain",
        "quality_band",
        column_order=["weak", "usable", "strong"],
    )
    quality_domain = quality_domain.sort_values(by="weak", ascending=False)
    figures.append(
        draw_stacked_bar(
            quality_domain,
            color_map=COLOR_QUALITY,
            title="Quality Band by Domain (Full Gemini-Annotated Pool)",
            xlabel="Domain",
            ylabel="Proportion of QA pairs",
            filename="06_quality_domain_stacked.png",
            legend_title="Quality band",
            rotate_x=25,
            figsize=(10.2, 5.3),
            legend_loc="lower center",
            legend_bbox=(0.5, 1.12),
            legend_ncol=3,
            title_pad=22,
            top_margin=0.82,
        )
    )

    figures.append(draw_length_boxplots(release_df))
    table_csv, table_md = build_length_statistics(release_df)
    build_summary_markdown(release_df, judged_df, figures, table_md)

    report = {
        "release_dataset": repo_rel(QA_THREE_WAY_READY),
        "analysis_dataset": repo_rel(QA_THREE_WAY_ANALYSIS),
        "gemini_annotated_dataset": repo_rel(QA_CANONICAL_JUDGED_CONTEXT_CLEANED),
        "figure_dir": repo_rel(FIGURE_DIR),
        "table_dir": repo_rel(TABLE_DIR),
        "figures": figures,
        "vector_figures": [name.replace(".png", ".pdf") for name in figures],
        "tables": {"csv": table_csv, "markdown": table_md},
        "summary_markdown": SUMMARY_MD.name,
        "notes": {
            "overview_charts_source": "Internal three-way analysis dataset with legacy difficulty labels retained for diagnostics",
            "public_release_path": repo_rel(QA_THREE_WAY_READY),
            "quality_diagnostic_charts_source": "Full Gemini-annotated pool before weak-quality filtering",
            "difficulty_note": "The hard band is extremely sparse (n=5) in the final release dataset and is therefore interpreted cautiously.",
            "answer_type_note": "Answer-type analysis has been moved to the feature-engineering EDA stage.",
        },
    }
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
