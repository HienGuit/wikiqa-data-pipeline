"""Build full and filtered feature matrices for the QA dataset.

Workflow
--------
1. Build the *full* feature matrix via ``build_feature_frame``.
2. Drop degenerate columns (all-null or single-value).
3. Compute pairwise Pearson correlations for numeric features.
4. For each pair exceeding the multicollinearity threshold (|r| >= 0.80),
   drop the less interpretable feature according to a predefined policy.
5. Save both *full* (before drop) and *final* (after drop) matrices.
6. Write a detailed JSON report documenting every decision.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eda.utils.loader import load_qa_dataset  # noqa: E402
from src.config import (  # noqa: E402
    ENTITY_DB,
    FEATURE_MATRIX_BUILD_REPORT,
    FEATURE_MATRIX_FINAL,
    FEATURE_MATRIX_FULL,
    QA_THREE_WAY_ANALYSIS,
    ensure_dirs,
)
from src.features import build_feature_frame, load_entity_db  # noqa: E402

# ---------------------------------------------------------------------------
# Multicollinearity resolution policy
# ---------------------------------------------------------------------------
# Each entry maps a (feature_left, feature_right) pair to the column to DROP
# and a short rationale.  Only pairs with |r| >= MULTICOLLINEARITY_THRESHOLD
# in the EDA correlation matrix need entries here.  The builder will emit a
# warning if it encounters a new high-correlation pair not covered by this
# table so nothing is silently ignored.

MULTICOLLINEARITY_THRESHOLD = 0.80

MULTICOLLINEARITY_POLICY: list[dict[str, str]] = [
    {
        "feature_left": "a_length",
        "feature_right": "answer_density",
        "drop": "answer_density",
        "keep": "a_length",
        "rationale": (
            "answer_density = a_length / ctx_length, nearly perfectly correlated "
            "(|r|=0.99).  Keeping a_length because it is a direct measurement; "
            "the density relationship can be reconstructed from a_length and "
            "ctx_length if needed."
        ),
    },
    {
        "feature_left": "statements_rank",
        "feature_right": "references_rank",
        "drop": "references_rank",
        "keep": "statements_rank",
        "rationale": (
            "Wikidata statements_rank and references_rank are highly correlated "
            "(|r|=0.95).  Keeping statements_rank because it captures structured "
            "factual claims, which is more directly relevant to QA difficulty "
            "than reference count."
        ),
    },
    {
        "feature_left": "answer_position_ratio",
        "feature_right": "answer_sentence_index_ratio",
        "drop": "answer_sentence_index_ratio",
        "keep": "answer_position_ratio",
        "rationale": (
            "Both encode where the answer appears in the context (|r|=0.94).  "
            "Keeping answer_position_ratio (character-level) because it has "
            "finer granularity than sentence-index-based measurement."
        ),
    },
    {
        "feature_left": "site_links_rank",
        "feature_right": "statements_rank",
        "drop": "site_links_rank",
        "keep": "statements_rank",
        "rationale": (
            "site_links_rank is highly correlated with statements_rank "
            "(|r|=0.93).  Keeping statements_rank (factual claims) over "
            "site_links_rank (cross-wiki linkage) as the former is more "
            "semantically relevant for knowledge difficulty."
        ),
    },
    {
        "feature_left": "site_links_rank",
        "feature_right": "references_rank",
        "drop": "site_links_rank",
        "keep": "references_rank",
        "rationale": (
            "Also highly correlated (|r|=0.87).  site_links_rank is already "
            "marked for removal by the statements_rank pair above, so this "
            "entry is confirmatory.  No additional column needs to be dropped."
        ),
    },
    {
        "feature_left": "knowledge_difficulty",
        "feature_right": "page_views_rank",
        "drop": "page_views_rank",
        "keep": "knowledge_difficulty",
        "rationale": (
            "knowledge_difficulty is a composite score (weighted combination of "
            "page_views, site_links, wiki_count, statements, references ranks). "
            "page_views_rank is its dominant component (|r|=0.80).  Keeping the "
            "composite because it integrates multiple signals into a single "
            "interpretable difficulty proxy."
        ),
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build feature matrices for the QA dataset.")
    parser.add_argument("--input", default=str(QA_THREE_WAY_ANALYSIS))
    parser.add_argument("--entity-db", default=str(ENTITY_DB))
    parser.add_argument("--output-full", default=str(FEATURE_MATRIX_FULL))
    parser.add_argument("--output-final", default=str(FEATURE_MATRIX_FINAL))
    parser.add_argument("--report", default=str(FEATURE_MATRIX_BUILD_REPORT))
    parser.add_argument("--limit-rows", type=int, default=None)
    parser.add_argument(
        "--threshold",
        type=float,
        default=MULTICOLLINEARITY_THRESHOLD,
        help="Correlation threshold for multicollinearity detection (default: 0.80).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Step 1: Remove degenerate columns
# ---------------------------------------------------------------------------


def _drop_degenerate_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Drop all-null or single-value columns.  Returns cleaned df and drop log."""
    drop_log: list[dict[str, Any]] = []
    removable: list[str] = []
    for column in df.columns:
        series = df[column]
        if series.isna().all():
            removable.append(column)
            drop_log.append({"column": column, "reason": "all_null"})
            continue
        non_null = series.dropna()
        if non_null.empty:
            removable.append(column)
            drop_log.append({"column": column, "reason": "all_null"})
            continue
        if non_null.nunique(dropna=True) <= 1:
            removable.append(column)
            drop_log.append(
                {
                    "column": column,
                    "reason": "single_value",
                    "constant_value": str(non_null.iloc[0]),
                }
            )
    return df.drop(columns=removable), drop_log


# ---------------------------------------------------------------------------
# Step 2: Detect and resolve multicollinearity
# ---------------------------------------------------------------------------


def _detect_multicollinearity(
    df: pd.DataFrame,
    threshold: float,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Detect high-correlation pairs and resolve using the policy table."""
    numeric = df.select_dtypes(include=["number"]).copy()
    numeric = numeric.drop(columns=[c for c in ["row_id"] if c in numeric.columns])
    corr = numeric.corr(numeric_only=True).abs()

    # Discover all pairs above threshold
    detected_pairs: list[dict[str, Any]] = []
    columns = list(corr.columns)
    for i, left in enumerate(columns):
        for right in columns[i + 1 :]:
            value = corr.loc[left, right]
            if pd.notna(value) and value >= threshold:
                detected_pairs.append(
                    {
                        "feature_left": left,
                        "feature_right": right,
                        "abs_corr": round(float(value), 4),
                    }
                )

    # Build lookup from policy
    policy_lookup: dict[tuple[str, str], dict[str, str]] = {}
    for entry in MULTICOLLINEARITY_POLICY:
        key = (entry["feature_left"], entry["feature_right"])
        policy_lookup[key] = entry
        # Also register reversed key
        key_rev = (entry["feature_right"], entry["feature_left"])
        policy_lookup[key_rev] = entry

    # Resolve each detected pair
    to_drop: set[str] = set()
    resolution_log: list[dict[str, Any]] = []
    uncovered_pairs: list[dict[str, Any]] = []

    for pair in detected_pairs:
        key = (pair["feature_left"], pair["feature_right"])
        policy = policy_lookup.get(key)
        if policy is None:
            uncovered_pairs.append(pair)
            resolution_log.append(
                {
                    **pair,
                    "action": "WARNING_NO_POLICY",
                    "drop": None,
                    "keep": None,
                    "rationale": "No policy entry found for this pair. Manual review required.",
                }
            )
            continue

        drop_col = policy["drop"]
        keep_col = policy["keep"]

        # Only drop if not already removed and present in the dataframe
        if drop_col in df.columns:
            to_drop.add(drop_col)

        resolution_log.append(
            {
                **pair,
                "action": "drop",
                "drop": drop_col,
                "keep": keep_col,
                "rationale": policy["rationale"],
            }
        )

    if uncovered_pairs:
        print(f"\nWARNING: {len(uncovered_pairs)} high-correlation pair(s) have no policy entry:")
        for pair in uncovered_pairs:
            print(f"  {pair['feature_left']} <-> {pair['feature_right']} (|r|={pair['abs_corr']})")
        print()

    final_df = df.drop(columns=sorted(to_drop))
    return final_df, resolution_log


def select_final_columns(
    df: pd.DataFrame,
    threshold: float = MULTICOLLINEARITY_THRESHOLD,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Full column selection pipeline: degenerate removal + multicollinearity resolution."""
    columns_before = list(df.columns)

    # Step 1: degenerate
    df_clean, degenerate_log = _drop_degenerate_columns(df)
    columns_after_degenerate = list(df_clean.columns)

    # Step 2: multicollinearity
    df_final, multicollinearity_log = _detect_multicollinearity(df_clean, threshold)
    columns_after_final = list(df_final.columns)

    all_dropped = sorted(set(columns_before) - set(columns_after_final))

    drop_report = {
        "threshold": threshold,
        "columns_before": columns_before,
        "columns_after_degenerate_removal": columns_after_degenerate,
        "columns_after_multicollinearity_removal": columns_after_final,
        "degenerate_drops": degenerate_log,
        "degenerate_drop_count": len(degenerate_log),
        "multicollinearity_resolutions": multicollinearity_log,
        "multicollinearity_drop_count": len(
            [r for r in multicollinearity_log if r["action"] == "drop" and r["drop"] is not None]
        ),
        "total_columns_dropped": len(all_dropped),
        "dropped_column_names": all_dropped,
        "final_column_count": len(columns_after_final),
        "final_numeric_features": [
            c for c in columns_after_final if c in df_final.select_dtypes(include=["number"]).columns and c != "row_id"
        ],
    }
    return df_final, drop_report


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ensure_dirs()
    args = parse_args()

    dataset = load_qa_dataset(
        args.input,
        required_columns={"final_reasoning_bucket", "quality_band", "difficulty_band", "inferential_validity_band"},
    )
    if args.limit_rows:
        dataset = dataset.head(args.limit_rows).copy()
    entity_db = load_entity_db(args.entity_db)

    full_df = build_feature_frame(dataset, entity_db)
    final_df, drop_report = select_final_columns(full_df, threshold=args.threshold)

    Path(args.output_full).parent.mkdir(parents=True, exist_ok=True)
    full_df.to_csv(args.output_full, index=False, encoding="utf-8")
    final_df.to_csv(args.output_final, index=False, encoding="utf-8")

    report: dict[str, Any] = {
        "input": str(args.input),
        "entity_db": str(args.entity_db),
        "output_full": str(args.output_full),
        "output_final": str(args.output_final),
        "row_count": int(len(full_df)),
        "full_column_count": int(len(full_df.columns)),
        "final_column_count": int(len(final_df.columns)),
        "dropped_columns": drop_report["dropped_column_names"],
        "limit_rows": args.limit_rows,
        "knowledge_difficulty_non_null": int(full_df["knowledge_difficulty"].notna().sum()),
        "popularity_source_counts": {
            key: int(value) for key, value in full_df["popularity_source"].value_counts(dropna=False).to_dict().items()
        },
        "feature_selection": drop_report,
    }
    Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
