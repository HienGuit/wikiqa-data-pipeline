"""
chunk_filter.py -- Hard filter + Quality filter + Stratified sampler
============================================================
Input : data/interim/wiki_chunks.jsonl
Output: data/interim/chunks_filtered.jsonl
        data/interim/chunks_sampled.jsonl

Thiet ke tu EDA findings:
  - 29 chunks wiki artifact -> loai tai day
  - Chunk ngan <300 chars: <10% moi domain -> safe to filter
  - history/law chiem ty trong lon -> bat buoc stratify theo domain
  - Bai dai max 125 chunks/bai -> can cap per title
  - Intro ratio cao o mot so domain -> can kiem soat
  - Chunk index 1-3 thuong co noi dung "thit" hon intro
"""
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    FILTERED_CHUNKS,
    RAW_CHUNKS,
    SAMPLED_CHUNKS,
    ensure_dirs,
    load_filter_config,
)


def hard_filter(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Loai cac chunk vi pham tieu chi ky thuat khong the chap nhan.
    """
    hard_cfg = cfg["hard_filter"]
    endings = list(".?!\"'…")
    mask = (
        (~df["text"].str.contains(r"\{\{|\[\[", regex=True))
        & (~df["text"].str.contains(r"<[a-zA-Z][^>]*>", regex=True))
        & (df["char_count"] >= hard_cfg["min_char_count"])
        & (df["char_count"] <= hard_cfg["max_char_count"])
        & (df["text"].str.strip().str[-1].isin(endings))
        & (df["text"].str.count(r"[.!?]") >= hard_cfg["min_sentences"])
    )
    result = df.loc[mask].copy()
    print(f"[hard_filter] {len(df):,} -> {len(result):,} (loai {len(df) - len(result):,})")
    return result


def quality_filter(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Loai chunk dang bang so lieu qua nhieu, khong phai van xuoi."""
    quality_cfg = cfg["quality_filter"]
    digit_ratio = df["text"].str.count(r"\d") / df["char_count"]
    result = df.loc[digit_ratio <= quality_cfg["max_digit_ratio"]].copy()
    print(f"[quality_filter] {len(df):,} -> {len(result):,} (loai {len(df) - len(result):,})")
    return result


def _assign_priority(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Gan diem uu tien truoc khi sampling.

    Priority 0 = non-intro va chunk_index thuoc nhom uu tien.
    Priority 1 = non-intro con lai.
    Priority 2 = intro.
    """
    sampling_cfg = cfg["sampling"]
    prioritized_positions = set(sampling_cfg["priority_sections"])
    result = df.copy()
    result["is_intro"] = result["section"].fillna("").eq("")
    in_priority_position = result["chunk_index"].isin(prioritized_positions)
    conditions = [
        (~result["is_intro"]) & in_priority_position,
        (~result["is_intro"]) & (~in_priority_position),
    ]
    result["_priority"] = np.select(conditions, [0, 1], default=2)
    return result


def stratified_sampler(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Stratified sampling theo domain, per-title cap va gioi han intro ratio.
    """
    sampling_cfg = cfg["sampling"]
    target = sampling_cfg["target_per_domain"]
    max_per_title = sampling_cfg["max_chunks_per_title"]
    max_intro_ratio = sampling_cfg["max_intro_ratio_per_domain"]

    prioritized = _assign_priority(df, cfg)
    sampled_parts: list[pd.DataFrame] = []

    for domain, group in prioritized.groupby("domain", sort=True):
        limited = (
            group.sort_values(["_priority", "chunk_index", "char_count"], ascending=[True, True, False])
            .groupby("title", group_keys=False)
            .head(max_per_title)
        )

        intro_pool = limited.loc[limited["is_intro"]].sort_values(
            ["_priority", "char_count", "chunk_index"], ascending=[True, False, True]
        )
        non_intro_pool = limited.loc[~limited["is_intro"]].sort_values(
            ["_priority", "char_count", "chunk_index"], ascending=[True, False, True]
        )

        max_intro = int(target * max_intro_ratio)
        n_intro = min(len(intro_pool), max_intro)
        n_non_intro = min(len(non_intro_pool), target - n_intro)

        selected = pd.concat([intro_pool.head(n_intro), non_intro_pool.head(n_non_intro)])

        remaining = target - len(selected)
        if remaining > 0:
            filler = limited.loc[~limited.index.isin(selected.index)].sort_values(
                ["_priority", "char_count", "chunk_index"], ascending=[True, False, True]
            )
            selected = pd.concat([selected, filler.head(remaining)])

        selected = selected.head(target)
        print(
            f"[sampler] {domain}: pool={len(limited):,} -> "
            f"sampled={len(selected)} (intro={int(selected['is_intro'].sum())})"
        )
        sampled_parts.append(selected)

    result = pd.concat(sampled_parts, ignore_index=False).drop(columns=["_priority", "is_intro"])
    print(f"\n[sampler] Tong sampled: {len(result):,} chunks")
    return result


def sanity_check(df_raw: pd.DataFrame, df_filtered: pd.DataFrame, df_sampled: pd.DataFrame) -> None:
    """Kiem chung tu dong. Raise neu loi nghiem trong."""
    assert df_filtered["text"].str.contains(r"\{\{|\[\[", regex=True).sum() == 0, (
        "Still have wiki artifact in filtered pool!"
    )
    assert (df_filtered["char_count"] < 300).sum() == 0, "Still have chunk <300 chars in filtered pool!"
    assert df_sampled["domain"].nunique() == 8, "Missing domain in sampled set!"

    intro_ratio = df_sampled.assign(is_intro=df_sampled["section"].fillna("").eq("")).groupby("domain")[
        "is_intro"
    ].mean()
    over_intro = intro_ratio[intro_ratio > 0.20]
    if not over_intro.empty:
        print(f"WARNING: Domain intro ratio >20%: {over_intro.to_dict()}")

    print(f"\nChunks per domain:\n{df_sampled['domain'].value_counts().sort_index().to_string()}")
    print("\nSanity check passed.")


def run() -> None:
    ensure_dirs()
    cfg = load_filter_config()
    print(f"\n{'=' * 55}\n  chunk_filter.py\n{'=' * 55}\n")

    df = pd.read_json(RAW_CHUNKS, lines=True)
    print(f"[load] {len(df):,} chunks\n")

    df_hard = hard_filter(df, cfg)
    df_quality = quality_filter(df_hard, cfg)
    df_quality.to_json(FILTERED_CHUNKS, orient="records", lines=True, force_ascii=False)
    print(f"\nFiltered saved -> {Path(FILTERED_CHUNKS).name}")

    df_sampled = stratified_sampler(df_quality, cfg)
    df_sampled.to_json(SAMPLED_CHUNKS, orient="records", lines=True, force_ascii=False)
    print(f"Sampled saved  -> {Path(SAMPLED_CHUNKS).name}")

    sanity_check(df, df_quality, df_sampled)


if __name__ == "__main__":
    run()
