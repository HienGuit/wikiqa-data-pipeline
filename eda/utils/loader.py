"""EDA loaders for chunk and QA JSONL datasets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pandas as pd

from src.config import RAW_CHUNKS


REQUIRED_CHUNK_COLUMNS = {"chunk_id", "title", "domain", "section", "text"}


def load_jsonl_records(path: str | Path) -> list[dict]:
    file_path = Path(path)
    records = []
    with file_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def require_columns(df: pd.DataFrame, required_columns: Iterable[str]) -> None:
    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        missing_str = ", ".join(sorted(missing_columns))
        raise ValueError(f"Thieu cot bat buoc trong du lieu: {missing_str}")


def load_jsonl_frame(path: str | Path, required_columns: Iterable[str] | None = None) -> pd.DataFrame:
    df = pd.DataFrame(load_jsonl_records(path))
    if required_columns:
        require_columns(df, required_columns)
    return df


def load_chunks(path: str | Path = RAW_CHUNKS) -> pd.DataFrame:
    """Load chunk JSONL as a DataFrame with common derived columns."""

    file_path = Path(path)
    df = load_jsonl_frame(file_path, REQUIRED_CHUNK_COLUMNS)

    if "char_count" not in df.columns:
        df["char_count"] = df["text"].fillna("").str.len()

    df["is_intro"] = df["section"].fillna("").eq("")
    print(f"Loaded {len(df):,} chunks from {file_path}")
    return df
