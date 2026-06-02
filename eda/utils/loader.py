"""EDA loaders for chunk and QA datasets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pandas as pd

from src.config import QA_THREE_WAY_READY, RAW_CHUNKS

REQUIRED_CHUNK_COLUMNS = {"chunk_id", "title", "domain", "section", "text"}
REQUIRED_QA_COLUMNS = {
    "chunk_id",
    "title",
    "domain",
    "section",
    "context",
    "question",
    "answer",
}


def simple_token_len(text: str) -> int:
    """Return a lightweight whitespace token count for EDA summaries."""

    cleaned = " ".join(str(text or "").split())
    if not cleaned:
        return 0
    return len(cleaned.split(" "))


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


def load_table_frame(path: str | Path, required_columns: Iterable[str] | None = None) -> pd.DataFrame:
    file_path = Path(path)
    suffix = file_path.suffix.lower()
    if suffix == ".jsonl":
        return load_jsonl_frame(file_path, required_columns)
    if suffix == ".parquet":
        df = pd.read_parquet(file_path)
        if required_columns:
            require_columns(df, required_columns)
        return df
    raise ValueError(f"Unsupported dataset format: {file_path}")


def load_chunks(path: str | Path = RAW_CHUNKS) -> pd.DataFrame:
    """Load chunk JSONL as a DataFrame with common derived columns."""

    file_path = Path(path)
    df = load_jsonl_frame(file_path, REQUIRED_CHUNK_COLUMNS)

    if "char_count" not in df.columns:
        df["char_count"] = df["text"].fillna("").str.len()

    df["is_intro"] = df["section"].fillna("").eq("")
    print(f"Loaded {len(df):,} chunks from {file_path}")
    return df


def load_qa_dataset(
    path: str | Path = QA_THREE_WAY_READY, required_columns: Iterable[str] | None = None
) -> pd.DataFrame:
    """Load a QA dataset with common derived text-length columns."""

    file_path = Path(path)
    columns = set(REQUIRED_QA_COLUMNS)
    if required_columns:
        columns.update(required_columns)

    df = load_table_frame(file_path, columns)

    for field in ("context", "question", "answer"):
        text = df[field].fillna("").astype(str)
        df[f"{field}_char_len"] = text.str.len()
        df[f"{field}_token_len"] = text.map(simple_token_len)

    if "section" in df.columns:
        df["section"] = df["section"].fillna("").replace({"": "Giới thiệu"})

    print(f"Loaded {len(df):,} QA rows from {file_path}")
    return df
