import json
from pathlib import Path

import pandas as pd

from src.config import RAW_CHUNKS


REQUIRED_COLUMNS = {"chunk_id", "title", "domain", "section", "text"}


def load_chunks(path: str | Path = RAW_CHUNKS) -> pd.DataFrame:
    """Đọc file JSONL chunk và trả về DataFrame với các cột chuẩn."""
    path = Path(path)

    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))

    df = pd.DataFrame(records)

    missing_columns = REQUIRED_COLUMNS - set(df.columns)
    if missing_columns:
        missing_str = ", ".join(sorted(missing_columns))
        raise ValueError(f"Thiếu cột bắt buộc trong dữ liệu: {missing_str}")

    if "char_count" not in df.columns:
        df["char_count"] = df["text"].str.len()

    df["is_intro"] = df["section"].fillna("").eq("")

    print(f"Loaded {len(df):,} chunks from {path}")
    return df
