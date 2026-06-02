"""Retro-clean canonical QA datasets after generation and judging."""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import (  # noqa: E402
    QA_CANONICAL,
    QA_CANONICAL_CONTEXT_CLEANED,
    QA_CANONICAL_CONTEXT_CLEANING_REJECTS,
    QA_CANONICAL_JUDGED,
    QA_CANONICAL_JUDGED_CONTEXT_CLEANED,
    QA_CANONICAL_JUDGED_CONTEXT_CLEANING_REJECTS,
    QA_CONTEXT_CLEANING_REPORT,
    ensure_dirs,
)
from src.processing.text_cleaning import clean_article_text, clean_short_text  # noqa: E402
from src.qa.release_validation import validate_release_row  # noqa: E402

SUCCINCT_DANGLING_SUFFIXES = (
    "nơi",
    "khi",
    "sau khi",
    "trong khi",
    "nhưng",
    "và",
    "hoặc",
    "để",
    "với",
    "rằng",
    "trong đó",
)


def load_rows(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_for_key(value: Any) -> str:
    return " ".join(str(value or "").lower().split())


def duplicate_key(row: Dict[str, Any]) -> Tuple[str, str, str]:
    return (
        str(row.get("chunk_id", "")),
        normalize_for_key(row.get("question", "")),
        normalize_for_key(row.get("answer", "")),
    )


def duplicate_priority(row: Dict[str, Any]) -> Tuple[int, int, int]:
    type_rank = 0 if str(row.get("reasoning_type", "")) == "extraction" else 1
    quality_rank = {"strong": 0, "usable": 1, "weak": 2}.get(str(row.get("quality_band", "")), 3)
    difficulty_rank = {"hard": 0, "medium": 1, "easy": 2}.get(str(row.get("difficulty_band", "")), 3)
    return (type_rank, quality_rank, difficulty_rank)


def clean_sample_fields(row: Dict[str, Any]) -> Dict[str, Any]:
    cleaned = dict(row)
    cleaned["context"] = clean_article_text(str(row.get("context", "")))
    cleaned["question"] = clean_short_text(str(row.get("question", "")))
    cleaned["answer"] = clean_short_text(str(row.get("answer", "")))
    cleaned["title"] = clean_short_text(str(row.get("title", "")))
    cleaned["section"] = clean_short_text(str(row.get("section", "")))
    cleaned["domain"] = clean_short_text(str(row.get("domain", "")))
    cleaned["reasoning_log"] = clean_short_text(str(row.get("reasoning_log", "")))
    cleaned["ablation_test_log"] = clean_short_text(str(row.get("ablation_test_log", "")))
    cleaned["succinct_context"] = clean_short_text(str(row.get("succinct_context", "")))
    return cleaned


def is_bad_succinct_context(row: Dict[str, Any]) -> bool:
    succinct = str(row.get("succinct_context", "") or "").strip()
    if not succinct:
        return False
    lowered = succinct.lower()
    answer = str(row.get("answer", "") or "").strip().lower()
    if succinct[-1] in ",:;-":
        return True
    if succinct[-1] not in ".!?…":
        return True
    if any(lowered.endswith(suffix) for suffix in SUCCINCT_DANGLING_SUFFIXES):
        return True
    if answer and answer in lowered:
        return True
    return False


def clean_dataset(*, name: str, input_path: Path, output_path: Path, rejects_path: Path) -> Dict[str, Any]:
    rows = load_rows(input_path)
    reject_counts = Counter()
    rejects: List[Dict[str, Any]] = []
    cleaned_candidates: List[Tuple[int, Dict[str, Any]]] = []

    for index, row in enumerate(rows, start=1):
        is_valid, error = validate_release_row(row)
        if not is_valid:
            reject_counts[error] += 1
            rejects.append({"dataset": name, "source_index": index, "reject_reason": error, "row": row})
            continue

        cleaned_row = clean_sample_fields(row)

        if not cleaned_row["context"].strip():
            reject_counts["empty_cleaned_context"] += 1
            rejects.append(
                {"dataset": name, "source_index": index, "reject_reason": "empty_cleaned_context", "row": row}
            )
            continue
        if not cleaned_row["question"].strip():
            reject_counts["empty_cleaned_question"] += 1
            rejects.append(
                {"dataset": name, "source_index": index, "reject_reason": "empty_cleaned_question", "row": row}
            )
            continue
        if not cleaned_row["answer"].strip():
            reject_counts["empty_cleaned_answer"] += 1
            rejects.append(
                {"dataset": name, "source_index": index, "reject_reason": "empty_cleaned_answer", "row": row}
            )
            continue

        is_valid, error = validate_release_row(cleaned_row)
        if not is_valid:
            mapped_error = "answer_not_in_cleaned_context" if error == "answer_not_in_context" else error
            reject_counts[mapped_error] += 1
            rejects.append({"dataset": name, "source_index": index, "reject_reason": mapped_error, "row": row})
            continue

        if is_bad_succinct_context(cleaned_row):
            reject_counts["bad_succinct_context_after_clean"] += 1
            rejects.append(
                {
                    "dataset": name,
                    "source_index": index,
                    "reject_reason": "bad_succinct_context_after_clean",
                    "row": row,
                }
            )
            continue

        cleaned_candidates.append((index, cleaned_row))

    grouped: Dict[Tuple[str, str, str], List[Tuple[int, Dict[str, Any]]]] = defaultdict(list)
    for index, row in cleaned_candidates:
        grouped[duplicate_key(row)].append((index, row))

    final_rows: List[Dict[str, Any]] = []
    duplicate_examples: List[Dict[str, Any]] = []

    for key, items in grouped.items():
        ranked = sorted(items, key=lambda item: (duplicate_priority(item[1]), item[0]))
        kept_index, kept_row = ranked[0]
        final_rows.append(kept_row)
        if len(ranked) > 1:
            for rejected_index, rejected_row in ranked[1:]:
                reject_counts["duplicate_semantic_cross_type"] += 1
                rejects.append(
                    {
                        "dataset": name,
                        "source_index": rejected_index,
                        "kept_source_index": kept_index,
                        "reject_reason": "duplicate_semantic_cross_type",
                        "row": rejected_row,
                    }
                )
            if len(duplicate_examples) < 10:
                duplicate_examples.append(
                    {
                        "chunk_id": key[0],
                        "question": kept_row.get("question", ""),
                        "answer": kept_row.get("answer", ""),
                        "kept_reasoning_type": kept_row.get("reasoning_type", ""),
                        "dropped_reasoning_types": [row.get("reasoning_type", "") for _, row in ranked[1:]],
                    }
                )

    final_rows.sort(
        key=lambda row: (
            str(row.get("chunk_id", "")),
            str(row.get("reasoning_type", "")),
            normalize_for_key(row.get("question", "")),
        )
    )

    write_jsonl(output_path, final_rows)
    write_jsonl(rejects_path, rejects)

    return {
        "name": name,
        "input_path": str(input_path),
        "output_path": str(output_path),
        "rejects_path": str(rejects_path),
        "source_rows": len(rows),
        "cleaned_rows": len(final_rows),
        "rejected_rows": len(rejects),
        "reject_counts": dict(sorted(reject_counts.items())),
        "reasoning_type_distribution": dict(
            sorted(Counter(row.get("reasoning_type", "") for row in final_rows).items())
        ),
        "duplicate_examples": duplicate_examples,
    }


def main() -> None:
    ensure_dirs()
    report = {
        "canonical": clean_dataset(
            name="canonical",
            input_path=QA_CANONICAL,
            output_path=QA_CANONICAL_CONTEXT_CLEANED,
            rejects_path=QA_CANONICAL_CONTEXT_CLEANING_REJECTS,
        ),
        "judged": clean_dataset(
            name="judged",
            input_path=QA_CANONICAL_JUDGED,
            output_path=QA_CANONICAL_JUDGED_CONTEXT_CLEANED,
            rejects_path=QA_CANONICAL_JUDGED_CONTEXT_CLEANING_REJECTS,
        ),
    }
    QA_CONTEXT_CLEANING_REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
