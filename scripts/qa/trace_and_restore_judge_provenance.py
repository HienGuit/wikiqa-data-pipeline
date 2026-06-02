"""Trace judge provenance and restore a Gemini judged artifact side-by-side."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import (  # noqa: E402
    QA_CANONICAL,
    QA_CANONICAL_CONTEXT_CLEANED,
    QA_CANONICAL_JUDGED,
    QA_CANONICAL_JUDGED_CONTEXT_CLEANED,
    QA_CANONICAL_JUDGED_DEEPSEEK_V4_FLASH,
    QA_CANONICAL_JUDGED_DEEPSEEK_V4_FLASH_CONTEXT_CLEANED,
    QA_CANONICAL_JUDGED_GEMINI31_FLASH_LITE,
    QA_CANONICAL_JUDGED_GEMINI31_FLASH_LITE_CONTEXT_CLEANED,
    QA_JUDGE_FULL_FLASH_SUMMARY,
    QA_JUDGE_PROVENANCE_REPORT,
    QA_JUDGED_FLASH_LEGACY,
    QA_JUDGED_GEMINI31_FLASH_LITE,
    QA_JUDGED_GEMINI31_FLASH_LITE_SUMMARY,
    ensure_dirs,
)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def norm(text: Any) -> str:
    return " ".join(str(text or "").split()).strip().lower()


def row_key(row: Dict[str, Any]) -> Tuple[str, str, str, str]:
    return (
        str(row.get("chunk_id", "")),
        str(row.get("reasoning_type", "")),
        norm(row.get("question", "")),
        norm(row.get("answer", "")),
    )


def pair_key(row: Dict[str, Any]) -> Tuple[str, str]:
    return (str(row.get("chunk_id", "")), str(row.get("reasoning_type", "")))


def bucket_counts(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, int]]:
    return {
        "quality_band": dict(Counter(str(row.get("quality_band", "")) for row in rows)),
        "difficulty_band": dict(Counter(str(row.get("difficulty_band", "")) for row in rows)),
        "inferential_validity_band": dict(Counter(str(row.get("inferential_validity_band", "")) for row in rows)),
        "reasoning_type": dict(Counter(str(row.get("reasoning_type", "")) for row in rows)),
    }


def compare_labels(a_rows: List[Dict[str, Any]], b_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    a_map = {row_key(row): row for row in a_rows}
    b_map = {row_key(row): row for row in b_rows}
    common = sorted(set(a_map) & set(b_map))
    multi = [key for key in common if key[1] == "multi-sentence"]

    quality_same = sum(1 for key in common if a_map[key].get("quality_band") == b_map[key].get("quality_band"))
    difficulty_same = sum(
        1 for key in common if a_map[key].get("difficulty_band") == b_map[key].get("difficulty_band")
    )
    inferential_same = sum(
        1
        for key in multi
        if a_map[key].get("inferential_validity_band") == b_map[key].get("inferential_validity_band")
    )

    return {
        "common_rows": len(common),
        "common_multi_sentence_rows": len(multi),
        "quality_agreement": quality_same,
        "difficulty_agreement": difficulty_same,
        "inferential_validity_agreement": inferential_same,
    }


def infer_provenance(
    current_total_rows: int,
    current_multi_rows: int,
    current_vs_deepseek: Dict[str, Any],
    current_vs_gemini: Dict[str, Any],
) -> str:
    def is_exact_match(comp: Dict[str, Any]) -> bool:
        return (
            comp["common_rows"] == current_total_rows
            and comp["quality_agreement"] == current_total_rows
            and comp["difficulty_agreement"] == current_total_rows
            and comp["common_multi_sentence_rows"] == current_multi_rows
            and comp["inferential_validity_agreement"] == current_multi_rows
        )

    if is_exact_match(current_vs_deepseek):
        return "deepseek_v4_flash"
    if is_exact_match(current_vs_gemini):
        return "gemini_3_1_flash_lite"
    return "mixed_or_unknown"


def restore_gemini_judged(base_rows: List[Dict[str, Any]], gemini_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    base_by_key = {row_key(row): row for row in base_rows}
    restored: List[Dict[str, Any]] = []

    for judge_row in gemini_rows:
        base = base_by_key.get(row_key(judge_row))
        if not base:
            continue
        merged = dict(base)
        merged["quality_band"] = judge_row.get("quality_band", "")
        merged["difficulty_band"] = judge_row.get("difficulty_band", "")
        if merged.get("reasoning_type") == "multi-sentence":
            merged["inferential_validity_band"] = judge_row.get("inferential_validity_band", "")
        restored.append(merged)

    restored.sort(key=lambda row: (pair_key(row), norm(row.get("question", ""))))
    return restored


def restore_gemini_judged_cleaned(
    cleaned_rows: List[Dict[str, Any]], gemini_rows: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    cleaned_by_key = {row_key(row): row for row in cleaned_rows}
    restored: List[Dict[str, Any]] = []

    for judge_row in gemini_rows:
        base = cleaned_by_key.get(row_key(judge_row))
        if not base:
            continue
        merged = dict(base)
        merged["quality_band"] = judge_row.get("quality_band", "")
        merged["difficulty_band"] = judge_row.get("difficulty_band", "")
        if merged.get("reasoning_type") == "multi-sentence":
            merged["inferential_validity_band"] = judge_row.get("inferential_validity_band", "")
        restored.append(merged)

    restored.sort(key=lambda row: (pair_key(row), norm(row.get("question", ""))))
    return restored


def main() -> None:
    ensure_dirs()

    canonical_rows = load_jsonl(QA_CANONICAL)
    canonical_cleaned_rows = load_jsonl(QA_CANONICAL_CONTEXT_CLEANED)
    canonical_judged_rows = load_jsonl(QA_CANONICAL_JUDGED)
    canonical_judged_cleaned_rows = load_jsonl(QA_CANONICAL_JUDGED_CONTEXT_CLEANED)
    deepseek_rows = load_jsonl(QA_JUDGED_FLASH_LEGACY)
    gemini_rows = load_jsonl(QA_JUDGED_GEMINI31_FLASH_LITE)

    restored_gemini = restore_gemini_judged(canonical_rows, gemini_rows)
    restored_gemini_cleaned = restore_gemini_judged_cleaned(canonical_cleaned_rows, gemini_rows)

    write_jsonl(QA_CANONICAL_JUDGED_GEMINI31_FLASH_LITE, restored_gemini)
    write_jsonl(QA_CANONICAL_JUDGED_GEMINI31_FLASH_LITE_CONTEXT_CLEANED, restored_gemini_cleaned)

    current_vs_deepseek = compare_labels(canonical_judged_cleaned_rows, deepseek_rows)
    current_vs_gemini = compare_labels(canonical_judged_cleaned_rows, gemini_rows)
    current_multi_rows = sum(
        1 for row in canonical_judged_cleaned_rows if row.get("reasoning_type") == "multi-sentence"
    )
    current_provenance = infer_provenance(
        current_total_rows=len(canonical_judged_cleaned_rows),
        current_multi_rows=current_multi_rows,
        current_vs_deepseek=current_vs_deepseek,
        current_vs_gemini=current_vs_gemini,
    )

    provenance = {
        "artifacts": {
            "canonical_judged_current": str(QA_CANONICAL_JUDGED),
            "canonical_judged_context_cleaned_current": str(QA_CANONICAL_JUDGED_CONTEXT_CLEANED),
            "canonical_judged_deepseek_parallel": str(QA_CANONICAL_JUDGED_DEEPSEEK_V4_FLASH),
            "canonical_judged_deepseek_parallel_context_cleaned": str(
                QA_CANONICAL_JUDGED_DEEPSEEK_V4_FLASH_CONTEXT_CLEANED
            ),
            "deepseek_archive": str(QA_JUDGED_FLASH_LEGACY),
            "deepseek_summary": str(QA_JUDGE_FULL_FLASH_SUMMARY),
            "gemini_archive": str(QA_JUDGED_GEMINI31_FLASH_LITE),
            "gemini_summary": str(QA_JUDGED_GEMINI31_FLASH_LITE_SUMMARY),
            "restored_gemini_judged": str(QA_CANONICAL_JUDGED_GEMINI31_FLASH_LITE),
            "restored_gemini_judged_context_cleaned": str(QA_CANONICAL_JUDGED_GEMINI31_FLASH_LITE_CONTEXT_CLEANED),
        },
        "counts": {
            "canonical_judged_current": len(canonical_judged_rows),
            "canonical_judged_context_cleaned_current": len(canonical_judged_cleaned_rows),
            "deepseek_archive": len(deepseek_rows),
            "gemini_archive": len(gemini_rows),
            "restored_gemini_judged": len(restored_gemini),
            "restored_gemini_judged_context_cleaned": len(restored_gemini_cleaned),
        },
        "label_comparison": {
            "current_vs_deepseek_archive": current_vs_deepseek,
            "current_vs_gemini_archive": current_vs_gemini,
        },
        "bucket_counts": {
            "current_canonical_judged": bucket_counts(canonical_judged_rows),
            "deepseek_archive": bucket_counts(deepseek_rows),
            "gemini_archive": bucket_counts(gemini_rows),
            "restored_gemini_judged": bucket_counts(restored_gemini),
        },
        "conclusion": {
            "current_canonical_judged_provenance": current_provenance,
            "why": [
                "Current provenance is inferred by exact label agreement against the archive judge outputs on the current cleaned payload.",
                "refresh-derived only refreshes payload fields from canonical QA and does not overwrite judge labels.",
                "parallel DeepSeek and restored Gemini artifacts are kept side-by-side for provenance and agreement analysis.",
            ],
            "gemini_was_promoted_to_canonical": current_provenance == "gemini_3_1_flash_lite",
            "restoration_possible": True,
        },
    }

    QA_JUDGE_PROVENANCE_REPORT.write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(provenance, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
