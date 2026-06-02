"""Promote Gemini judged artifacts to canonical while preserving DeepSeek in parallel."""

from __future__ import annotations

import json
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import (  # noqa: E402
    QA_CANONICAL_JUDGED,
    QA_CANONICAL_JUDGED_CONTEXT_CLEANED,
    QA_CANONICAL_JUDGED_DEEPSEEK_V4_FLASH,
    QA_CANONICAL_JUDGED_DEEPSEEK_V4_FLASH_CONTEXT_CLEANED,
    QA_CANONICAL_JUDGED_GEMINI31_FLASH_LITE,
    QA_CANONICAL_JUDGED_GEMINI31_FLASH_LITE_CONTEXT_CLEANED,
    QA_DATASET_FINALIZATION_REPORT,
    QA_JUDGE_CANONICAL_PROMOTION_REPORT,
    QA_JUDGE_PROVENANCE_REPORT,
    QA_THREE_WAY_FINAL_VALIDATION_REPORT,
    QA_THREE_WAY_REPORT,
    ensure_dirs,
)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)


def count_buckets(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, int]]:
    return {
        "quality_band": dict(Counter(str(row.get("quality_band", "")) for row in rows)),
        "difficulty_band": dict(Counter(str(row.get("difficulty_band", "")) for row in rows)),
        "inferential_validity_band": dict(Counter(str(row.get("inferential_validity_band", "")) for row in rows)),
        "reasoning_type": dict(Counter(str(row.get("reasoning_type", "")) for row in rows)),
    }


def main() -> None:
    ensure_dirs()

    if not QA_CANONICAL_JUDGED_GEMINI31_FLASH_LITE.exists():
        raise FileNotFoundError(f"Missing restored Gemini judged artifact: {QA_CANONICAL_JUDGED_GEMINI31_FLASH_LITE}")
    if not QA_CANONICAL_JUDGED_GEMINI31_FLASH_LITE_CONTEXT_CLEANED.exists():
        raise FileNotFoundError(
            f"Missing restored Gemini cleaned artifact: {QA_CANONICAL_JUDGED_GEMINI31_FLASH_LITE_CONTEXT_CLEANED}"
        )
    if not QA_CANONICAL_JUDGED.exists():
        raise FileNotFoundError(f"Missing current canonical judged artifact: {QA_CANONICAL_JUDGED}")
    if not QA_CANONICAL_JUDGED_CONTEXT_CLEANED.exists():
        raise FileNotFoundError(
            f"Missing current canonical judged cleaned artifact: {QA_CANONICAL_JUDGED_CONTEXT_CLEANED}"
        )

    previous_canonical = load_jsonl(QA_CANONICAL_JUDGED)
    previous_canonical_cleaned = load_jsonl(QA_CANONICAL_JUDGED_CONTEXT_CLEANED)
    gemini_rows = load_jsonl(QA_CANONICAL_JUDGED_GEMINI31_FLASH_LITE)
    gemini_cleaned_rows = load_jsonl(QA_CANONICAL_JUDGED_GEMINI31_FLASH_LITE_CONTEXT_CLEANED)

    copy_file(QA_CANONICAL_JUDGED, QA_CANONICAL_JUDGED_DEEPSEEK_V4_FLASH)
    copy_file(QA_CANONICAL_JUDGED_CONTEXT_CLEANED, QA_CANONICAL_JUDGED_DEEPSEEK_V4_FLASH_CONTEXT_CLEANED)

    copy_file(QA_CANONICAL_JUDGED_GEMINI31_FLASH_LITE, QA_CANONICAL_JUDGED)
    copy_file(QA_CANONICAL_JUDGED_GEMINI31_FLASH_LITE_CONTEXT_CLEANED, QA_CANONICAL_JUDGED_CONTEXT_CLEANED)

    report = {
        "promoted_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "canonical_target": {
            "judged": str(QA_CANONICAL_JUDGED),
            "judged_context_cleaned": str(QA_CANONICAL_JUDGED_CONTEXT_CLEANED),
        },
        "preserved_parallel": {
            "deepseek_judged": str(QA_CANONICAL_JUDGED_DEEPSEEK_V4_FLASH),
            "deepseek_judged_context_cleaned": str(QA_CANONICAL_JUDGED_DEEPSEEK_V4_FLASH_CONTEXT_CLEANED),
            "gemini_source_judged": str(QA_CANONICAL_JUDGED_GEMINI31_FLASH_LITE),
            "gemini_source_judged_context_cleaned": str(QA_CANONICAL_JUDGED_GEMINI31_FLASH_LITE_CONTEXT_CLEANED),
        },
        "counts_before": {
            "canonical_judged": len(previous_canonical),
            "canonical_judged_context_cleaned": len(previous_canonical_cleaned),
        },
        "counts_after": {
            "canonical_judged": len(gemini_rows),
            "canonical_judged_context_cleaned": len(gemini_cleaned_rows),
        },
        "bucket_counts_before": count_buckets(previous_canonical_cleaned),
        "bucket_counts_after": count_buckets(gemini_cleaned_rows),
        "follow_up_reports": {
            "provenance": str(QA_JUDGE_PROVENANCE_REPORT),
            "finalization": str(QA_DATASET_FINALIZATION_REPORT),
            "three_way": str(QA_THREE_WAY_REPORT),
            "three_way_final_validation": str(QA_THREE_WAY_FINAL_VALIDATION_REPORT),
        },
    }
    QA_JUDGE_CANONICAL_PROMOTION_REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
