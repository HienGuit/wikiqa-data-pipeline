"""Build explicit Gemini/DeepSeek annotation keys and agreement reports."""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import (  # noqa: E402
    QA_CANONICAL_JUDGED_DEEPSEEK_V4_FLASH_CONTEXT_CLEANED,
    QA_CANONICAL_JUDGED_GEMINI31_FLASH_LITE_CONTEXT_CLEANED,
    QA_HUMAN_VERIFICATION_TASK1,
    QA_HUMAN_VERIFICATION_TASK2,
    QA_JUDGE_FULL_FLASH_SUMMARY,
    QA_JUDGED_GEMINI31_FLASH_LITE_SUMMARY,
    QA_REPORTS_DIR,
    ensure_dirs,
)

TASK1_GEMINI_KEY = QA_REPORTS_DIR / "human_verification_task1_quality_difficulty_100_key_gemini31_flash_lite.jsonl"
TASK1_DEEPSEEK_KEY = QA_REPORTS_DIR / "human_verification_task1_quality_difficulty_100_key_deepseek_v4_flash.jsonl"
TASK1_AGREEMENT = QA_REPORTS_DIR / "human_verification_task1_quality_difficulty_100_judge_agreement.jsonl"

TASK2_GEMINI_KEY = QA_REPORTS_DIR / "human_verification_task2_inferential_validity_50_key_gemini31_flash_lite.jsonl"
TASK2_DEEPSEEK_KEY = QA_REPORTS_DIR / "human_verification_task2_inferential_validity_50_key_deepseek_v4_flash.jsonl"
TASK2_AGREEMENT = QA_REPORTS_DIR / "human_verification_task2_inferential_validity_50_judge_agreement.jsonl"

REPORT_PATH = QA_REPORTS_DIR / "judge_agreement_mapping_report.json"


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_text(value: Any) -> str:
    text = "" if value is None else str(value)
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def sample_key(row: Dict[str, Any]) -> Tuple[str, str, str, str]:
    return (
        str(row.get("chunk_id", "")),
        str(row.get("reasoning_type", "")),
        normalize_text(row.get("question", "")),
        normalize_text(row.get("answer", "")),
    )


def bucket_counts(rows: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, int]]:
    quality = Counter()
    difficulty = Counter()
    inferential = Counter()
    reasoning = Counter()
    for row in rows:
        quality[str(row.get("quality_band", ""))] += 1
        difficulty[str(row.get("difficulty_band", ""))] += 1
        inferential[str(row.get("inferential_validity_band", ""))] += 1
        reasoning[str(row.get("reasoning_type", ""))] += 1
    return {
        "quality_band": dict(quality),
        "difficulty_band": dict(difficulty),
        "inferential_validity_band": dict(inferential),
        "reasoning_type": dict(reasoning),
    }


def build_task1_key_row(sample: Dict[str, Any], judge_row: Dict[str, Any] | None, model_name: str) -> Dict[str, Any]:
    return {
        "sample_id": sample.get("sample_id"),
        "chunk_id": sample.get("chunk_id"),
        "reasoning_type": sample.get("reasoning_type"),
        "judge_model": model_name,
        "status": "matched" if judge_row else "missing",
        "quality_band_ref": judge_row.get("quality_band", "") if judge_row else "",
        "difficulty_band_ref": judge_row.get("difficulty_band", "") if judge_row else "",
    }


def build_task2_key_row(sample: Dict[str, Any], judge_row: Dict[str, Any] | None, model_name: str) -> Dict[str, Any]:
    return {
        "sample_id": sample.get("sample_id"),
        "chunk_id": sample.get("chunk_id"),
        "reasoning_type": sample.get("reasoning_type"),
        "judge_model": model_name,
        "status": "matched" if judge_row else "missing",
        "inferential_validity_band_ref": judge_row.get("inferential_validity_band", "") if judge_row else "",
    }


def build_task1_agreement_row(
    sample: Dict[str, Any],
    gemini_row: Dict[str, Any] | None,
    deepseek_row: Dict[str, Any] | None,
) -> Dict[str, Any]:
    comparable = bool(gemini_row and deepseek_row)
    return {
        "sample_id": sample.get("sample_id"),
        "chunk_id": sample.get("chunk_id"),
        "reasoning_type": sample.get("reasoning_type"),
        "gemini_status": "matched" if gemini_row else "missing",
        "deepseek_status": "matched" if deepseek_row else "missing",
        "gemini_quality_band": gemini_row.get("quality_band", "") if gemini_row else "",
        "gemini_difficulty_band": gemini_row.get("difficulty_band", "") if gemini_row else "",
        "deepseek_quality_band": deepseek_row.get("quality_band", "") if deepseek_row else "",
        "deepseek_difficulty_band": deepseek_row.get("difficulty_band", "") if deepseek_row else "",
        "comparable": comparable,
        "quality_agree": comparable and gemini_row.get("quality_band") == deepseek_row.get("quality_band"),
        "difficulty_agree": comparable and gemini_row.get("difficulty_band") == deepseek_row.get("difficulty_band"),
        "joint_agree": comparable
        and gemini_row.get("quality_band") == deepseek_row.get("quality_band")
        and gemini_row.get("difficulty_band") == deepseek_row.get("difficulty_band"),
    }


def build_task2_agreement_row(
    sample: Dict[str, Any],
    gemini_row: Dict[str, Any] | None,
    deepseek_row: Dict[str, Any] | None,
) -> Dict[str, Any]:
    comparable = bool(gemini_row and deepseek_row)
    return {
        "sample_id": sample.get("sample_id"),
        "chunk_id": sample.get("chunk_id"),
        "reasoning_type": sample.get("reasoning_type"),
        "gemini_status": "matched" if gemini_row else "missing",
        "deepseek_status": "matched" if deepseek_row else "missing",
        "gemini_inferential_validity_band": gemini_row.get("inferential_validity_band", "") if gemini_row else "",
        "deepseek_inferential_validity_band": deepseek_row.get("inferential_validity_band", "")
        if deepseek_row
        else "",
        "comparable": comparable,
        "inferential_validity_agree": comparable
        and gemini_row.get("inferential_validity_band") == deepseek_row.get("inferential_validity_band"),
    }


def summarize_task1(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    comparable = [row for row in rows if row["comparable"]]
    return {
        "rows": len(rows),
        "gemini_matched": sum(row["gemini_status"] == "matched" for row in rows),
        "deepseek_matched": sum(row["deepseek_status"] == "matched" for row in rows),
        "comparable_rows": len(comparable),
        "quality_agreement": sum(row["quality_agree"] for row in comparable),
        "difficulty_agreement": sum(row["difficulty_agree"] for row in comparable),
        "joint_agreement": sum(row["joint_agree"] for row in comparable),
        "joint_confusions": dict(
            Counter(
                f"{row['gemini_quality_band']}|{row['gemini_difficulty_band']} -> "
                f"{row['deepseek_quality_band']}|{row['deepseek_difficulty_band']}"
                for row in comparable
            )
        ),
    }


def summarize_task2(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    comparable = [row for row in rows if row["comparable"]]
    return {
        "rows": len(rows),
        "gemini_matched": sum(row["gemini_status"] == "matched" for row in rows),
        "deepseek_matched": sum(row["deepseek_status"] == "matched" for row in rows),
        "comparable_rows": len(comparable),
        "inferential_validity_agreement": sum(row["inferential_validity_agree"] for row in comparable),
        "confusions": dict(
            Counter(
                f"{row['gemini_inferential_validity_band']} -> {row['deepseek_inferential_validity_band']}"
                for row in comparable
            )
        ),
    }


def main() -> None:
    ensure_dirs()

    task1_samples = load_jsonl(QA_HUMAN_VERIFICATION_TASK1)
    task2_samples = load_jsonl(QA_HUMAN_VERIFICATION_TASK2)
    gemini_rows = load_jsonl(QA_CANONICAL_JUDGED_GEMINI31_FLASH_LITE_CONTEXT_CLEANED)
    deepseek_rows = load_jsonl(QA_CANONICAL_JUDGED_DEEPSEEK_V4_FLASH_CONTEXT_CLEANED)

    gemini_by_key = {sample_key(row): row for row in gemini_rows}
    deepseek_by_key = {sample_key(row): row for row in deepseek_rows}

    task1_gemini_key: List[Dict[str, Any]] = []
    task1_deepseek_key: List[Dict[str, Any]] = []
    task1_agreement: List[Dict[str, Any]] = []
    for sample in task1_samples:
        key = sample_key(sample)
        gemini_row = gemini_by_key.get(key)
        deepseek_row = deepseek_by_key.get(key)
        task1_gemini_key.append(build_task1_key_row(sample, gemini_row, "gemini-3.1-flash-lite"))
        task1_deepseek_key.append(build_task1_key_row(sample, deepseek_row, "deepseek-v4-flash"))
        task1_agreement.append(build_task1_agreement_row(sample, gemini_row, deepseek_row))

    task2_gemini_key: List[Dict[str, Any]] = []
    task2_deepseek_key: List[Dict[str, Any]] = []
    task2_agreement: List[Dict[str, Any]] = []
    for sample in task2_samples:
        key = sample_key(sample)
        gemini_row = gemini_by_key.get(key)
        deepseek_row = deepseek_by_key.get(key)
        task2_gemini_key.append(build_task2_key_row(sample, gemini_row, "gemini-3.1-flash-lite"))
        task2_deepseek_key.append(build_task2_key_row(sample, deepseek_row, "deepseek-v4-flash"))
        task2_agreement.append(build_task2_agreement_row(sample, gemini_row, deepseek_row))

    write_jsonl(TASK1_GEMINI_KEY, task1_gemini_key)
    write_jsonl(TASK1_DEEPSEEK_KEY, task1_deepseek_key)
    write_jsonl(TASK1_AGREEMENT, task1_agreement)
    write_jsonl(TASK2_GEMINI_KEY, task2_gemini_key)
    write_jsonl(TASK2_DEEPSEEK_KEY, task2_deepseek_key)
    write_jsonl(TASK2_AGREEMENT, task2_agreement)

    report = {
        "artifacts": {
            "task1_gemini_key": str(TASK1_GEMINI_KEY),
            "task1_deepseek_key": str(TASK1_DEEPSEEK_KEY),
            "task1_agreement": str(TASK1_AGREEMENT),
            "task2_gemini_key": str(TASK2_GEMINI_KEY),
            "task2_deepseek_key": str(TASK2_DEEPSEEK_KEY),
            "task2_agreement": str(TASK2_AGREEMENT),
            "gemini_summary": str(QA_JUDGED_GEMINI31_FLASH_LITE_SUMMARY),
            "deepseek_summary": str(QA_JUDGE_FULL_FLASH_SUMMARY),
        },
        "gemini_distribution_full_archive": json.loads(
            QA_JUDGED_GEMINI31_FLASH_LITE_SUMMARY.read_text(encoding="utf-8")
        )["bucket_counts"],
        "gemini_distribution_restored_context_cleaned": bucket_counts(gemini_rows),
        "deepseek_distribution_context_cleaned": bucket_counts(deepseek_rows),
        "task1": summarize_task1(task1_agreement),
        "task2": summarize_task2(task2_agreement),
    }
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
