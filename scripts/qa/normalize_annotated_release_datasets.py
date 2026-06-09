"""Normalize annotated datasets into release-style artifacts."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import (  # noqa: E402
    QA_ANNOTATED_RELEASE_NORMALIZATION_REPORT,
    QA_CANONICAL_ANNOTATED_CONTEXT_CLEANED,
    QA_CANONICAL_ANNOTATED_RELEASE,
    ensure_dirs,
)
from src.qa.release_validation import validate_release_row  # noqa: E402

INTRO_SECTION_LABEL = "Giá»›i thiá»‡u"
DROP_FIELDS = {
    "succinct_context",
    "reasoning_log",
    "ablation_test_log",
    "is_valid",
    "error",
    "judge_model",
}


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def merge_succinct_into_context(row: Dict[str, Any]) -> str:
    succinct_context = str(row.get("succinct_context", "") or "").strip()
    context = str(row.get("context", "") or "").strip()
    if not succinct_context:
        return context
    if context.startswith(succinct_context):
        return context
    return f"{succinct_context}\n\n{context}" if context else succinct_context


def derive_final_reasoning_bucket(row: Dict[str, Any]) -> str:
    existing = str(row.get("final_reasoning_bucket", "") or "").strip()
    if existing:
        return existing
    reasoning_type = str(row.get("reasoning_type", "") or "").strip()
    inferential_band = str(row.get("inferential_validity_band", "") or "").strip().lower()
    if reasoning_type == "extraction":
        return "extraction"
    if reasoning_type == "multi-sentence" and inferential_band == "weak":
        return "bridge"
    if reasoning_type == "multi-sentence" and inferential_band in {"usable", "strong"}:
        return "multi-sentence"
    return ""


def normalize_row(row: Dict[str, Any], annotator_model: str) -> Dict[str, Any]:
    normalized = {key: value for key, value in row.items() if key not in DROP_FIELDS}
    normalized["section"] = str(row.get("section", "") or "").strip() or INTRO_SECTION_LABEL
    normalized["context"] = merge_succinct_into_context(row)
    normalized["inferential_validity_band"] = (
        str(row.get("inferential_validity_band", "") or "").strip().lower()
        or ("weak" if str(row.get("reasoning_type", "")).strip() == "extraction" else "")
    )
    normalized["final_reasoning_bucket"] = derive_final_reasoning_bucket(normalized)
    normalized["annotator_model"] = annotator_model
    return normalized


def normalize_dataset(input_path: Path, output_path: Path, annotator_model: str) -> Dict[str, Any]:
    source_rows = load_jsonl(input_path)
    normalized_rows = [normalize_row(row, annotator_model) for row in source_rows]

    error_counts = Counter()
    invalid_examples: List[Dict[str, Any]] = []
    intro_sections = 0

    for index, row in enumerate(normalized_rows, start=1):
        if row.get("section") == INTRO_SECTION_LABEL:
            intro_sections += 1
        is_valid, error = validate_release_row(row)
        if not is_valid:
            error_counts[error] += 1
            if len(invalid_examples) < 10:
                invalid_examples.append(
                    {
                        "source_index": index,
                        "error": error,
                        "chunk_id": row.get("chunk_id"),
                        "question": row.get("question"),
                        "answer": row.get("answer"),
                    }
                )

    if error_counts:
        raise RuntimeError(
            f"Normalized release dataset failed validation for {output_path.name}: {dict(error_counts)}"
        )

    write_jsonl(output_path, normalized_rows)
    return {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "rows": len(normalized_rows),
        "intro_section_rows": intro_sections,
        "removed_fields": sorted(DROP_FIELDS),
        "annotator_model": annotator_model,
        "validation_status": "pass",
        "invalid_examples": invalid_examples,
        "field_presence": {
            "has_succinct_context": any("succinct_context" in row for row in normalized_rows),
            "has_reasoning_log": any("reasoning_log" in row for row in normalized_rows),
            "has_ablation_test_log": any("ablation_test_log" in row for row in normalized_rows),
            "has_annotator_model": all(
                bool(str(row.get("annotator_model", "")).strip()) for row in normalized_rows
            ),
        },
        "reasoning_type_distribution": dict(
            sorted(Counter(str(row.get("reasoning_type", "")) for row in normalized_rows).items())
        ),
        "quality_band_distribution": dict(
            sorted(Counter(str(row.get("quality_band", "")) for row in normalized_rows).items())
        ),
    }


def main() -> None:
    ensure_dirs()
    report = {
        "gemini_canonical_release": normalize_dataset(
            input_path=QA_CANONICAL_ANNOTATED_CONTEXT_CLEANED,
            output_path=QA_CANONICAL_ANNOTATED_RELEASE,
            annotator_model="gemini-3.1-flash-lite",
        ),
    }
    QA_ANNOTATED_RELEASE_NORMALIZATION_REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
