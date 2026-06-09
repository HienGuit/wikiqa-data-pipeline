"""Build a three-way QA dataset: extraction, bridge, and multi-sentence."""

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
    QA_CANONICAL_ANNOTATED_CONTEXT_CLEANED,
    QA_REPORTS_DIR,
    QA_THREE_WAY_ANALYSIS,
    QA_THREE_WAY_BRIDGE,
    QA_THREE_WAY_EXTRACTION,
    QA_THREE_WAY_MULTI_SENTENCE,
    QA_THREE_WAY_READY,
    QA_THREE_WAY_REPORT,
    ensure_dirs,
)
from src.qa.release_schema import (  # noqa: E402
    build_release_base_row,
    project_analysis_release_row,
    project_public_release_row,
)

ALLOWED_QUALITY_BANDS = {"usable", "strong"}
INFERENTIAL_VALID_BANDS = {"usable", "strong"}


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def assign_bucket(row: Dict[str, Any]) -> str | None:
    quality_band = str(row.get("quality_band", "")).strip().lower()
    if quality_band not in ALLOWED_QUALITY_BANDS:
        return None

    reasoning_type = str(row.get("reasoning_type", "")).strip()
    if reasoning_type == "extraction":
        return "extraction"

    if reasoning_type != "multi-sentence":
        return None

    inferential_band = str(row.get("inferential_validity_band", "")).strip().lower()
    if inferential_band == "weak":
        return "bridge"
    if inferential_band in INFERENTIAL_VALID_BANDS:
        return "multi-sentence"
    return None


def merge_succinct_into_context(row: Dict[str, Any]) -> str:
    succinct_context = str(row.get("succinct_context", "") or "").strip()
    context = str(row.get("context", "") or "").strip()
    if not succinct_context:
        return context
    if context.startswith(succinct_context):
        return context
    return f"{succinct_context}\n\n{context}" if context else succinct_context


def to_release_rows(row: Dict[str, Any], bucket: str) -> tuple[Dict[str, Any], Dict[str, Any]]:
    base_row = build_release_base_row(
        row,
        context=merge_succinct_into_context(row),
        section=str(row.get("section", "") or ""),
        bucket=bucket,
    )
    return project_public_release_row(base_row), project_analysis_release_row(base_row)


def main() -> None:
    ensure_dirs()
    rows = load_jsonl(QA_CANONICAL_ANNOTATED_CONTEXT_CLEANED)

    public_rows: List[Dict[str, Any]] = []
    analysis_rows: List[Dict[str, Any]] = []
    extraction_rows: List[Dict[str, Any]] = []
    bridge_rows: List[Dict[str, Any]] = []
    multi_rows: List[Dict[str, Any]] = []

    filter_counts = Counter()

    for row in rows:
        bucket = assign_bucket(row)
        if not bucket:
            quality_band = str(row.get("quality_band", "")).strip().lower()
            reasoning_type = str(row.get("reasoning_type", "")).strip()
            inferential_band = str(row.get("inferential_validity_band", "")).strip().lower()
            if quality_band not in ALLOWED_QUALITY_BANDS:
                filter_counts["quality_band_weak_or_missing"] += 1
            elif reasoning_type == "multi-sentence" and inferential_band not in {"weak", *INFERENTIAL_VALID_BANDS}:
                filter_counts["unsupported_inferential_validity"] += 1
            else:
                filter_counts["unsupported_reasoning_type"] += 1
            continue

        public_row, analysis_row = to_release_rows(row, bucket)
        public_rows.append(public_row)
        analysis_rows.append(analysis_row)

        if bucket == "extraction":
            extraction_rows.append(public_row)
        elif bucket == "bridge":
            bridge_rows.append(public_row)
        else:
            multi_rows.append(public_row)

    public_rows.sort(
        key=lambda row: (
            row.get("final_reasoning_bucket", ""),
            str(row.get("domain", "")),
            str(row.get("chunk_id", "")),
            str(row.get("question", "")),
        )
    )
    analysis_rows.sort(
        key=lambda row: (
            row.get("final_reasoning_bucket", ""),
            str(row.get("domain", "")),
            str(row.get("chunk_id", "")),
            str(row.get("question", "")),
        )
    )

    write_jsonl(QA_THREE_WAY_READY, public_rows)
    write_jsonl(QA_THREE_WAY_ANALYSIS, analysis_rows)
    write_jsonl(QA_THREE_WAY_EXTRACTION, extraction_rows)
    write_jsonl(QA_THREE_WAY_BRIDGE, bridge_rows)
    write_jsonl(QA_THREE_WAY_MULTI_SENTENCE, multi_rows)

    report = {
        "input_path": str(QA_CANONICAL_ANNOTATED_CONTEXT_CLEANED),
        "output_path": str(QA_THREE_WAY_READY),
        "analysis_output_path": str(QA_THREE_WAY_ANALYSIS),
        "bucket_outputs": {
            "extraction": str(QA_THREE_WAY_EXTRACTION),
            "bridge": str(QA_THREE_WAY_BRIDGE),
            "multi_sentence": str(QA_THREE_WAY_MULTI_SENTENCE),
        },
        "source_rows": len(rows),
        "kept_rows": len(public_rows),
        "filtered_rows": len(rows) - len(public_rows),
        "filter_counts": dict(sorted(filter_counts.items())),
        "bucket_counts": {
            "extraction": len(extraction_rows),
            "bridge": len(bridge_rows),
            "multi_sentence": len(multi_rows),
        },
        "quality_band_distribution": dict(
            sorted(Counter(str(row.get("quality_band", "")).strip().lower() for row in analysis_rows).items())
        ),
        "original_reasoning_type_distribution": dict(
            sorted(Counter(str(row.get("reasoning_type", "")).strip() for row in analysis_rows).items())
        ),
        "public_schema_fields": list(public_rows[0].keys()) if public_rows else [],
        "analysis_schema_fields": list(analysis_rows[0].keys()) if analysis_rows else [],
    }

    QA_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    QA_THREE_WAY_REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
