"""Finalize QA dataset artifacts after context cleaning."""

from __future__ import annotations

import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import (  # noqa: E402
    QA_ARCHIVE_DIR,
    QA_CANONICAL,
    QA_CANONICAL_CONTEXT_CLEANED,
    QA_CANONICAL_JUDGED,
    QA_CANONICAL_JUDGED_CONTEXT_CLEANED,
    QA_CONTEXT_CLEANED_SYNC_REPORT,
    QA_CONTEXT_CLEANING_REPORT,
    QA_DATASET_FINALIZATION_REPORT,
    QA_INFERENTIAL_USABLE_ONLY,
    QA_SPLIT_READY,
    QA_CANONICAL_CONTEXT_CLEANING_REJECTS,
    QA_CANONICAL_JUDGED_CONTEXT_CLEANING_REJECTS,
    ensure_dirs,
)


def load_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def backup_file(path: Path, archive_dir: Path, suffix: str) -> str:
    if not path.exists():
        return ""
    archive_dir.mkdir(parents=True, exist_ok=True)
    backup_path = archive_dir / f"{path.stem}.{suffix}{path.suffix}"
    backup_path.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    return str(backup_path)


def sync_split_from_judged_cleaned() -> dict:
    rows = load_rows(QA_CANONICAL_JUDGED_CONTEXT_CLEANED)
    split_ready_rows = []
    inferential_rows = []

    for row in rows:
        reasoning_type = row.get("reasoning_type")
        inferential_band = row.get("inferential_validity_band", "")
        if reasoning_type == "extraction":
            split_ready_rows.append(row)
        elif reasoning_type == "multi-sentence" and inferential_band == "usable":
            split_ready_rows.append(row)
            inferential_rows.append(row)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_split = backup_file(QA_SPLIT_READY, QA_ARCHIVE_DIR, f"before_context_sync_{timestamp}")
    backup_inferential = backup_file(QA_INFERENTIAL_USABLE_ONLY, QA_ARCHIVE_DIR, f"before_context_sync_{timestamp}")

    write_jsonl(QA_SPLIT_READY, split_ready_rows)
    write_jsonl(QA_INFERENTIAL_USABLE_ONLY, inferential_rows)

    report = {
        "synced_from": str(QA_CANONICAL_JUDGED_CONTEXT_CLEANED),
        "split_ready_path": str(QA_SPLIT_READY),
        "inferential_usable_only_path": str(QA_INFERENTIAL_USABLE_ONLY),
        "backup_split_ready": backup_split,
        "backup_inferential_usable_only": backup_inferential,
        "source_rows": len(rows),
        "split_ready_rows": len(split_ready_rows),
        "inferential_usable_rows": len(inferential_rows),
        "split_ready_reasoning_type": dict(Counter(row.get("reasoning_type", "") for row in split_ready_rows)),
        "inferential_validity_band_source": dict(Counter(row.get("inferential_validity_band", "") for row in rows)),
        "inferential_validity_band_split_ready": dict(
            Counter(row.get("inferential_validity_band", "") for row in split_ready_rows)
        ),
    }
    QA_CONTEXT_CLEANED_SYNC_REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def build_finalization_report(sync_report: dict) -> dict:
    canonical_rows = len(load_rows(QA_CANONICAL))
    canonical_cleaned_rows = len(load_rows(QA_CANONICAL_CONTEXT_CLEANED))
    judged_rows = len(load_rows(QA_CANONICAL_JUDGED))
    judged_cleaned_rows = len(load_rows(QA_CANONICAL_JUDGED_CONTEXT_CLEANED))
    canonical_rejects = len(load_rows(QA_CANONICAL_CONTEXT_CLEANING_REJECTS))
    judged_rejects = len(load_rows(QA_CANONICAL_JUDGED_CONTEXT_CLEANING_REJECTS))
    split_ready_rows = len(load_rows(QA_SPLIT_READY))
    inferential_rows = len(load_rows(QA_INFERENTIAL_USABLE_ONLY))

    report = {
        "finalized_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "artifacts": {
            "canonical": str(QA_CANONICAL),
            "canonical_context_cleaned": str(QA_CANONICAL_CONTEXT_CLEANED),
            "canonical_judged": str(QA_CANONICAL_JUDGED),
            "canonical_judged_context_cleaned": str(QA_CANONICAL_JUDGED_CONTEXT_CLEANED),
            "split_ready": str(QA_SPLIT_READY),
            "inferential_usable_only": str(QA_INFERENTIAL_USABLE_ONLY),
        },
        "counts": {
            "canonical_rows": canonical_rows,
            "canonical_context_cleaned_rows": canonical_cleaned_rows,
            "canonical_context_cleaning_rejects": canonical_rejects,
            "canonical_judged_rows": judged_rows,
            "canonical_judged_context_cleaned_rows": judged_cleaned_rows,
            "canonical_judged_context_cleaning_rejects": judged_rejects,
            "split_ready_rows": split_ready_rows,
            "inferential_usable_only_rows": inferential_rows,
        },
        "reports": {
            "context_cleaning_report": str(QA_CONTEXT_CLEANING_REPORT),
            "context_cleaned_sync_report": str(QA_CONTEXT_CLEANED_SYNC_REPORT),
        },
        "sync_summary": sync_report,
    }
    return report


def main() -> None:
    ensure_dirs()
    sync_report = sync_split_from_judged_cleaned()
    final_report = build_finalization_report(sync_report)
    QA_DATASET_FINALIZATION_REPORT.write_text(json.dumps(final_report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(final_report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
