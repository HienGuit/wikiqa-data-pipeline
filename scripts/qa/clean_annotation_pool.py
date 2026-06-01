"""Compatibility wrapper for the annotation pool artifact.

This script keeps the historical `qa_pairs_canonical_judged_cleaned.jsonl`
artifact in sync with the canonical context-cleaned judged dataset so older
notebooks and ad-hoc exports do not break.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import (  # noqa: E402
    QA_ANNOTATION_POOL,
    QA_ANNOTATION_POOL_COMPAT_REPORT,
    QA_ANNOTATION_POOL_LEGACY,
    QA_ANNOTATION_POOL_LEGACY_REJECTS,
    ensure_dirs,
)


def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def count_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def main() -> None:
    ensure_dirs()
    QA_ANNOTATION_POOL_LEGACY.write_text(load_text(QA_ANNOTATION_POOL), encoding="utf-8")
    QA_ANNOTATION_POOL_LEGACY_REJECTS.write_text("", encoding="utf-8")

    report = {
        "mode": "compatibility_copy",
        "source_annotation_pool": str(QA_ANNOTATION_POOL),
        "legacy_output_path": str(QA_ANNOTATION_POOL_LEGACY),
        "legacy_rejects_path": str(QA_ANNOTATION_POOL_LEGACY_REJECTS),
        "copied_rows": count_rows(QA_ANNOTATION_POOL_LEGACY),
        "notes": [
            "This script no longer performs a separate cleaning pass.",
            "Use scripts/qa/retro_clean_context.py for canonical context cleaning.",
            "Use scripts/qa/finalize_qa_dataset.py to refresh downstream artifacts.",
        ],
    }
    QA_ANNOTATION_POOL_COMPAT_REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
