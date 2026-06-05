"""Read-only final schema and answer-in-context validation for release datasets."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import (  # noqa: E402
    QA_THREE_WAY_FINAL_VALIDATION_REPORT,
    QA_THREE_WAY_READY,
    ensure_dirs,
)
from src.qa.release_validation import validate_release_row  # noqa: E402

ALLOWED_FINAL_BUCKETS = {"extraction", "bridge", "multi-sentence"}


def repo_rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Final read-only validation for a release-ready QA dataset.")
    parser.add_argument("--input", default=str(QA_THREE_WAY_READY), help="Input JSONL dataset to validate.")
    parser.add_argument(
        "--report",
        default=str(QA_THREE_WAY_FINAL_VALIDATION_REPORT),
        help="Path to write the validation report JSON.",
    )
    parser.add_argument(
        "--invalid-output",
        default="",
        help="Optional JSONL path for invalid rows. If omitted, invalid rows are only summarized in the report.",
    )
    return parser


def main() -> None:
    ensure_dirs()
    args = build_parser().parse_args()
    input_path = Path(args.input)
    report_path = Path(args.report)
    invalid_output = Path(args.invalid_output) if args.invalid_output else None

    rows = load_jsonl(input_path)
    invalid_rows: List[Dict[str, Any]] = []
    error_counts = Counter()

    for index, row in enumerate(rows, start=1):
        is_valid, error = validate_release_row(row, allowed_final_buckets=ALLOWED_FINAL_BUCKETS)
        if not is_valid:
            error_counts[error] += 1
            invalid_rows.append(
                {
                    "source_index": index,
                    "error": error,
                    "chunk_id": row.get("chunk_id"),
                    "final_reasoning_bucket": row.get("final_reasoning_bucket"),
                    "question": row.get("question"),
                    "answer": row.get("answer"),
                }
            )

    if invalid_output is not None:
        write_jsonl(invalid_output, invalid_rows)

    report = {
        "input_path": repo_rel(input_path),
        "validated_rows": len(rows),
        "invalid_rows": len(invalid_rows),
        "error_counts": dict(sorted(error_counts.items())),
        "invalid_output_path": repo_rel(invalid_output) if invalid_output is not None else "",
        "status": "pass" if not invalid_rows else "fail",
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))

    if invalid_rows:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
