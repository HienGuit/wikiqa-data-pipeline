"""Shared release-validation helpers for final QA artifacts."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Tuple

REQUIRED_RELEASE_FIELDS = (
    "chunk_id",
    "title",
    "domain",
    "context",
    "question",
    "answer",
    "final_reasoning_bucket",
)


def has_required_fields(row: Dict[str, Any], required_fields: Iterable[str] = REQUIRED_RELEASE_FIELDS) -> bool:
    return all(str(row.get(field, "")).strip() for field in required_fields)


def answer_in_context(row: Dict[str, Any]) -> bool:
    answer = str(row.get("answer", "")).strip()
    context = str(row.get("context", ""))
    return bool(answer) and answer in context


def validate_release_row(
    row: Dict[str, Any],
    *,
    required_fields: Iterable[str] = REQUIRED_RELEASE_FIELDS,
    allowed_final_buckets: set[str] | None = None,
) -> Tuple[bool, str]:
    if not has_required_fields(row, required_fields):
        return False, "missing_required_fields"
    if not answer_in_context(row):
        return False, "answer_not_in_context"

    if allowed_final_buckets is not None:
        bucket = str(row.get("final_reasoning_bucket", "")).strip()
        if bucket not in allowed_final_buckets:
            return False, "invalid_final_reasoning_bucket"

    return True, ""
