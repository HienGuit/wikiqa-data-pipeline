"""Schema helpers for public and analysis-ready QA release artifacts."""

from __future__ import annotations

from typing import Any, Dict

INTRO_SECTION_LABEL = "Giới thiệu"

PUBLIC_RELEASE_FIELDS = (
    "chunk_id",
    "domain",
    "title",
    "section",
    "context",
    "question",
    "answer",
    "final_reasoning_bucket",
    "quality_band",
    "inferential_validity_band",
)

ANALYSIS_RELEASE_FIELDS = (
    "chunk_id",
    "domain",
    "title",
    "section",
    "context",
    "reasoning_type",
    "question",
    "answer",
    "quality_band",
    "difficulty_band",
    "inferential_validity_band",
    "final_reasoning_bucket",
)


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def _normalize_required_band(value: Any) -> str:
    return _clean_text(value).lower()


def _normalize_inferential_band(row: Dict[str, Any]) -> str:
    cleaned = _clean_text(row.get("inferential_validity_band")).lower()
    if cleaned:
        return cleaned
    if _clean_text(row.get("reasoning_type")) == "extraction":
        return "weak"
    return ""


def build_release_base_row(row: Dict[str, Any], *, context: str, section: str, bucket: str) -> Dict[str, Any]:
    """Build a normalized intermediate row before public/internal projection."""

    return {
        "chunk_id": _clean_text(row.get("chunk_id")),
        "domain": _clean_text(row.get("domain")),
        "title": _clean_text(row.get("title")),
        "section": _clean_text(section) or INTRO_SECTION_LABEL,
        "context": _clean_text(context),
        "reasoning_type": _clean_text(row.get("reasoning_type")),
        "question": _clean_text(row.get("question")),
        "answer": _clean_text(row.get("answer")),
        "quality_band": _normalize_required_band(row.get("quality_band")),
        "difficulty_band": _normalize_required_band(row.get("difficulty_band")),
        "inferential_validity_band": _normalize_inferential_band(row),
        "final_reasoning_bucket": _clean_text(bucket),
    }


def project_public_release_row(base_row: Dict[str, Any]) -> Dict[str, Any]:
    """Project an intermediate release row into the public HF-ready schema."""

    return {field: base_row.get(field) for field in PUBLIC_RELEASE_FIELDS}


def project_analysis_release_row(base_row: Dict[str, Any]) -> Dict[str, Any]:
    """Project an intermediate release row into the internal analysis schema."""

    return {field: base_row.get(field) for field in ANALYSIS_RELEASE_FIELDS}
