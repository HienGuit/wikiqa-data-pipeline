"""Build final release metadata reports for provenance and HF-ready publishing."""

from __future__ import annotations

import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import (  # noqa: E402
    FEATURE_MATRIX_FINAL,
    FEATURE_MATRIX_FULL,
    FEATURE_PHASE1_PROVENANCE_JSON,
    FEATURE_PHASE1_PROVENANCE_MD,
    FEATURE_MATRIX_EDA_REPORT,
    FINAL_RELEASE_MANIFEST_JSON,
    FINAL_RELEASE_MANIFEST_MD,
    QA_CANONICAL_JUDGED,
    QA_CANONICAL_JUDGED_CONTEXT_CLEANED,
    QA_CANONICAL_JUDGED_RELEASE,
    QA_HUMAN_VERIFICATION_TASK1,
    QA_HUMAN_VERIFICATION_TASK2,
    QA_REPORTS_DIR,
    QA_THREE_WAY_ANALYSIS,
    QA_THREE_WAY_FINAL_VALIDATION_REPORT,
    QA_THREE_WAY_READY,
    ensure_dirs,
)
from src.qa.release_schema import ANALYSIS_RELEASE_FIELDS, PUBLIC_RELEASE_FIELDS  # noqa: E402

HUMAN_VERIFICATION_BUNDLE_DIR = QA_THREE_WAY_READY.parent / "human_verification_bundle_external_gemini_20260605"
IAA_SUMMARY_MD = HUMAN_VERIFICATION_BUNDLE_DIR / "reports" / "iaa_summary.md"
EDA1_DIR = ROOT / "eda" / "figures" / "02_qa_dataset_eda"
EDA2_DIR = ROOT / "eda" / "figures" / "03_feature_engineering_eda"
EDA1_REPORT = EDA1_DIR / "eda_report.json"

ACTIVE_KNOWLEDGE_SIGNALS = [
    "page_views_rank",
    "site_links_rank",
    "wiki_count_rank",
    "statements_rank",
    "references_rank",
    "knowledge_difficulty",
]
RETAINED_FINAL_KNOWLEDGE_SIGNALS = [
    "page_views_rank",
    "wiki_count_rank",
    "statements_rank",
    "knowledge_difficulty",
]
ACTIVE_OTHER_FEATURES = [
    "structural_features",
    "question_type",
    "answer_type",
    "popularity_source",
]
EXCLUDED_PHASE1_FEATURES = {
    "wiki_level": "Excluded because crawl depth is not reliably preserved in the current raw metadata and would require a separate taxonomy-recovery pass.",
    "linked_entities": "Excluded because API coverage and stability were not strong enough for the final phase-1 feature set.",
}
ANNOTATION_RATIONALE = "Gemini is used as the external annotator because it is separate from the QA generation model."


def _iso_timestamp(path: Path) -> str:
    if not path.exists():
        return ""
    return datetime.fromtimestamp(path.stat().st_mtime).astimezone().isoformat(timespec="seconds")


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _count_jsonl_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def _csv_shape(path: Path) -> tuple[int, list[str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        row_count = sum(1 for _ in reader)
    return row_count, fieldnames


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_markdown(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def _repo_rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path.resolve())


def build_feature_phase1_provenance() -> dict[str, Any]:
    full_rows, full_columns = _csv_shape(FEATURE_MATRIX_FULL)
    final_rows, final_columns = _csv_shape(FEATURE_MATRIX_FINAL)

    payload = {
        "phase": "feature_engineering_phase1",
        "source_dataset_analysis": str(QA_THREE_WAY_ANALYSIS.resolve()),
        "public_release_dataset": str(QA_THREE_WAY_READY.resolve()),
        "feature_matrix_full": str(FEATURE_MATRIX_FULL.resolve()),
        "feature_matrix_final": str(FEATURE_MATRIX_FINAL.resolve()),
        "full_row_count": full_rows,
        "full_column_count": len(full_columns),
        "final_row_count": final_rows,
        "final_column_count": len(final_columns),
        "full_columns": full_columns,
        "final_columns": final_columns,
        "active_knowledge_signals": ACTIVE_KNOWLEDGE_SIGNALS,
        "retained_final_knowledge_signals": RETAINED_FINAL_KNOWLEDGE_SIGNALS,
        "active_other_features": ACTIVE_OTHER_FEATURES,
        "excluded_features": EXCLUDED_PHASE1_FEATURES,
        "timestamps": {
            "feature_matrix_full": _iso_timestamp(FEATURE_MATRIX_FULL),
            "feature_matrix_final": _iso_timestamp(FEATURE_MATRIX_FINAL),
        },
    }
    _write_json(FEATURE_PHASE1_PROVENANCE_JSON, payload)

    md_lines = [
        "# Feature Engineering Phase 1 Provenance",
        "",
        f"- Analysis source dataset: `{_repo_rel(QA_THREE_WAY_ANALYSIS)}`",
        f"- Public release dataset: `{_repo_rel(QA_THREE_WAY_READY)}`",
        f"- Full matrix: `{_repo_rel(FEATURE_MATRIX_FULL)}` ({full_rows:,} rows, {len(full_columns)} columns)",
        f"- Final matrix: `{_repo_rel(FEATURE_MATRIX_FINAL)}` ({final_rows:,} rows, {len(final_columns)} columns)",
        "",
        "## Active Phase-1 Features",
        *[f"- Full-matrix knowledge signal: `{feature}`" for feature in ACTIVE_KNOWLEDGE_SIGNALS],
        *[f"- Retained final-matrix knowledge signal: `{feature}`" for feature in RETAINED_FINAL_KNOWLEDGE_SIGNALS],
        *[f"- Other feature group: `{feature}`" for feature in ACTIVE_OTHER_FEATURES],
        "",
        "## Excluded From Phase 1",
        *[f"- `{feature}`: {reason}" for feature, reason in EXCLUDED_PHASE1_FEATURES.items()],
    ]
    _write_markdown(FEATURE_PHASE1_PROVENANCE_MD, md_lines)
    return payload


def build_final_release_manifest(feature_payload: dict[str, Any]) -> dict[str, Any]:
    validation = _load_json(QA_THREE_WAY_FINAL_VALIDATION_REPORT)
    eda1_report = _load_json(EDA1_REPORT)
    eda2_report = _load_json(FEATURE_MATRIX_EDA_REPORT)
    public_rows = _count_jsonl_rows(QA_THREE_WAY_READY)
    analysis_rows = _count_jsonl_rows(QA_THREE_WAY_ANALYSIS)
    canonical_rows = _count_jsonl_rows(QA_CANONICAL_JUDGED)
    canonical_cleaned_rows = _count_jsonl_rows(QA_CANONICAL_JUDGED_CONTEXT_CLEANED)
    task1_rows = _count_jsonl_rows(QA_HUMAN_VERIFICATION_TASK1)
    task2_rows = _count_jsonl_rows(QA_HUMAN_VERIFICATION_TASK2)
    train_rows = _count_jsonl_rows(ROOT / "data" / "final" / "train.jsonl")
    val_rows = _count_jsonl_rows(ROOT / "data" / "final" / "val.jsonl")
    test_rows = _count_jsonl_rows(ROOT / "data" / "final" / "test.jsonl")

    payload = {
        "external_annotation": {
            "qa_generator": "DeepSeek V4 Flash",
            "automatic_annotator": "Gemini",
            "decision_rationale": ANNOTATION_RATIONALE,
            "deepseek_label_policy": (
                "Labels produced by DeepSeek for evaluation are excluded from the official release pipeline to avoid same-model evaluation bias."
            ),
            "judged_source": str(QA_CANONICAL_JUDGED.resolve()),
            "judged_source_rows": canonical_rows,
            "judged_context_cleaned": str(QA_CANONICAL_JUDGED_CONTEXT_CLEANED.resolve()),
            "judged_context_cleaned_rows": canonical_cleaned_rows,
            "judged_release_normalized": str(QA_CANONICAL_JUDGED_RELEASE.resolve()),
        },
        "release_datasets": {
            "public_final": str(QA_THREE_WAY_READY.resolve()),
            "public_final_rows": public_rows,
            "analysis_source": str(QA_THREE_WAY_ANALYSIS.resolve()),
            "analysis_source_rows": analysis_rows,
            "train_split": str((ROOT / "data" / "final" / "train.jsonl").resolve()),
            "train_split_rows": train_rows,
            "val_split": str((ROOT / "data" / "final" / "val.jsonl").resolve()),
            "val_split_rows": val_rows,
            "test_split": str((ROOT / "data" / "final" / "test.jsonl").resolve()),
            "test_split_rows": test_rows,
            "public_schema_fields": list(PUBLIC_RELEASE_FIELDS),
            "analysis_schema_fields": list(ANALYSIS_RELEASE_FIELDS),
        },
        "validation": {
            "final_validation_report": str(QA_THREE_WAY_FINAL_VALIDATION_REPORT.resolve()),
            "status": validation.get("status", ""),
            "validated_rows": validation.get("validated_rows", 0),
            "invalid_rows": validation.get("invalid_rows", 0),
            "error_counts": validation.get("error_counts", {}),
        },
        "feature_engineering_phase1": {
            "full_matrix": str(FEATURE_MATRIX_FULL.resolve()),
            "final_matrix": str(FEATURE_MATRIX_FINAL.resolve()),
            "provenance_report": str(FEATURE_PHASE1_PROVENANCE_JSON.resolve()),
            "active_knowledge_signals": ACTIVE_KNOWLEDGE_SIGNALS,
            "retained_final_knowledge_signals": RETAINED_FINAL_KNOWLEDGE_SIGNALS,
            "active_other_features": ACTIVE_OTHER_FEATURES,
            "excluded_features": EXCLUDED_PHASE1_FEATURES,
            "full_row_count": feature_payload["full_row_count"],
            "full_column_count": feature_payload["full_column_count"],
            "final_row_count": feature_payload["final_row_count"],
            "final_column_count": feature_payload["final_column_count"],
        },
        "eda": {
            "eda1_dir": str(EDA1_DIR.resolve()),
            "eda1_report": str(EDA1_REPORT.resolve()),
            "eda1_timestamp": _iso_timestamp(EDA1_REPORT),
            "eda1_figure_count": len(eda1_report.get("figures", [])),
            "eda2_dir": str(EDA2_DIR.resolve()),
            "eda2_report": str(FEATURE_MATRIX_EDA_REPORT.resolve()),
            "eda2_timestamp": _iso_timestamp(FEATURE_MATRIX_EDA_REPORT),
            "eda2_figure_count": len(eda2_report.get("figures", [])),
        },
        "human_verification": {
            "bundle_dir": str(HUMAN_VERIFICATION_BUNDLE_DIR.resolve()),
            "task1_rows": task1_rows,
            "task2_rows": task2_rows,
            "iaa_summary": str(IAA_SUMMARY_MD.resolve()),
        },
        "timestamps": {
            "public_final_dataset": _iso_timestamp(QA_THREE_WAY_READY),
            "analysis_source_dataset": _iso_timestamp(QA_THREE_WAY_ANALYSIS),
            "feature_matrix_full": feature_payload["timestamps"]["feature_matrix_full"],
            "feature_matrix_final": feature_payload["timestamps"]["feature_matrix_final"],
            "manifest_built_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        },
    }
    _write_json(FINAL_RELEASE_MANIFEST_JSON, payload)

    md_lines = [
        "# Final Release Manifest",
        "",
        "## External Annotation",
        "- QA generator: `DeepSeek V4 Flash`",
        "- External automatic annotator: `Gemini`",
        f"- Rationale: {ANNOTATION_RATIONALE}",
        "- Labels produced by DeepSeek for evaluation are excluded from the official release pipeline to avoid same-model evaluation bias.",
        f"- Canonical judged source: `data/processed/datasets/{QA_CANONICAL_JUDGED.name}` ({canonical_rows:,} rows)",
        f"- Canonical judged, context-cleaned: `data/processed/datasets/{QA_CANONICAL_JUDGED_CONTEXT_CLEANED.name}` ({canonical_cleaned_rows:,} rows)",
        "",
        "## Release Datasets",
        f"- Public final dataset: `{_repo_rel(QA_THREE_WAY_READY)}` ({public_rows:,} rows)",
        f"- Internal analysis source: `{_repo_rel(QA_THREE_WAY_ANALYSIS)}` ({analysis_rows:,} rows)",
        f"- Train split: `data/final/train.jsonl` ({train_rows:,} rows)",
        f"- Validation split: `data/final/val.jsonl` ({val_rows:,} rows)",
        f"- Test split: `data/final/test.jsonl` ({test_rows:,} rows)",
        f"- Public schema fields: {', '.join(PUBLIC_RELEASE_FIELDS)}",
        f"- Analysis schema fields: {', '.join(ANALYSIS_RELEASE_FIELDS)}",
        "",
        "## Validation",
        f"- Final validation status: `{validation.get('status', '')}`",
        f"- Validated rows: `{validation.get('validated_rows', 0):,}`",
        f"- Invalid rows: `{validation.get('invalid_rows', 0):,}`",
        "",
        "## Feature Engineering Phase 1",
        f"- Full matrix: `{_repo_rel(FEATURE_MATRIX_FULL)}` ({feature_payload['full_row_count']:,} rows, {feature_payload['full_column_count']} columns)",
        f"- Final matrix: `{_repo_rel(FEATURE_MATRIX_FINAL)}` ({feature_payload['final_row_count']:,} rows, {feature_payload['final_column_count']} columns)",
        *[f"- Full-matrix knowledge signal: `{feature}`" for feature in ACTIVE_KNOWLEDGE_SIGNALS],
        *[f"- Retained final-matrix knowledge signal: `{feature}`" for feature in RETAINED_FINAL_KNOWLEDGE_SIGNALS],
        *[f"- Active other feature group: `{feature}`" for feature in ACTIVE_OTHER_FEATURES],
        *[f"- Excluded phase-1 feature: `{feature}` because {reason}" for feature, reason in EXCLUDED_PHASE1_FEATURES.items()],
        "",
        "## EDA and Human Verification",
        f"- EDA1 final: `eda/figures/{EDA1_DIR.name}/`",
        f"- EDA2 final: `eda/figures/{EDA2_DIR.name}/`",
        f"- Human verification bundle: `data/processed/datasets/{HUMAN_VERIFICATION_BUNDLE_DIR.name}/`",
        f"- IAA summary: `data/processed/datasets/{HUMAN_VERIFICATION_BUNDLE_DIR.name}/reports/{IAA_SUMMARY_MD.name}`",
    ]
    _write_markdown(FINAL_RELEASE_MANIFEST_MD, md_lines)
    return payload


def main() -> None:
    ensure_dirs()
    feature_payload = build_feature_phase1_provenance()
    manifest_payload = build_final_release_manifest(feature_payload)
    print(json.dumps({"feature_phase1_provenance": feature_payload, "final_release_manifest": manifest_payload}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
