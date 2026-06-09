"""Assemble a reproducible human-verification bundle with tasks, keys, and annotator outputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.config import ensure_dirs  # noqa: E402
from src.qa.human_verification import (  # noqa: E402
    annotation_distribution,
    build_combined_task1_rows,
    build_combined_task2_rows,
    load_jsonl,
    load_jsonl_robust,
    load_task1_from_export,
    validate_bundle_alignment,
    write_json,
    write_text,
)

BUNDLE_DIR = ROOT / "data" / "processed" / "datasets" / "human_verification_bundle_external_gemini_20260605"
TASKS_DIR = BUNDLE_DIR / "tasks"
KEYS_DIR = BUNDLE_DIR / "keys"
ANNOTATIONS_DIR = BUNDLE_DIR / "annotations"
REPORTS_DIR = BUNDLE_DIR / "reports"
SOURCE_DIR = ROOT / "data" / "processed" / "datasets" / "human_verification_external_gemini_20260602"

TASK1_SOURCE = SOURCE_DIR / "task1_quality_difficulty_100.jsonl"
TASK2_SOURCE = SOURCE_DIR / "task2_inferential_validity_50.jsonl"

TASK1_GEMINI_ANNOTATION_SOURCE = SOURCE_DIR / "task1_quality_difficulty_100_key_gemini31_flash_lite.jsonl"
TASK2_GEMINI_ANNOTATION_SOURCE = SOURCE_DIR / "task2_inferential_validity_50_key_gemini31_flash_lite.jsonl"

EXTERNAL_SOURCE_REDACTED = "external_annotation_export_redacted"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Assemble external-Gemini human-verification bundle.")
    parser.add_argument("--task1-export", required=True, help="Path to Task 1 annotation export JSON.")
    parser.add_argument("--task2-annotator1", required=True, help="Path to Task 2 annotator 1 JSONL.")
    parser.add_argument("--task2-annotator2", required=True, help="Path to Task 2 annotator 2 JSONL.")
    return parser


def _repo_rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return EXTERNAL_SOURCE_REDACTED


def _redact_task1_export_meta(meta: dict) -> dict:
    slots = meta.get("annotator_slots") or [{"slot": "annotator1"}, {"slot": "annotator2"}]
    return {
        "source": EXTERNAL_SOURCE_REDACTED,
        "annotator_slots": [{"slot": str(item.get("slot", ""))} for item in slots],
    }


def build_manifest(report: dict) -> dict:
    return {
        "bundle_name": BUNDLE_DIR.name,
        "bundle_version": "2026-06-05",
        "source_annotation_version": "2026-06-02",
        "structure": {
            "tasks": {"task1": "tasks/task1.json", "task2": "tasks/task2.json"},
            "keys": {
                "gemini": {"task1": "keys/gemini/task1.json", "task2": "keys/gemini/task2.json"},
            },
            "annotations": {
                "annotator1": {
                    "task1": "annotations/annotator1/task1.json",
                    "task2": "annotations/annotator1/task2.json",
                },
                "annotator2": {
                    "task1": "annotations/annotator2/task1.json",
                    "task2": "annotations/annotator2/task2.json",
                },
            },
            "reports": {"assembly": "reports/assembly_report.json"},
        },
        "notes": [
            "Tasks and Gemini reference labels are copied from the external-Gemini human-verification set.",
            "The bundle includes only task payloads, Gemini reference labels, and human annotator outputs.",
            "Annotations are normalized into JSON arrays for easier downstream processing.",
            "Task 2 annotator 1 source contained a malformed final line; the bundle repairs it deterministically by trimming trailing garbage after the first valid JSON object.",
            "Task 1 now comes directly from the external annotation export with two annotations embedded per sample.",
        ],
        "summary": report["summary"],
    }


def main() -> None:
    args = build_parser().parse_args()
    task1_export_source = Path(args.task1_export)
    task2_annotator1_source = Path(args.task2_annotator1)
    task2_annotator2_source = Path(args.task2_annotator2)

    ensure_dirs()
    for directory in (
        TASKS_DIR,
        KEYS_DIR / "gemini",
        ANNOTATIONS_DIR / "annotator1",
        ANNOTATIONS_DIR / "annotator2",
        REPORTS_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    task1_rows = load_jsonl(TASK1_SOURCE)
    task2_rows = load_jsonl(TASK2_SOURCE)
    task1_gemini_annotation = load_jsonl(TASK1_GEMINI_ANNOTATION_SOURCE)
    task2_gemini_annotation = load_jsonl(TASK2_GEMINI_ANNOTATION_SOURCE)

    task1_annotator1, task1_annotator2, task1_export_meta = load_task1_from_export(task1_export_source)
    task2_annotator1, task2_annotator1_repairs = load_jsonl_robust(task2_annotator1_source)
    task2_annotator2, task2_annotator2_repairs = load_jsonl_robust(task2_annotator2_source)

    task1_alignment = validate_bundle_alignment(
        label="task1",
        task_rows=task1_rows,
        gemini_annotation_rows=task1_gemini_annotation,
        annotator1_rows=task1_annotator1,
        annotator2_rows=task1_annotator2,
    )
    task2_alignment = validate_bundle_alignment(
        label="task2",
        task_rows=task2_rows,
        gemini_annotation_rows=task2_gemini_annotation,
        annotator1_rows=task2_annotator1,
        annotator2_rows=task2_annotator2,
    )
    if not task1_alignment["all_ids_match"]:
        raise RuntimeError(f"Task 1 bundle assembly failed alignment: {task1_alignment}")
    if not task2_alignment["all_ids_match"]:
        raise RuntimeError(f"Task 2 bundle assembly failed alignment: {task2_alignment}")

    write_json(
        TASKS_DIR / "task1.json",
        build_combined_task1_rows(
            task_rows=task1_rows,
            gemini_annotation_rows=task1_gemini_annotation,
            annotator1_rows=task1_annotator1,
            annotator2_rows=task1_annotator2,
        ),
    )
    write_json(
        TASKS_DIR / "task2.json",
        build_combined_task2_rows(
            task_rows=task2_rows,
            gemini_annotation_rows=task2_gemini_annotation,
            annotator1_rows=task2_annotator1,
            annotator2_rows=task2_annotator2,
        ),
    )

    for subdir, name, payload in [
        (KEYS_DIR / "gemini", "task1.json", task1_gemini_annotation),
        (KEYS_DIR / "gemini", "task2.json", task2_gemini_annotation),
        (ANNOTATIONS_DIR / "annotator1", "task1.json", task1_annotator1),
        (ANNOTATIONS_DIR / "annotator1", "task2.json", task2_annotator1),
        (ANNOTATIONS_DIR / "annotator2", "task1.json", task1_annotator2),
        (ANNOTATIONS_DIR / "annotator2", "task2.json", task2_annotator2),
    ]:
        write_json(subdir / name, payload)

    report = {
        "bundle_dir": _repo_rel(BUNDLE_DIR),
        "sources": {
            "task1": _repo_rel(TASK1_SOURCE),
            "task2": _repo_rel(TASK2_SOURCE),
            "task1_gemini_annotation": _repo_rel(TASK1_GEMINI_ANNOTATION_SOURCE),
            "task2_gemini_annotation": _repo_rel(TASK2_GEMINI_ANNOTATION_SOURCE),
            "task1_export": _repo_rel(task1_export_source),
            "task2_annotator1": _repo_rel(task2_annotator1_source),
            "task2_annotator2": _repo_rel(task2_annotator2_source),
        },
        "task1_export_meta": _redact_task1_export_meta(task1_export_meta),
        "alignment": {"task1": task1_alignment, "task2": task2_alignment},
        "repair_log": {
            "task1_annotator1": [],
            "task1_annotator2": [],
            "task2_annotator1": task2_annotator1_repairs,
            "task2_annotator2": task2_annotator2_repairs,
        },
        "summary": {
            "task1_rows": len(task1_rows),
            "task2_rows": len(task2_rows),
            "task1_annotator1_distribution": {
                "quality": annotation_distribution(task1_annotator1, "human_quality_band"),
                "difficulty": annotation_distribution(task1_annotator1, "human_difficulty_band"),
            },
            "task1_annotator2_distribution": {
                "quality": annotation_distribution(task1_annotator2, "human_quality_band"),
                "difficulty": annotation_distribution(task1_annotator2, "human_difficulty_band"),
            },
            "task2_annotator1_distribution": annotation_distribution(
                task2_annotator1, "human_inferential_validity_band"
            ),
            "task2_annotator2_distribution": annotation_distribution(
                task2_annotator2, "human_inferential_validity_band"
            ),
        },
    }

    write_json(REPORTS_DIR / "assembly_report.json", report)
    write_json(BUNDLE_DIR / "manifest.json", build_manifest(report))
    write_text(
        BUNDLE_DIR / "README.md",
        "\n".join(
            [
                "# Human Verification Bundle",
                "",
                "This bundle packages human-verification tasks, Gemini reference labels,",
                "and two human annotator result sets.",
                "",
                "## Structure",
                "",
                "- `tasks/task1.json`: combined Task 1 payload with Gemini key, annotator1, annotator2",
                "- `tasks/task2.json`: combined Task 2 payload with Gemini key, annotator1, annotator2",
                "- `keys/gemini/*.json`: Gemini reference labels",
                "- `annotations/annotator1/*.json`: annotator 1 labels",
                "- `annotations/annotator2/*.json`: annotator 2 labels",
                "- `reports/assembly_report.json`: provenance, alignment, and repair details",
                "- `manifest.json`: machine-readable bundle manifest",
                "",
                "## Notes",
                "",
                "- Task 2 annotator 1 source required deterministic repair on one malformed line.",
                "- Task 1 now comes directly from the external annotation export with two annotations embedded per sample.",
                "",
            ]
        ),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
