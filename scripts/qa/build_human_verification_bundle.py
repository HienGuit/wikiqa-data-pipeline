"""Assemble a reproducible human-verification bundle with tasks, keys, and annotator outputs."""

from __future__ import annotations

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

BUNDLE_DIR = ROOT / "data" / "processed" / "datasets" / "human_verification_bundle_20260602"
TASKS_DIR = BUNDLE_DIR / "tasks"
KEYS_DIR = BUNDLE_DIR / "keys"
ANNOTATIONS_DIR = BUNDLE_DIR / "annotations"
REPORTS_DIR = BUNDLE_DIR / "reports"

TASK1_SOURCE = (
    ROOT
    / "data"
    / "processed"
    / "datasets"
    / "human_verification_dual_judge_20260602"
    / "task1_quality_difficulty_100.jsonl"
)
TASK2_SOURCE = (
    ROOT
    / "data"
    / "processed"
    / "datasets"
    / "human_verification_dual_judge_20260602"
    / "task2_inferential_validity_50.jsonl"
)

TASK1_GEMINI_KEY_SOURCE = (
    ROOT
    / "data"
    / "processed"
    / "datasets"
    / "human_verification_dual_judge_20260602"
    / "task1_quality_difficulty_100_key_gemini31_flash_lite.jsonl"
)
TASK1_DEEPSEEK_KEY_SOURCE = (
    ROOT
    / "data"
    / "processed"
    / "datasets"
    / "human_verification_dual_judge_20260602"
    / "task1_quality_difficulty_100_key_deepseek_v4_flash.jsonl"
)
TASK2_GEMINI_KEY_SOURCE = (
    ROOT
    / "data"
    / "processed"
    / "datasets"
    / "human_verification_dual_judge_20260602"
    / "task2_inferential_validity_50_key_gemini31_flash_lite.jsonl"
)
TASK2_DEEPSEEK_KEY_SOURCE = (
    ROOT
    / "data"
    / "processed"
    / "datasets"
    / "human_verification_dual_judge_20260602"
    / "task2_inferential_validity_50_key_deepseek_v4_flash.jsonl"
)

TASK1_EXPORT_SOURCE = Path(r"C:\Users\GHien\Downloads\task1.json")
TASK2_ANNOTATOR1_SOURCE = Path(r"C:\Users\GHien\Downloads\task2_inferential_validity_50_completed_annotator1.jsonl")
TASK2_ANNOTATOR2_SOURCE = Path(
    r"C:\Users\GHien\Downloads\gemini31_pro_task_runner_20260602\outputs\task2_inferential_validity_50_completed_annotator2.jsonl"
)


def build_manifest(report: dict) -> dict:
    return {
        "bundle_name": BUNDLE_DIR.name,
        "version": "2026-06-02",
        "structure": {
            "tasks": {"task1": "tasks/task1.json", "task2": "tasks/task2.json"},
            "keys": {
                "gemini": {"task1": "keys/gemini/task1.json", "task2": "keys/gemini/task2.json"},
                "deepseek": {"task1": "keys/deepseek/task1.json", "task2": "keys/deepseek/task2.json"},
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
            "Tasks and judge keys are copied from the latest dual-judge human-verification set.",
            "Annotations are normalized into JSON arrays for easier downstream processing.",
            "Task 2 annotator 1 source contained a malformed final line; the bundle repairs it deterministically by trimming trailing garbage after the first valid JSON object.",
            "Task 1 now comes directly from the external annotation export with two annotations embedded per sample.",
        ],
        "summary": report["summary"],
    }


def main() -> None:
    ensure_dirs()
    for directory in (
        TASKS_DIR,
        KEYS_DIR / "gemini",
        KEYS_DIR / "deepseek",
        ANNOTATIONS_DIR / "annotator1",
        ANNOTATIONS_DIR / "annotator2",
        REPORTS_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    task1_rows = load_jsonl(TASK1_SOURCE)
    task2_rows = load_jsonl(TASK2_SOURCE)
    task1_gemini_key = load_jsonl(TASK1_GEMINI_KEY_SOURCE)
    task1_deepseek_key = load_jsonl(TASK1_DEEPSEEK_KEY_SOURCE)
    task2_gemini_key = load_jsonl(TASK2_GEMINI_KEY_SOURCE)
    task2_deepseek_key = load_jsonl(TASK2_DEEPSEEK_KEY_SOURCE)

    task1_annotator1, task1_annotator2, task1_export_meta = load_task1_from_export(TASK1_EXPORT_SOURCE)
    task2_annotator1, task2_annotator1_repairs = load_jsonl_robust(TASK2_ANNOTATOR1_SOURCE)
    task2_annotator2, task2_annotator2_repairs = load_jsonl_robust(TASK2_ANNOTATOR2_SOURCE)

    task1_alignment = validate_bundle_alignment(
        label="task1",
        task_rows=task1_rows,
        gemini_key_rows=task1_gemini_key,
        deepseek_key_rows=task1_deepseek_key,
        annotator1_rows=task1_annotator1,
        annotator2_rows=task1_annotator2,
    )
    task2_alignment = validate_bundle_alignment(
        label="task2",
        task_rows=task2_rows,
        gemini_key_rows=task2_gemini_key,
        deepseek_key_rows=task2_deepseek_key,
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
            gemini_key_rows=task1_gemini_key,
            deepseek_key_rows=task1_deepseek_key,
            annotator1_rows=task1_annotator1,
            annotator2_rows=task1_annotator2,
        ),
    )
    write_json(
        TASKS_DIR / "task2.json",
        build_combined_task2_rows(
            task_rows=task2_rows,
            gemini_key_rows=task2_gemini_key,
            deepseek_key_rows=task2_deepseek_key,
            annotator1_rows=task2_annotator1,
            annotator2_rows=task2_annotator2,
        ),
    )

    for subdir, name, payload in [
        (KEYS_DIR / "gemini", "task1.json", task1_gemini_key),
        (KEYS_DIR / "gemini", "task2.json", task2_gemini_key),
        (KEYS_DIR / "deepseek", "task1.json", task1_deepseek_key),
        (KEYS_DIR / "deepseek", "task2.json", task2_deepseek_key),
        (ANNOTATIONS_DIR / "annotator1", "task1.json", task1_annotator1),
        (ANNOTATIONS_DIR / "annotator1", "task2.json", task2_annotator1),
        (ANNOTATIONS_DIR / "annotator2", "task1.json", task1_annotator2),
        (ANNOTATIONS_DIR / "annotator2", "task2.json", task2_annotator2),
    ]:
        write_json(subdir / name, payload)

    report = {
        "bundle_dir": str(BUNDLE_DIR),
        "sources": {
            "task1": str(TASK1_SOURCE),
            "task2": str(TASK2_SOURCE),
            "task1_gemini_key": str(TASK1_GEMINI_KEY_SOURCE),
            "task1_deepseek_key": str(TASK1_DEEPSEEK_KEY_SOURCE),
            "task2_gemini_key": str(TASK2_GEMINI_KEY_SOURCE),
            "task2_deepseek_key": str(TASK2_DEEPSEEK_KEY_SOURCE),
            "task1_export": str(TASK1_EXPORT_SOURCE),
            "task2_annotator1": str(TASK2_ANNOTATOR1_SOURCE),
            "task2_annotator2": str(TASK2_ANNOTATOR2_SOURCE),
        },
        "task1_export_meta": task1_export_meta,
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
                "This bundle packages the latest dual-judge human-verification tasks,",
                "Gemini and DeepSeek reference keys, and two annotator result sets.",
                "",
                "## Structure",
                "",
                "- `tasks/task1.json`: combined Task 1 payload with Gemini key, DeepSeek key, annotator1, annotator2",
                "- `tasks/task2.json`: combined Task 2 payload with Gemini key, DeepSeek key, annotator1, annotator2",
                "- `keys/gemini/*.json`: Gemini reference labels",
                "- `keys/deepseek/*.json`: DeepSeek reference labels",
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
