"""Sync public-facing QA reports from internal processed artifacts into tracked repo folders."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

INTERNAL_QA_REPORTS = ROOT / "data" / "processed" / "reports" / "qa"
INTERNAL_HV_BUNDLE = ROOT / "data" / "processed" / "datasets" / "human_verification_bundle_20260602"

TRACKED_REPORTS = ROOT / "reports"
TRACKED_QA_RELEASE = TRACKED_REPORTS / "release"
TRACKED_HUMAN_VERIFICATION = TRACKED_REPORTS / "evaluation"

QA_RELEASE_FILES = [
    "final_release_manifest.json",
    "final_release_manifest.md",
    "feature_phase1_provenance.json",
    "feature_phase1_provenance.md",
    "qa_three_way_final_validation_report.json",
    "qa_judge_canonical_promotion_report.json",
    "qa_judge_provenance_report.json",
]

HUMAN_VERIFICATION_FILES = [
    ("manifest.json", INTERNAL_HV_BUNDLE / "manifest.json"),
    ("assembly_report.json", INTERNAL_HV_BUNDLE / "reports" / "assembly_report.json"),
    ("iaa_summary.json", INTERNAL_HV_BUNDLE / "reports" / "iaa_summary.json"),
    ("iaa_summary.md", INTERNAL_HV_BUNDLE / "reports" / "iaa_summary.md"),
    ("iaa_visualization_report.json", INTERNAL_HV_BUNDLE / "reports" / "iaa_visualization_report.json"),
    ("iaa_kappa_heatmap_matrix.png", INTERNAL_HV_BUNDLE / "reports" / "iaa_kappa_heatmap_matrix.png"),
    ("task1_a1_vs_a2_confusion.png", INTERNAL_HV_BUNDLE / "reports" / "task1_a1_vs_a2_confusion.png"),
    ("task2_a1_vs_a2_confusion.png", INTERNAL_HV_BUNDLE / "reports" / "task2_a1_vs_a2_confusion.png"),
    ("full_llm_agreement_report.json", INTERNAL_QA_REPORTS / "full_llm_agreement_report.json"),
    ("full_llm_agreement_report.md", INTERNAL_QA_REPORTS / "full_llm_agreement_report.md"),
    ("full_llm_agreement_overview.png", INTERNAL_QA_REPORTS / "full_llm_agreement_overview.png"),
    ("full_llm_agreement_visualization_report.json", INTERNAL_QA_REPORTS / "full_llm_agreement_visualization_report.json"),
]


def copy_file(src: Path, dst: Path) -> dict[str, str | int]:
    if not src.exists():
        raise FileNotFoundError(f"Missing source file: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return {
        "source": str(src),
        "destination": str(dst),
        "size_bytes": dst.stat().st_size,
    }


def sync_qa_release() -> list[dict[str, str | int]]:
    copied: list[dict[str, str | int]] = []
    for filename in QA_RELEASE_FILES:
        copied.append(copy_file(INTERNAL_QA_REPORTS / filename, TRACKED_QA_RELEASE / filename))
    return copied


def sync_human_verification() -> list[dict[str, str | int]]:
    copied: list[dict[str, str | int]] = []
    for filename, src in HUMAN_VERIFICATION_FILES:
        copied.append(copy_file(src, TRACKED_HUMAN_VERIFICATION / filename))
    return copied


def write_evaluation_readme() -> dict[str, str | int]:
    dst = TRACKED_HUMAN_VERIFICATION / "README.md"
    content = """# Evaluation Reports

This folder contains the GitHub-tracked evaluation artifacts for the QA release.

## Contents

- `manifest.json`: machine-readable manifest for the human-verification bundle
- `assembly_report.json`: bundle provenance and alignment checks
- `iaa_summary.json` / `iaa_summary.md`: human-verification inter-annotator agreement summary
- `iaa_visualization_report.json`: metadata for the IAA figures
- `iaa_kappa_heatmap_matrix.png`: pairwise kappa heatmap for A1, A2, Gemini, and DeepSeek
- `task1_a1_vs_a2_confusion.png`: Task 1 confusion matrices
- `task2_a1_vs_a2_confusion.png`: Task 2 confusion matrix
- `full_llm_agreement_report.json` / `full_llm_agreement_report.md`: Gemini vs DeepSeek agreement on the full shared judged population
- `full_llm_agreement_overview.png`: compact visualization of full-population observed agreement and Cohen's kappa
- `full_llm_agreement_visualization_report.json`: metadata for the full-population agreement visualization

## Notes

- Human-verification artifacts are derived from the internal bundle under `data/processed/datasets/human_verification_bundle_20260602/`.
- Full-population LLM agreement is derived from the shared rows between the cleaned Gemini and DeepSeek judged datasets.
"""
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(content, encoding="utf-8")
    return {"source": "generated", "destination": str(dst), "size_bytes": dst.stat().st_size}


def main() -> None:
    TRACKED_QA_RELEASE.mkdir(parents=True, exist_ok=True)
    TRACKED_HUMAN_VERIFICATION.mkdir(parents=True, exist_ok=True)

    report = {
        "release": sync_qa_release(),
        "evaluation": [write_evaluation_readme(), *sync_human_verification()],
    }
    manifest_path = TRACKED_REPORTS / "sync_manifest.json"
    manifest_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Synced reports to {TRACKED_REPORTS}")


if __name__ == "__main__":
    main()
