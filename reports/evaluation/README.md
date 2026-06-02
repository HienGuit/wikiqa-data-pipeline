# Evaluation Reports

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
