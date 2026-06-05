# QA Dataset EDA Summary

## Dataset Scope
- Public final release dataset: `7,592` QA pairs mirrored from the analysis source `qa_pairs_three_way_analysis.jsonl` into `qa_pairs_three_way_ready.jsonl`.
- External-Gemini diagnostic dataset: `8,005` QA pairs from `qa_pairs_canonical_judged_context_cleaned.jsonl`.

## Key Findings
- Extraction remains the dominant bucket (72.3%), while bridge (18.0%) captures the transitional reasoning region between literal extraction and fully multi-sentence inference (9.7%).
- Multi-sentence QA has a markedly higher weak-quality share than extraction in the full Gemini-annotated pool (11.6% vs 2.4%), supporting the decision to apply stricter human verification to inferential content.
- Within this final three-way release, the combined medium+hard share is 2.0% for extraction, 2.8% for bridge, and 99.9% for multi-sentence. This reflects the current release composition rather than a universal claim about all inferential QA.
- `culture` shows the highest weak-quality proportion in the full Gemini-annotated pool and should therefore be discussed explicitly in the limitations section.

## Outputs
- Figure: `01_reasoning_bucket_distribution.png`
- Figure: `02_domain_distribution.png`
- Figure: `03_reasoning_difficulty_heatmap.png`
- Figure: `04_domain_reasoning_stacked.png`
- Figure: `05_quality_reasoning_stacked.png`
- Figure: `06_quality_domain_stacked.png`
- Figure: `07_length_by_reasoning_boxplots.png`
- Table: `text_length_statistics.md`