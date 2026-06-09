# Final Release Manifest

## External Annotation
- QA generator: `DeepSeek V4 Flash`
- External automatic annotator: `Gemini`
- Rationale: Gemini is used as the external annotator because it is separate from the QA generation model.
- Official annotation labels come from Gemini and are audited with human verification.
- Gemini-annotated source: `data/processed/datasets/qa_pairs_canonical_annotated.jsonl` (8,314 rows)
- Gemini-annotated, context-cleaned: `data/processed/datasets/qa_pairs_canonical_annotated_context_cleaned.jsonl` (8,005 rows)

## Release Datasets
- Public final dataset: `data/processed/datasets/qa_pairs_three_way_ready.jsonl` (7,592 rows)
- Internal analysis source: `data/processed/datasets/qa_pairs_three_way_analysis.jsonl` (7,592 rows)
- Train split: `data/final/train.jsonl` (6,074 rows)
- Validation split: `data/final/val.jsonl` (759 rows)
- Test split: `data/final/test.jsonl` (759 rows)
- Public schema fields: chunk_id, domain, title, section, context, question, answer, final_reasoning_bucket, quality_band, inferential_validity_band
- Analysis schema fields: chunk_id, domain, title, section, context, reasoning_type, question, answer, quality_band, difficulty_band, inferential_validity_band, final_reasoning_bucket

## Validation
- Final validation status: `pass`
- Validated rows: `7,592`
- Invalid rows: `0`

## Feature Engineering Phase 1
- Full matrix: `data/processed/features/feature_matrix_full.csv` (7,592 rows, 28 columns)
- Final matrix: `data/final/feature_matrix_final.csv` (7,592 rows, 24 columns)
- Full-matrix knowledge signal: `page_views_rank`
- Full-matrix knowledge signal: `site_links_rank`
- Full-matrix knowledge signal: `wiki_count_rank`
- Full-matrix knowledge signal: `statements_rank`
- Full-matrix knowledge signal: `references_rank`
- Full-matrix knowledge signal: `knowledge_difficulty`
- Retained final-matrix knowledge signal: `page_views_rank`
- Retained final-matrix knowledge signal: `wiki_count_rank`
- Retained final-matrix knowledge signal: `statements_rank`
- Retained final-matrix knowledge signal: `knowledge_difficulty`
- Active other feature group: `structural_features`
- Active other feature group: `question_type`
- Active other feature group: `answer_type`
- Active other feature group: `popularity_source`
- Excluded phase-1 feature: `wiki_level`: Excluded because crawl depth is not reliably preserved in the current raw metadata and would require a separate taxonomy-recovery pass.
- Excluded phase-1 feature: `linked_entities`: Excluded because API coverage and stability were not strong enough for the final phase-1 feature set.

## EDA and Human Verification
- EDA1 final: `eda/figures/02_qa_dataset_eda/`
- EDA2 final: `eda/figures/03_feature_engineering_eda/`
- Human verification bundle: `data/processed/datasets/human_verification_bundle_external_gemini_20260605/`
- IAA summary: `data/processed/datasets/human_verification_bundle_external_gemini_20260605/reports/iaa_summary.md`
