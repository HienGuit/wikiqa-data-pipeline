# Feature Engineering Phase 1 Provenance

- Analysis source dataset: `data/processed/datasets/qa_pairs_three_way_analysis.jsonl`
- Public release dataset: `data/processed/datasets/qa_pairs_three_way_ready.jsonl`
- Full matrix: `data/processed/features/feature_matrix_full.csv` (7,592 rows, 28 columns)
- Final matrix: `data/final/feature_matrix_final.csv` (7,592 rows, 24 columns)

## Active Phase-1 Features
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
- Other feature group: `structural_features`
- Other feature group: `question_type`
- Other feature group: `answer_type`
- Other feature group: `popularity_source`

## Excluded From Phase 1
- `wiki_level`: Excluded because crawl depth is not reliably preserved in the current raw metadata and would require a separate taxonomy-recovery pass.
- `linked_entities`: Excluded because API coverage and stability were not strong enough for the final phase-1 feature set.
