# Feature Engineering EDA Summary

- Feature matrix rows: `7,592`
- Feature matrix columns: `28`
- Non-null knowledge difficulty rows: `7,592`
- Top question types: `{'other': 5121, 'how_many': 871, 'when': 630}`
- Top answer types: `{'Proper Noun - Location': 2229, 'Common Noun Phrase': 1905, 'Proper Noun - Organization': 1091}`
- Popularity source mix: `{'question_entity': 3993, 'answer_entity': 2485, 'wiki_title': 959, 'wiki_section': 155}`
- Full-matrix phase-1 knowledge signals available before feature selection: `page_views_rank`, `site_links_rank`, `wiki_count_rank`, `statements_rank`, `references_rank`, and `knowledge_difficulty`.
- After multicollinearity-based feature selection, the retained knowledge signals are `page_views_rank`, `wiki_count_rank`, `statements_rank`, and `knowledge_difficulty`.
- Excluded from phase 1: `wiki_level` and `linked_entities` due to insufficient provenance or API stability.
- Multicollinearity table: `multicollinearity_pairs.csv`
- The correlation heatmap is retained as the main visual evidence for feature-pruning decisions.

## Figures
- `01_knowledge_difficulty_distribution.png`
- `02_answer_type_distribution.png`
- `03_popularity_source_distribution.png`
- `04_question_type_distribution.png`
- `05_structural_boxplots_by_reasoning.png`
- `07_feature_correlation_heatmap.png`