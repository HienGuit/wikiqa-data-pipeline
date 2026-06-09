# Vietnamese WikiQA Data Pipeline

This repository builds a Vietnamese question-answering dataset from Vietnamese Wikipedia and packages the full evaluation stack around it: external Gemini annotation, human verification, inter-annotator agreement, release normalization, and feature-engineering analysis.

## Documentation

- Dataset card: `dataset_card.md`
- Data dictionary: `data_dictionary.md`
- QA bucket guideline for external annotation and human verification: `qa_bucket_guideline.md`
- Final public-facing reports tracked on GitHub: `reports/`
- Hugging Face dataset: `https://huggingface.co/datasets/HienNGuit/Domain-ViWikiQA`

## Dataset Overview

### Public Release
- Public HF-ready dataset: `data/processed/datasets/qa_pairs_three_way_ready.jsonl`
- Hugging Face dataset page: `https://huggingface.co/datasets/HienNGuit/Domain-ViWikiQA`
- Rows: `7,592`
- Final reasoning buckets:
  - `extraction`
  - `bridge`
  - `multi-sentence`

### Public Schema
- `chunk_id`
- `domain`
- `title`
- `section`
- `context`
- `question`
- `answer`
- `final_reasoning_bucket`
- `quality_band`
- `inferential_validity_band`

### Final Splits
- Train split: `data/final/train.jsonl` (`6,074` rows)
- Validation split: `data/final/val.jsonl` (`759` rows)
- Test split: `data/final/test.jsonl` (`759` rows)
- Final feature matrix: `data/final/feature_matrix_final.csv`

### Internal Analysis Source
- Analysis dataset: `data/processed/datasets/qa_pairs_three_way_analysis.jsonl`
- Purpose: internal diagnostics and downstream analysis
- Extra legacy fields retained internally:
  - `reasoning_type`
  - `difficulty_band`

## Data Creation Pipeline

The dataset is created in six stages:

1. Crawl Vietnamese Wikipedia pages and metadata from a curated taxonomy.
2. Clean raw page text and split pages into section-aware chunks.
3. Generate QA candidates with DeepSeek V4 Flash using two reasoning styles:
   - `extraction`
   - `multi-sentence`
4. Validate QA candidates with rule-based checks:
   - exact answer span
   - evidence span for multi-sentence QA
   - single-target question constraint
5. Assign automatic bucket labels with Gemini as an external LLM annotator.
6. Audit the annotation protocol with human verification and release the final three-way dataset:
   - `extraction`
   - `bridge`
   - `multi-sentence`

## Final Artifacts

### Annotation Provenance
- QA generator: `DeepSeek V4 Flash`
- External automatic annotator: `Gemini`
- Human verification: two human annotators on sampled audit tasks
- Official annotation labels come from the external Gemini annotation pass and are audited with human verification.
- Gemini-annotated source: `data/processed/datasets/qa_pairs_canonical_annotated.jsonl`
- Gemini-annotated, context-cleaned: `data/processed/datasets/qa_pairs_canonical_annotated_context_cleaned.jsonl`
- Gemini-annotated, release-normalized: `data/processed/datasets/qa_pairs_canonical_annotated_release.jsonl`

### Human Verification
- Internal bundle source: `data/processed/datasets/human_verification_bundle_external_gemini_20260605/`
- GitHub-tracked IAA summary: `reports/evaluation/iaa_summary.md`
- GitHub-tracked IAA visualizations: `reports/evaluation/`
- Agreement is computed for:
  - Annotator 1 vs Annotator 2
  - Annotator 1 vs Gemini
  - Annotator 2 vs Gemini
- Guideline used for bucket-based external annotation and human verification: `qa_bucket_guideline.md`

### EDA
- EDA1 dataset overview: `eda/figures/02_qa_dataset_eda/`
- EDA2 feature-engineering analysis: `eda/figures/03_feature_engineering_eda/`

### Release Reports
- Final release reports: `reports/release/`
- Human verification and IAA reports: `reports/evaluation/`

### Feature Engineering Phase 1
- Full matrix: `data/processed/features/feature_matrix_full.csv`
- Final matrix after multicollinearity-based pruning: `data/final/feature_matrix_final.csv`

Full-matrix knowledge signals:
- `page_views_rank`
- `site_links_rank`
- `wiki_count_rank`
- `statements_rank`
- `references_rank`
- `knowledge_difficulty`

Retained knowledge signals in the final matrix:
- `page_views_rank`
- `wiki_count_rank`
- `statements_rank`
- `knowledge_difficulty`

Excluded from phase 1:
- `wiki_level`
- `linked_entities`

## Repository Structure

```text
wikiqa-data-pipeline/
├── dataset_card.md               # public dataset card
├── data_dictionary.md            # field-level schema notes
├── qa_bucket_guideline.md        # bucket-label guideline for external annotation and human verification
├── reports/
│   ├── release/                  # tracked final release manifests and validation reports
│   └── evaluation/               # tracked IAA summaries and visualization artifacts
├── configs/                     # YAML configs
├── data/
│   ├── raw/                     # raw Wikipedia pages and metadata
│   ├── interim/                 # chunks and intermediate artifacts
│   ├── processed/
│   │   ├── datasets/            # annotation, release, and analysis datasets
│   │   ├── features/            # feature matrices and build reports
│   │   ├── final/               # final selected feature matrices
│   │   ├── reports/qa/          # provenance, validation, and release manifests
│   │   └── wiki_metrics/        # entity-level wiki metrics
├── eda/
│   ├── figures/                 # publication-ready EDA outputs
│   ├── scripts/                 # EDA build scripts
│   └── utils/                   # EDA loading and plotting helpers
├── scripts/
│   ├── features/                # feature engineering pipeline
│   └── qa/                      # QA generation, annotation, verification, release metadata
└── src/
    ├── features/                # reusable feature logic
    ├── ingestion/               # crawling and raw ingestion
    ├── processing/              # cleaning and chunking
    └── qa/                      # QA generation, validation, release schemas
```

## How To Run The Pipeline

### 1. Run external Gemini annotation
```bash
python -m src.qa.batch annotate ...
python -m src.qa.dataset merge-annotation ...
```

### 2. Build the final three-way release
```bash
python scripts/qa/build_three_way_dataset.py
python scripts/qa/final_validate_release_dataset.py
```

### 3. Build dataset-overview EDA
```bash
python eda/scripts/build_qa_dataset_eda.py
```

### 4. Build the feature matrix
```bash
python scripts/features/build_feature_matrix.py
```

### 5. Build feature-engineering EDA
```bash
python eda/scripts/build_feature_engineering_eda.py
```

### 6. Rebuild human verification and IAA reports
Human annotation exports are not tracked publicly because they may contain annotator metadata. To rebuild the redacted external-Gemini verification bundle, provide local annotation export paths:

```bash
python scripts/qa/build_human_verification_bundle.py \
  --task1-export path/to/task1_export.json \
  --task2-annotator1 path/to/task2_annotator1.jsonl \
  --task2-annotator2 path/to/task2_annotator2.jsonl

python scripts/qa/compute_iaa.py
python scripts/qa/visualize_iaa.py
python scripts/qa/build_release_metadata.py
python scripts/qa/sync_repo_reports.py
```

The public tracked reports are redacted and contain only repository-relative paths or `external_annotation_export_redacted`.

### 7. Rebuild provenance and release metadata
```bash
python scripts/qa/build_release_metadata.py
```

### 8. Sync GitHub-tracked final reports
```bash
python scripts/qa/sync_repo_reports.py
```

## Notes On Public Release Design

- `difficulty_band` is intentionally excluded from the public final dataset because it is treated as a legacy diagnostic signal rather than a high-confidence public target label.
- `reasoning_type` is retained only in the internal analysis dataset; the public release exposes `final_reasoning_bucket` instead.
- `is_valid` and `error` are pipeline/debug fields and are excluded from the public final release.

## Provenance Entry Points

- Final manifest JSON: `reports/release/final_release_manifest.json`
- Final manifest Markdown: `reports/release/final_release_manifest.md`
- Feature phase-1 provenance JSON: `reports/release/feature_phase1_provenance.json`
- Feature phase-1 provenance Markdown: `reports/release/feature_phase1_provenance.md`
- Human verification manifest: `reports/evaluation/manifest.json`
- IAA summary: `reports/evaluation/iaa_summary.md`

These files are the main entry points for checking which dataset is public, which dataset is internal, how external annotation was audited, which features were kept or excluded, and how inter-annotator agreement was measured. The train/validation/test artifacts intended for direct downstream use live under `data/final/`.
