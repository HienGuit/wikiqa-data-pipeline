---
language:
- vi
task_categories:
- question-answering
- text-generation
pretty_name: Vietnamese WikiQA
size_categories:
- 1K<n<10K
---

# Dataset Card for Vietnamese WikiQA

## Dataset Description

- Repository: Vietnamese WikiQA Pipeline
- Hugging Face Dataset: https://huggingface.co/datasets/HienNGuit/Domain-ViWikiQA
- Language: Vietnamese (`vi`)
- License: CC BY-SA 4.0

### Dataset Summary

Vietnamese WikiQA contains 7,592 Vietnamese question-answer pairs built from Vietnamese Wikipedia. Each row includes a context, question, answer, and final reasoning bucket: `extraction`, `bridge`, or `multi-sentence`.

The dataset is designed for extractive QA, open-domain QA, and RAG evaluation or training.

## Dataset Structure

The dataset is split by title to avoid leakage across train, validation, and test sets.

| Split | Rows |
|---|---:|
| `train` | 6,074 |
| `val` | 759 |
| `test` | 759 |
| Total | 7,592 |

### Reasoning Buckets

- `extraction`: direct evidence from one local span
- `bridge`: weak inferential multi-sentence cases
- `multi-sentence`: usable or strong multi-sentence reasoning cases

## Dataset Creation

### Source Data

- Source corpus: Vietnamese Wikipedia pages from curated core domains.
- Raw pages are cleaned and split into section-aware chunks.

### Annotation Provenance

QA candidates were generated with DeepSeek V4 Flash. Automatic bucket labels were assigned by Gemini as an external automatic annotator after rule-based validation and audited with human verification.

The official pipeline is:

1. Crawl Vietnamese Wikipedia pages and metadata.
2. Clean raw page text and split pages into section-aware chunks.
3. Generate QA candidates with DeepSeek V4 Flash.
4. Validate QA candidates with rule-based checks.
5. Assign automatic bucket labels with Gemini from the context, question, answer, and annotation guideline.
6. Audit sampled cases with two human annotators.
7. Release the final three-way dataset.

Human verification computes pairwise agreement for Annotator 1 vs Annotator 2, Annotator 1 vs Gemini annotation, and Annotator 2 vs Gemini annotation.

## Considerations for Using the Data

- The source text comes from Wikipedia and reflects encyclopedic writing style.
- `quality_band` and `inferential_validity_band` are automatic Gemini annotations audited through sampled human verification.
- `difficulty_band` is retained in internal analysis artifacts but is not a public target label in the final release.

## Citation

```bibtex
@dataset{vietnamese_wikiqa,
  title = {Vietnamese WikiQA: High-Quality MRC and RAG Dataset},
  year = {2026},
  publisher = {Hugging Face},
  url = {https://huggingface.co/datasets/HienNGuit/Domain-ViWikiQA}
}
```
