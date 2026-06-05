# Data Dictionary

This file documents the public Vietnamese WikiQA release files and the final feature matrix.

## QA Pairs (`train.jsonl`, `val.jsonl`, `test.jsonl`)

Each line is a JSON object for one question-answer pair.

| Field | Type | Meaning | Values |
|---|---|---|---|
| `chunk_id` | string | Unique source chunk identifier. | Example: `19794_0` |
| `domain` | string | Wikipedia domain/category used during crawl. | `Kinh te`, `Van hoa`, `Lich su`, `Khoa hoc`, `Dia ly`, ... |
| `title` | string | Wikipedia page title. | Free text |
| `section` | string | Section heading for the context. | Free text |
| `context` | string | Text evidence used to answer the question. | Vietnamese text |
| `question` | string | Generated Vietnamese question. | Free text |
| `answer` | string | Concise answer grounded in the context. | Free text |
| `final_reasoning_bucket` | string | Public reasoning category after release mapping. | `extraction`, `bridge`, `multi-sentence` |
| `quality_band` | string | QA quality label assigned by the external Gemini annotation pass and audited with human verification. | `strong`, `usable`, `weak` |
| `inferential_validity_band` | string | Multi-sentence inference validity label assigned by the external Gemini annotation pass and audited with human verification. | `strong`, `usable`, `weak` |

### Label Reliability Note

`quality_band` and `inferential_validity_band` are external Gemini annotations audited on a sampled human-verification set. They should be interpreted as controlled metadata labels rather than absolute ground-truth labels. Human-human agreement and human-Gemini agreement are reported in `reports/evaluation/iaa_summary.md`.

`difficulty_band` is retained only in internal analysis artifacts because it is a subjective diagnostic signal and is not used as a public target label in the final release.

## Internal Analysis Fields

These fields may appear in internal processed datasets but are not all public target labels.

| Field | Type | Meaning | Values |
|---|---|---|---|
| `reasoning_type` | string | Initial generation mode or intermediate reasoning label. | `extraction`, `multi-sentence`, `bridge` |
| `difficulty_band` | string | Difficulty label assigned by the external Gemini annotation pass and audited with human verification. | `easy`, `medium`, `hard` |
| `is_valid` | boolean | Pipeline validation status for generated candidates. | `true`, `false` |
| `error` | string/null | Pipeline validation or generation error. | Free text or `null` |

## Feature Matrix (`feature_matrix_final.csv`)

The final feature matrix contains numerical and categorical features derived from the QA dataset.

| Field | Type | Meaning |
|---|---|---|
| `row_id` | string | Feature row identifier, usually aligned with `chunk_id`. |
| `chunk_id` | string | Source chunk identifier. |
| `title` | string | Wikipedia page title. |
| `domain` | string | Wikipedia domain/category. |
| `section` | string | Section heading. |
| `reasoning_type` | string | Internal reasoning type. |
| `final_reasoning_bucket` | string | Public reasoning bucket. |
| `quality_band` | string | External Gemini quality label audited with human verification. |
| `difficulty_band` | string | External Gemini difficulty label audited with human verification. |
| `inferential_validity_band` | string | External Gemini inferential-validity label audited with human verification. |
| `q_length` | integer | Question length. |
| `a_length` | integer | Answer length. |
| `ctx_length` | integer | Context length. |
| `ctx_sentence_count` | integer | Number of sentences in context. |
| `answer_position_ratio` | float | Approximate answer position in context. |
| `lexical_overlap_ratio` | float | Token overlap between question and answer. |
| `ttr_question` | float | Type-token ratio for the question. |
| `question_type` | string | Question-form category. |
| `answer_type` | string | Answer-type category. |
| `popularity_source` | string | Popularity source category, when available. |
| `knowledge_difficulty` | float | Estimated knowledge difficulty signal. |
| `page_views_rank` | integer | Wikipedia page-view rank. |
| `wiki_count_rank` | integer | Wiki-link or wiki-count rank. |
| `statements_rank` | integer | Wikidata statement-count rank. |

The final matrix is pruned for multicollinearity before release.
