# Vietnamese QA Bucket Label Guideline

**Version:** 1.1  
**Scope:** External Gemini Annotation and Human Verification  
**Evaluation object:** Vietnamese question-answer pairs that passed rule-based validation  
**Main labels:** `quality_band`, `difficulty_band`, `inferential_validity_band`

This guideline standardizes bucket labeling for Vietnamese QA samples. It aligns the external LLM annotator and human annotators on the same rubric, while preserving rule-based validation as the first quality gate.

In the official pipeline, the external LLM annotator is Gemini. The QA generation model is not used to assign official labels, which avoids same-model evaluation bias.

## 1. Task Overview

Each sample contains:

- `context`: source text used as evidence
- `question`: generated Vietnamese question
- `answer`: concise answer grounded in the context
- `reasoning_type`: initial generation mode, usually `extraction` or `multi-sentence`

The goal is to decide whether the QA pair is clear, useful, grounded in the context, and correctly characterized by its reasoning label.

## 2. Labels

| Label | Values | Applies to | Purpose |
|---|---|---|---|
| `quality_band` | `weak`, `usable`, `strong` | all QA samples | Overall QA usefulness and clarity |
| `difficulty_band` | `easy`, `medium`, `hard` | internal analysis | Difficulty of answering from context |
| `inferential_validity_band` | `weak`, `usable`, `strong` | multi-sentence samples | Whether multiple sentences are genuinely needed |

When a case sits between two buckets, choose the lower bucket unless the evidence clearly supports the higher one.

## 3. Rule-Based Validation Comes First

Bucket labels do not replace deterministic validation. Samples should be rejected or repaired before annotation when they have:

- empty, unsupported, or overlong answers
- questions that reveal the answer or ask for multiple targets
- evidence spans that do not contain the answer
- multi-sentence evidence that is actually answerable from one sentence
- answer type mismatches, such as asking for a duration but giving a single date

## 4. `quality_band`

| Bucket | Definition | Typical signs |
|---|---|---|
| `weak` | Formally valid but awkward, unclear, or weakly useful. | Mechanical wording, poor paraphrase, answer does not read naturally with the question |
| `usable` | Clear, grounded, and usable for training or evaluation. | Understandable question, matching answer, no serious semantic issue |
| `strong` | Natural, concise, representative, and tightly grounded. | Good paraphrase, independent question, precise answer |

Do not assign `strong` only because the topic sounds academic. Focus on the quality of the QA pair.

## 5. `difficulty_band`

| Bucket | Definition | Typical signs |
|---|---|---|
| `easy` | The answer is obvious from one sentence or a prominent phrase. | Named entity, date, number, or near-copy question |
| `medium` | The reader must inspect the context carefully or connect simple clues. | Clear paraphrase, answer is not too obvious, mild distractors |
| `hard` | The reader must synthesize or reject competing information. | Multiple entities, timelines, distractors, or partially correct alternatives |

`difficulty_band` is a diagnostic/internal label, not a public target label in the final release.

## 6. `inferential_validity_band`

This label applies only to multi-sentence samples.

| Bucket | Definition | Sentence removal test |
|---|---|---|
| `weak` | One sentence is enough, or the second sentence is ornamental. | Removing the extra sentence still leaves a certain answer |
| `usable` | At least two ideas are needed, but the connection is straightforward. | Removing one sentence lowers confidence but may still allow a guess |
| `strong` | Multiple sentences are essential. | Removing a key sentence makes the answer unsupported |

Use the sentence removal test: hide one evidence sentence at a time and ask whether a careful reader can still answer confidently.

## 7. Guidance for External LLM Annotator

The external LLM annotator must:

- use only the given context, question, answer, and guideline
- return only labels from the allowed bucket set
- avoid inventing labels or adding unsupported assumptions
- choose the lower bucket when evidence is ambiguous
- treat multi-sentence validity as an evidence requirement, not just a surface label

In the official pipeline, this external annotator is Gemini. Gemini assigns labels independently from the original text evidence and does not inherit labels from the QA generation model.

### Suggested Output Schema For Extraction

```json
{
  "quality_band": "weak | usable | strong",
  "difficulty_band": "easy | medium | hard"
}
```

### Suggested Output Schema For Multi-Sentence

```json
{
  "quality_band": "weak | usable | strong",
  "difficulty_band": "easy | medium | hard",
  "inferential_validity_band": "weak | usable | strong"
}
```

## 8. Guidance For Human Verification

Human verification audits the reliability of the rubric and the external annotation pass.

| Step | Action | Check |
|---|---|---|
| 1 | Read full context, question, and answer. | Do not label from question or answer alone. |
| 2 | Assign `quality_band`. | Is the QA pair natural, clear, and grounded? |
| 3 | Assign `difficulty_band`. | How hard is the answer to find from context? |
| 4 | For multi-sentence samples, assign `inferential_validity_band`. | Does the question require more than one sentence of evidence? |

## 9. Agreement And Calibration

Before broad use, sampled cases should be labeled independently by two human annotators. Agreement is reported for:

- Annotator 1 vs Annotator 2
- Annotator 1 vs Gemini annotation
- Annotator 2 vs Gemini annotation

Low agreement should be discussed as a limitation and used to refine the guideline.

## 10. Pipeline Use

- Use `quality_band` as the main signal for retaining or prioritizing samples.
- Use `difficulty_band` for diagnostics and EDA.
- Use `inferential_validity_band` to review multi-sentence samples and map weak inference cases into `bridge` when appropriate.
- Prefer `usable` and `strong`; inspect or downgrade `weak` cases.
