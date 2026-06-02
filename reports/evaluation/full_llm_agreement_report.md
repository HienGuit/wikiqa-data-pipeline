# Full LLM Agreement Report

This report computes Gemini vs DeepSeek agreement on the full shared judged population, not on the stratified human-verification stress-test sample.

- Shared rows across judges: `8005`
- Shared multi-sentence rows: `2378`

| Dimension | Rows | % Agreement | Expected % | Cohen's kappa | Interpretation |
|---|---:|---:|---:|---:|---|
| Quality Band | 8,005 | 65.86% | 60.33% | 0.1394 | slight agreement |
| Difficulty Band | 8,005 | 84.17% | 76.21% | 0.3346 | fair agreement |
| Inferential Validity Band | 2,378 | 63.92% | 56.12% | 0.1777 | slight agreement |

## Notes
- `quality_band` and `difficulty_band` are computed on all shared rows.
- `inferential_validity_band` is computed only on shared `multi-sentence` rows because the label is not defined for extraction samples.