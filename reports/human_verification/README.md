# Human Verification Bundle

This bundle packages the latest dual-judge human-verification tasks,
Gemini and DeepSeek reference keys, and two annotator result sets.

## Structure

- `tasks/task1.json`: combined Task 1 payload with Gemini key, DeepSeek key, annotator1, annotator2
- `tasks/task2.json`: combined Task 2 payload with Gemini key, DeepSeek key, annotator1, annotator2
- `keys/gemini/*.json`: Gemini reference labels
- `keys/deepseek/*.json`: DeepSeek reference labels
- `annotations/annotator1/*.json`: annotator 1 labels
- `annotations/annotator2/*.json`: annotator 2 labels
- `reports/assembly_report.json`: provenance, alignment, and repair details
- `manifest.json`: machine-readable bundle manifest

## Notes

- Task 2 annotator 1 source required deterministic repair on one malformed line.
- Task 1 now comes directly from the external annotation export with two annotations embedded per sample.
