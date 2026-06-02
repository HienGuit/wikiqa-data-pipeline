"""Compute full-population Gemini vs DeepSeek agreement on shared judged samples."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import (  # noqa: E402
    QA_CANONICAL_JUDGED_DEEPSEEK_V4_FLASH_CONTEXT_CLEANED,
    QA_CANONICAL_JUDGED_GEMINI31_FLASH_LITE_CONTEXT_CLEANED,
    QA_REPORTS_DIR,
    ensure_dirs,
)
from src.qa.iaa import compute_cohen_kappa  # noqa: E402

REPORT_JSON = QA_REPORTS_DIR / "full_llm_agreement_report.json"
REPORT_MD = QA_REPORTS_DIR / "full_llm_agreement_report.md"


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def normalize_text(value: Any) -> str:
    text = "" if value is None else str(value)
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def sample_key(row: Dict[str, Any]) -> Tuple[str, str, str, str]:
    return (
        str(row.get("chunk_id", "")),
        str(row.get("reasoning_type", "")),
        normalize_text(row.get("question", "")),
        normalize_text(row.get("answer", "")),
    )


def distribution(rows: Iterable[Dict[str, Any]], field: str) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for row in rows:
        value = str(row.get(field, ""))
        counts[value] = counts.get(value, 0) + 1
    return counts


def build_dimension_report(
    *,
    rows: List[Tuple[Dict[str, Any], Dict[str, Any]]],
    field: str,
    labels: List[str],
    dimension_name: str,
) -> Dict[str, Any]:
    gemini_values = [str(g.get(field, "")).strip().lower() for g, _ in rows]
    deepseek_values = [str(d.get(field, "")).strip().lower() for _, d in rows]
    result = compute_cohen_kappa(gemini_values, deepseek_values, labels)
    return {
        "dimension": dimension_name,
        "field": field,
        "rows": len(rows),
        "gemini_distribution": dict(sorted(distribution((g for g, _ in rows), field).items())),
        "deepseek_distribution": dict(sorted(distribution((d for _, d in rows), field).items())),
        **result,
    }


def render_markdown(report: Dict[str, Any]) -> str:
    lines = [
        "# Full LLM Agreement Report",
        "",
        "This report computes Gemini vs DeepSeek agreement on the full shared judged population, not on the stratified human-verification stress-test sample.",
        "",
        f"- Shared rows across judges: `{report['shared_rows']}`",
        f"- Shared multi-sentence rows: `{report['shared_multi_sentence_rows']}`",
        "",
        "| Dimension | Rows | % Agreement | Expected % | Cohen's kappa | Interpretation |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for key in ("quality_band", "difficulty_band", "inferential_validity_band"):
        block = report["dimensions"][key]
        lines.append(
            f"| {block['dimension']} | {block['rows']:,} | {block['observed_agreement_percent']:.2f}% | "
            f"{block['expected_agreement_percent']:.2f}% | {block['cohen_kappa']:.4f} | {block['kappa_interpretation']} |"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "- `quality_band` and `difficulty_band` are computed on all shared rows.",
            "- `inferential_validity_band` is computed only on shared `multi-sentence` rows because the label is not defined for extraction samples.",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    ensure_dirs()
    gemini_rows = load_jsonl(QA_CANONICAL_JUDGED_GEMINI31_FLASH_LITE_CONTEXT_CLEANED)
    deepseek_rows = load_jsonl(QA_CANONICAL_JUDGED_DEEPSEEK_V4_FLASH_CONTEXT_CLEANED)

    gemini_by_key = {sample_key(row): row for row in gemini_rows}
    deepseek_by_key = {sample_key(row): row for row in deepseek_rows}
    shared_keys = sorted(set(gemini_by_key) & set(deepseek_by_key))
    shared_pairs = [(gemini_by_key[key], deepseek_by_key[key]) for key in shared_keys]
    shared_multi_pairs = [
        (g, d) for g, d in shared_pairs if str(g.get("reasoning_type", "")) == "multi-sentence"
    ]

    report = {
        "sources": {
            "gemini": str(QA_CANONICAL_JUDGED_GEMINI31_FLASH_LITE_CONTEXT_CLEANED),
            "deepseek": str(QA_CANONICAL_JUDGED_DEEPSEEK_V4_FLASH_CONTEXT_CLEANED),
        },
        "shared_rows": len(shared_pairs),
        "shared_multi_sentence_rows": len(shared_multi_pairs),
        "dimensions": {
            "quality_band": build_dimension_report(
                rows=shared_pairs,
                field="quality_band",
                labels=["weak", "usable", "strong"],
                dimension_name="Quality Band",
            ),
            "difficulty_band": build_dimension_report(
                rows=shared_pairs,
                field="difficulty_band",
                labels=["easy", "medium", "hard"],
                dimension_name="Difficulty Band",
            ),
            "inferential_validity_band": build_dimension_report(
                rows=shared_multi_pairs,
                field="inferential_validity_band",
                labels=["weak", "usable", "strong"],
                dimension_name="Inferential Validity Band",
            ),
        },
    }

    REPORT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    REPORT_MD.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
