"""Restore pre-repair QA payload while keeping repaired succinct_context only."""

from __future__ import annotations

import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import QA_ARCHIVE_DIR, QA_CANONICAL, ensure_dirs  # noqa: E402

MERGE_MANIFEST = ROOT / "data/processed/reports/qa/qa_pairs_with_topup_round2_succinct_repair_merge_manifest.json"
RUNS_REPAIR_DIR = ROOT / "data/processed/runs/qa/repair_succinct"
REPORT_PATH = ROOT / "data/processed/reports/qa/qa_restore_pre_repair_with_succinct_only_report.json"
ARCHIVED_PRE_REPAIR_BACKUP = (
    ROOT / "data/processed/archive/qa/qa_pairs_with_topup_round2.before_succinct_repair_20260601.jsonl"
)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def sample_key(row: Dict[str, Any]) -> Tuple[str, str]:
    return (str(row.get("chunk_id", "")), str(row.get("reasoning_type", "")))


def resolve_repair_dir(manifest: Dict[str, Any]) -> Path:
    candidate = Path(str(manifest.get("repair_dir", "")))
    if candidate.exists():
        return candidate
    return RUNS_REPAIR_DIR


def resolve_backup_path(manifest: Dict[str, Any]) -> Path:
    candidate = Path(str(manifest.get("backup_path", "")))
    if candidate.exists():
        return candidate
    return ARCHIVED_PRE_REPAIR_BACKUP


def load_repaired_rows(repair_dir: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in sorted(repair_dir.glob("qa_repair_succinct_*.jsonl")):
        if "_rejects" in path.name:
            continue
        rows.extend(load_jsonl(path))
    return rows


def rebuild_context(base_context: str, base_succinct: str, repaired_succinct: str) -> str:
    base_context = str(base_context or "")
    base_succinct = str(base_succinct or "")
    repaired_succinct = str(repaired_succinct or "")

    if not repaired_succinct:
        return base_context

    if base_succinct and base_context.startswith(base_succinct):
        suffix = base_context[len(base_succinct) :]
        return repaired_succinct + suffix

    if base_context.startswith(repaired_succinct):
        return base_context

    if base_context:
        return f"{repaired_succinct}\n\n{base_context}"
    return repaired_succinct


def main() -> None:
    ensure_dirs()

    manifest = json.loads(MERGE_MANIFEST.read_text(encoding="utf-8"))
    backup_path = resolve_backup_path(manifest)
    repair_dir = resolve_repair_dir(manifest)

    current_rows = load_jsonl(QA_CANONICAL)
    backup_rows = load_jsonl(backup_path)
    repaired_rows = load_repaired_rows(repair_dir)

    repaired_by_key = {sample_key(row): row for row in repaired_rows}

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_current_path = (
        QA_ARCHIVE_DIR
        / f"{QA_CANONICAL.stem}.before_restore_pre_repair_with_succinct_only_{timestamp}{QA_CANONICAL.suffix}"
    )
    backup_current_path.parent.mkdir(parents=True, exist_ok=True)
    backup_current_path.write_text(QA_CANONICAL.read_text(encoding="utf-8"), encoding="utf-8")

    restored_rows: List[Dict[str, Any]] = []
    restored_count = 0
    restored_by_reasoning = Counter()
    missing_current = 0
    examples: List[Dict[str, Any]] = []

    current_by_key = {sample_key(row): row for row in current_rows}

    for base in backup_rows:
        key = sample_key(base)
        current = current_by_key.get(key)
        repaired = repaired_by_key.get(key)
        if not current:
            missing_current += 1
            current = base
        if not repaired:
            restored_rows.append(base)
            continue

        merged = dict(base)
        merged["succinct_context"] = repaired.get("succinct_context", base.get("succinct_context", ""))
        merged["context"] = rebuild_context(
            str(base.get("context", "")),
            str(base.get("succinct_context", "")),
            str(merged.get("succinct_context", "")),
        )

        # Preserve only the intended field-level repair; keep original QA payload.
        restored_rows.append(merged)
        restored_count += 1
        restored_by_reasoning[str(merged.get("reasoning_type", ""))] += 1

        if len(examples) < 10:
            examples.append(
                {
                    "chunk_id": merged.get("chunk_id"),
                    "reasoning_type": merged.get("reasoning_type"),
                    "question_before_restore": current.get("question"),
                    "question_after_restore": merged.get("question"),
                    "answer_before_restore": current.get("answer"),
                    "answer_after_restore": merged.get("answer"),
                    "succinct_before_restore": current.get("succinct_context"),
                    "succinct_after_restore": merged.get("succinct_context"),
                }
            )

    write_jsonl(QA_CANONICAL, restored_rows)

    report = {
        "restored_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "canonical_path": str(QA_CANONICAL),
        "canonical_backup_before_restore": str(backup_current_path),
        "pre_repair_backup_path": str(backup_path),
        "repair_dir": str(repair_dir),
        "row_count": len(restored_rows),
        "repaired_row_count": len(repaired_rows),
        "restored_from_pre_repair_with_succinct_only": restored_count,
        "restored_by_reasoning_type": dict(restored_by_reasoning),
        "missing_current_keys_against_backup": missing_current,
        "examples": examples,
    }
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
