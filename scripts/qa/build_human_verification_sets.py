"""Build human-verification sample sets from the canonical cleaned judged pool."""

from __future__ import annotations

import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import (  # noqa: E402
    QA_ANNOTATION_POOL,
    QA_HUMAN_VERIFICATION_SAMPLING_REPORT,
    QA_HUMAN_VERIFICATION_TASK1,
    QA_HUMAN_VERIFICATION_TASK1_KEY,
    QA_HUMAN_VERIFICATION_TASK2,
    QA_HUMAN_VERIFICATION_TASK2_KEY,
    ensure_dirs,
)


TASK1_SIZE = 100
TASK2_SIZE = 50
SEED = 42


def load_rows(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def largest_remainder_quota(counts: Counter, sample_size: int) -> Dict[Any, int]:
    total = sum(counts.values())
    if total == 0:
        return {key: 0 for key in counts}

    quotas: Dict[Any, int] = {}
    remainders: List[Tuple[float, Any]] = []
    assigned = 0

    for key, count in counts.items():
        exact = (count / total) * sample_size
        base = int(exact)
        quotas[key] = base
        assigned += base
        remainders.append((exact - base, key))

    for _, key in sorted(remainders, reverse=True)[: sample_size - assigned]:
        quotas[key] += 1

    return quotas


def build_task1_annotation_row(sample_id: str, row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "sample_id": sample_id,
        "task": "quality_difficulty",
        "chunk_id": row.get("chunk_id"),
        "title": row.get("title"),
        "domain": row.get("domain"),
        "section": row.get("section"),
        "reasoning_type": row.get("reasoning_type"),
        "succinct_context": row.get("succinct_context"),
        "context": row.get("context"),
        "question": row.get("question"),
        "answer": row.get("answer"),
        "human_quality_band": "",
        "human_difficulty_band": "",
        "notes": "",
    }


def build_task2_annotation_row(sample_id: str, row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "sample_id": sample_id,
        "task": "inferential_validity",
        "chunk_id": row.get("chunk_id"),
        "title": row.get("title"),
        "domain": row.get("domain"),
        "section": row.get("section"),
        "reasoning_type": row.get("reasoning_type"),
        "succinct_context": row.get("succinct_context"),
        "context": row.get("context"),
        "question": row.get("question"),
        "answer": row.get("answer"),
        "human_inferential_validity_band": "",
        "notes": "",
    }


def build_task1_key_row(sample_id: str, row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "sample_id": sample_id,
        "chunk_id": row.get("chunk_id"),
        "reasoning_type": row.get("reasoning_type"),
        "quality_band_ref": row.get("quality_band"),
        "difficulty_band_ref": row.get("difficulty_band"),
    }


def build_task2_key_row(sample_id: str, row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "sample_id": sample_id,
        "chunk_id": row.get("chunk_id"),
        "reasoning_type": row.get("reasoning_type"),
        "inferential_validity_band_ref": row.get("inferential_validity_band"),
    }


def sample_by_quota(
    rows: List[Dict[str, Any]],
    quotas: Dict[Any, int],
    key_fn,
    rng: random.Random,
    exclude_ids: set[str] | None = None,
) -> List[Dict[str, Any]]:
    buckets: Dict[Any, List[Dict[str, Any]]] = defaultdict(list)
    exclude_ids = exclude_ids or set()

    for row in rows:
        if str(row.get("chunk_id")) in exclude_ids:
            continue
        buckets[key_fn(row)].append(row)

    selected: List[Dict[str, Any]] = []
    for bucket_key, quota in quotas.items():
        candidates = buckets.get(bucket_key, [])
        rng.shuffle(candidates)
        if len(candidates) < quota:
            raise RuntimeError(f"Not enough candidates for bucket {bucket_key}: need {quota}, have {len(candidates)}")
        selected.extend(candidates[:quota])
    rng.shuffle(selected)
    return selected


def main() -> None:
    ensure_dirs()
    rows = load_rows(QA_ANNOTATION_POOL)
    rng = random.Random(SEED)

    task1_counts = Counter((row.get("quality_band"), row.get("difficulty_band")) for row in rows)
    task1_quotas = largest_remainder_quota(task1_counts, TASK1_SIZE)
    task1_selected = sample_by_quota(rows, task1_quotas, lambda row: (row.get("quality_band"), row.get("difficulty_band")), rng)

    task1_ids = {str(row.get("chunk_id")) for row in task1_selected}
    inferential_rows = [row for row in rows if row.get("reasoning_type") == "multi-sentence"]
    task2_counts = Counter(row.get("inferential_validity_band") for row in inferential_rows)
    task2_quotas = largest_remainder_quota(task2_counts, TASK2_SIZE)
    task2_selected = sample_by_quota(
        inferential_rows,
        task2_quotas,
        lambda row: row.get("inferential_validity_band"),
        rng,
        exclude_ids=task1_ids,
    )

    task1_annotation = [build_task1_annotation_row(f"T1_{index:03d}", row) for index, row in enumerate(task1_selected, start=1)]
    task1_key = [build_task1_key_row(f"T1_{index:03d}", row) for index, row in enumerate(task1_selected, start=1)]
    task2_annotation = [build_task2_annotation_row(f"T2_{index:03d}", row) for index, row in enumerate(task2_selected, start=1)]
    task2_key = [build_task2_key_row(f"T2_{index:03d}", row) for index, row in enumerate(task2_selected, start=1)]

    write_jsonl(QA_HUMAN_VERIFICATION_TASK1, task1_annotation)
    write_jsonl(QA_HUMAN_VERIFICATION_TASK2, task2_annotation)
    write_jsonl(QA_HUMAN_VERIFICATION_TASK1_KEY, task1_key)
    write_jsonl(QA_HUMAN_VERIFICATION_TASK2_KEY, task2_key)

    report = {
        "seed": SEED,
        "input_path": str(QA_ANNOTATION_POOL),
        "task1": {
            "output_path": str(QA_HUMAN_VERIFICATION_TASK1),
            "key_path": str(QA_HUMAN_VERIFICATION_TASK1_KEY),
            "sample_size": TASK1_SIZE,
            "source_rows": len(rows),
            "distribution_source": {f"{quality}|{difficulty}": count for (quality, difficulty), count in sorted(task1_counts.items())},
            "distribution_sampled": dict(
                sorted(Counter(f"{row.get('quality_band_ref')}|{row.get('difficulty_band_ref')}" for row in task1_key).items())
            ),
        },
        "task2": {
            "output_path": str(QA_HUMAN_VERIFICATION_TASK2),
            "key_path": str(QA_HUMAN_VERIFICATION_TASK2_KEY),
            "sample_size": TASK2_SIZE,
            "source_rows": len(inferential_rows),
            "distribution_source": dict(sorted(task2_counts.items())),
            "distribution_sampled": dict(sorted(Counter(row.get("inferential_validity_band_ref") for row in task2_key).items())),
            "excluded_task1_chunk_ids": len(task1_ids),
        },
    }
    QA_HUMAN_VERIFICATION_SAMPLING_REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
