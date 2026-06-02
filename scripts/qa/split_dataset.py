"""Split the final QA dataset into train/validation/test sets.

Groups by Wikipedia title to prevent data leakage.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import (  # noqa: E402
    QA_SPLIT_DISTRIBUTION_REPORT,
    QA_TEST_SPLIT,
    QA_THREE_WAY_READY,
    QA_TRAIN_SPLIT,
    QA_VAL_SPLIT,
    ensure_dirs,
)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Split dataset into Train/Val/Test preventing title leakage.")
    parser.add_argument("--input", default=str(QA_THREE_WAY_READY), help="Input dataset")
    parser.add_argument("--train-out", default=str(QA_TRAIN_SPLIT), help="Train split output")
    parser.add_argument("--val-out", default=str(QA_VAL_SPLIT), help="Validation split output")
    parser.add_argument("--test-out", default=str(QA_TEST_SPLIT), help="Test split output")
    parser.add_argument("--report-out", default=str(QA_SPLIT_DISTRIBUTION_REPORT), help="Split report output")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    return parser


def group_by_title(rows: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        title = str(row.get("title", "")).strip()
        if not title:
            title = "UNKNOWN_TITLE"
        groups.setdefault(title, []).append(row)
    return groups


def allocate_groups(
    groups: Dict[str, List[Dict[str, Any]]], ratios: List[float], seed: int
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    random.seed(seed)
    keys = list(groups.keys())
    # Shuffle first to ensure randomness among groups of the same size
    random.shuffle(keys)
    # Sort by size descending for optimal greedy packing
    keys.sort(key=lambda k: len(groups[k]), reverse=True)

    total_items = sum(len(groups[k]) for k in keys)
    targets = [total_items * r for r in ratios]
    current_counts = [0, 0, 0]
    splits: tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]] = ([], [], [])

    for k in keys:
        group_rows = groups[k]
        size = len(group_rows)
        # Find the bucket with the largest deficit
        deficits = [targets[i] - current_counts[i] for i in range(3)]
        best_idx = deficits.index(max(deficits))

        splits[best_idx].extend(group_rows)
        current_counts[best_idx] += size

    # Shuffle the resulting splits so rows aren't perfectly ordered by group size
    for split_rows in splits:
        random.shuffle(split_rows)

    return splits


def compute_distribution(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "rows": len(rows),
        "unique_titles": len(set(str(r.get("title", "")) for r in rows)),
        "reasoning_buckets": dict(Counter(r.get("final_reasoning_bucket", "missing") for r in rows)),
        "quality_bands": dict(Counter(r.get("quality_band", "missing") for r in rows)),
        "domains": dict(Counter(r.get("domain", "missing") for r in rows)),
    }


def main() -> None:
    ensure_dirs()
    args = build_parser().parse_args()

    ratios = [args.train_ratio, args.val_ratio, args.test_ratio]
    if not math.isclose(sum(ratios), 1.0):
        raise ValueError(f"Ratios must sum to 1.0, got {sum(ratios)}")

    print(f"Loading dataset from {args.input}...")
    rows = load_jsonl(Path(args.input))
    print(f"Loaded {len(rows)} rows.")

    groups = group_by_title(rows)
    print(f"Grouped into {len(groups)} unique titles.")

    train_rows, val_rows, test_rows = allocate_groups(groups, ratios, args.seed)
    print(f"Allocation complete: Train={len(train_rows)}, Val={len(val_rows)}, Test={len(test_rows)}")

    write_jsonl(Path(args.train_out), train_rows)
    write_jsonl(Path(args.val_out), val_rows)
    write_jsonl(Path(args.test_out), test_rows)

    report = {
        "input_path": args.input,
        "seed": args.seed,
        "target_ratios": {"train": args.train_ratio, "val": args.val_ratio, "test": args.test_ratio},
        "overall": compute_distribution(rows),
        "splits": {
            "train": compute_distribution(train_rows),
            "val": compute_distribution(val_rows),
            "test": compute_distribution(test_rows),
        },
    }

    report_path = Path(args.report_out)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved report to {args.report_out}")


if __name__ == "__main__":
    main()
