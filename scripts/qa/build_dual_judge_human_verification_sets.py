"""Build new human-verification tasks from Gemini/DeepSeek joint label buckets."""

from __future__ import annotations

import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import (  # noqa: E402
    QA_CANONICAL_JUDGED_CONTEXT_CLEANED,
    QA_CANONICAL_JUDGED_DEEPSEEK_V4_FLASH_CONTEXT_CLEANED,
    QA_CANONICAL_JUDGED_GEMINI31_FLASH_LITE_CONTEXT_CLEANED,
    ensure_dirs,
)

OUTPUT_DIR = ROOT / "data/processed/datasets/human_verification_dual_judge_20260602"
REPORT_PATH = OUTPUT_DIR / "sampling_report.json"

TASK1_SIZE = 100
TASK2_SIZE = 50
SEED = 42

TASK1_QUOTAS: Dict[str, int] = {
    "Q-A": 20,  # Gemini=strong AND DeepSeek=strong
    "Q-B": 15,  # Gemini=weak AND DeepSeek=weak
    "Q-C": 20,  # Gemini=strong, DeepSeek=usable
    "Q-D": 20,  # Gemini=usable, DeepSeek=weak
    "Q-E": 10,  # Gemini=strong, DeepSeek=weak
    "Q-F": 15,  # Gemini=usable AND DeepSeek=usable
}

TASK2_QUOTAS: Dict[str, int] = {
    "IV-A": 15,  # Gemini=weak AND DeepSeek=weak
    "IV-B": 12,  # Gemini=usable AND DeepSeek=usable
    "IV-C": 10,  # Gemini=strong, DeepSeek=usable
    "IV-D": 8,  # Gemini=usable, DeepSeek=weak
    "IV-E": 5,  # Gemini=strong, DeepSeek=weak
}


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


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


def quality_group(gemini: Dict[str, Any], deepseek: Dict[str, Any]) -> str | None:
    pair = (gemini.get("quality_band", ""), deepseek.get("quality_band", ""))
    return {
        ("strong", "strong"): "Q-A",
        ("weak", "weak"): "Q-B",
        ("strong", "usable"): "Q-C",
        ("usable", "weak"): "Q-D",
        ("strong", "weak"): "Q-E",
        ("usable", "usable"): "Q-F",
    }.get(pair)


def inferential_group(gemini: Dict[str, Any], deepseek: Dict[str, Any]) -> str | None:
    pair = (gemini.get("inferential_validity_band", ""), deepseek.get("inferential_validity_band", ""))
    return {
        ("weak", "weak"): "IV-A",
        ("usable", "usable"): "IV-B",
        ("strong", "usable"): "IV-C",
        ("usable", "weak"): "IV-D",
        ("strong", "weak"): "IV-E",
    }.get(pair)


def difficulty_flags(gemini: Dict[str, Any], deepseek: Dict[str, Any]) -> Dict[str, bool]:
    gem = str(gemini.get("difficulty_band", ""))
    deep = str(deepseek.get("difficulty_band", ""))
    return {
        "easy_anchor": gem == "easy" and deep == "easy",
        "medium_case": gem == "medium" or deep == "medium",
        "hard_anchor": gem == "hard" or deep == "hard",
        "difficulty_conflict": gem != deep,
    }


def merged_row(gemini: Dict[str, Any], deepseek: Dict[str, Any]) -> Dict[str, Any]:
    row = dict(gemini)
    row["_gemini_quality"] = gemini.get("quality_band", "")
    row["_gemini_difficulty"] = gemini.get("difficulty_band", "")
    row["_gemini_inferential"] = gemini.get("inferential_validity_band", "")
    row["_deepseek_quality"] = deepseek.get("quality_band", "")
    row["_deepseek_difficulty"] = deepseek.get("difficulty_band", "")
    row["_deepseek_inferential"] = deepseek.get("inferential_validity_band", "")
    row["_quality_group"] = quality_group(gemini, deepseek)
    row["_inferential_group"] = inferential_group(gemini, deepseek)
    row["_difficulty_flags"] = difficulty_flags(gemini, deepseek)
    return row


def build_joint_rows() -> List[Dict[str, Any]]:
    gemini_rows = load_jsonl(QA_CANONICAL_JUDGED_GEMINI31_FLASH_LITE_CONTEXT_CLEANED)
    deepseek_rows = load_jsonl(QA_CANONICAL_JUDGED_DEEPSEEK_V4_FLASH_CONTEXT_CLEANED)
    gemini_by_key = {sample_key(row): row for row in gemini_rows}
    deepseek_by_key = {sample_key(row): row for row in deepseek_rows}

    shared_keys = sorted(set(gemini_by_key) & set(deepseek_by_key))
    return [merged_row(gemini_by_key[key], deepseek_by_key[key]) for key in shared_keys]


def build_task1_annotation_row(sample_id: str, row: Dict[str, Any], group: str) -> Dict[str, Any]:
    return {
        "sample_id": sample_id,
        "task": "quality_difficulty_dual_judge",
        "bucket_group": group,
        "chunk_id": row.get("chunk_id"),
        "title": row.get("title"),
        "domain": row.get("domain"),
        "section": row.get("section"),
        "reasoning_type": row.get("reasoning_type"),
        "context": row.get("context"),
        "question": row.get("question"),
        "answer": row.get("answer"),
        "human_quality_band": "",
        "human_difficulty_band": "",
        "notes": "",
    }


def build_task2_annotation_row(sample_id: str, row: Dict[str, Any], group: str) -> Dict[str, Any]:
    return {
        "sample_id": sample_id,
        "task": "inferential_validity_dual_judge",
        "bucket_group": group,
        "chunk_id": row.get("chunk_id"),
        "title": row.get("title"),
        "domain": row.get("domain"),
        "section": row.get("section"),
        "reasoning_type": row.get("reasoning_type"),
        "context": row.get("context"),
        "question": row.get("question"),
        "answer": row.get("answer"),
        "human_inferential_validity_band": "",
        "notes": "",
    }


def build_task1_model_key(sample_id: str, row: Dict[str, Any], model: str) -> Dict[str, Any]:
    if model == "gemini":
        quality = row["_gemini_quality"]
        difficulty = row["_gemini_difficulty"]
        model_name = "gemini-3.1-flash-lite"
    else:
        quality = row["_deepseek_quality"]
        difficulty = row["_deepseek_difficulty"]
        model_name = "deepseek-v4-flash"
    return {
        "sample_id": sample_id,
        "bucket_group": row["_quality_group"],
        "chunk_id": row.get("chunk_id"),
        "reasoning_type": row.get("reasoning_type"),
        "judge_model": model_name,
        "quality_band_ref": quality,
        "difficulty_band_ref": difficulty,
    }


def build_task2_model_key(sample_id: str, row: Dict[str, Any], model: str) -> Dict[str, Any]:
    if model == "gemini":
        inferential = row["_gemini_inferential"]
        model_name = "gemini-3.1-flash-lite"
    else:
        inferential = row["_deepseek_inferential"]
        model_name = "deepseek-v4-flash"
    return {
        "sample_id": sample_id,
        "bucket_group": row["_inferential_group"],
        "chunk_id": row.get("chunk_id"),
        "reasoning_type": row.get("reasoning_type"),
        "judge_model": model_name,
        "inferential_validity_band_ref": inferential,
    }


def satisfies_task1_constraints(rows: Sequence[Dict[str, Any]]) -> bool:
    flags = Counter()
    for row in rows:
        row_flags = row["_difficulty_flags"]
        if row_flags["easy_anchor"]:
            flags["easy_anchor"] += 1
        if row_flags["medium_case"]:
            flags["medium_case"] += 1
        if row_flags["hard_anchor"]:
            flags["hard_anchor"] += 1
        if row_flags["difficulty_conflict"]:
            flags["difficulty_conflict"] += 1
    return (
        flags["easy_anchor"] >= 40
        and flags["medium_case"] >= 30
        and flags["hard_anchor"] >= 10
        and flags["difficulty_conflict"] >= 20
    )


def task1_constraint_counts(rows: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    flags = Counter()
    for row in rows:
        row_flags = row["_difficulty_flags"]
        for name, enabled in row_flags.items():
            if enabled:
                flags[name] += 1
    return dict(flags)


def flags_from_pair(gem_difficulty: str, deep_difficulty: str) -> Dict[str, int]:
    return {
        "easy_anchor": int(gem_difficulty == "easy" and deep_difficulty == "easy"),
        "medium_case": int(gem_difficulty == "medium" or deep_difficulty == "medium"),
        "hard_anchor": int(gem_difficulty == "hard" or deep_difficulty == "hard"),
        "difficulty_conflict": int(gem_difficulty != deep_difficulty),
    }


def add_counts(left: Dict[str, int], right: Dict[str, int]) -> Dict[str, int]:
    keys = ("easy_anchor", "medium_case", "hard_anchor", "difficulty_conflict")
    return {key: left.get(key, 0) + right.get(key, 0) for key in keys}


def plan_dominates(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    keys = ("easy_anchor", "medium_case", "hard_anchor", "difficulty_conflict")
    return all(a["counts"].get(key, 0) >= b["counts"].get(key, 0) for key in keys)


def pareto_plans(plans: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    frontier: List[Dict[str, Any]] = []
    for plan in plans:
        if any(plan_dominates(existing, plan) for existing in frontier):
            continue
        frontier = [existing for existing in frontier if not plan_dominates(plan, existing)]
        frontier.append(plan)
    frontier.sort(
        key=lambda plan: (
            plan["counts"]["hard_anchor"],
            plan["counts"]["medium_case"],
            plan["counts"]["difficulty_conflict"],
            plan["counts"]["easy_anchor"],
        ),
        reverse=True,
    )
    return frontier


def build_group_plans(group_rows: Sequence[Dict[str, Any]], quota: int, rng: random.Random) -> List[Dict[str, Any]]:
    rows_by_pair: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in group_rows:
        pair = (row["_gemini_difficulty"], row["_deepseek_difficulty"])
        rows_by_pair[pair].append(row)

    pairs = sorted(rows_by_pair)
    for pair_rows in rows_by_pair.values():
        rng.shuffle(pair_rows)

    plans: List[Dict[str, Any]] = []

    def rec(
        index: int, remaining: int, selected_counts: Dict[Tuple[str, str], int], current_counts: Dict[str, int]
    ) -> None:
        if index == len(pairs):
            if remaining == 0:
                plans.append(
                    {
                        "selected_counts": dict(selected_counts),
                        "counts": dict(current_counts),
                    }
                )
            return

        pair = pairs[index]
        available = min(len(rows_by_pair[pair]), remaining)
        pair_flags = flags_from_pair(*pair)
        for take in range(available + 1):
            next_selected = dict(selected_counts)
            if take:
                next_selected[pair] = take
            next_counts = dict(current_counts)
            if take:
                for name, value in pair_flags.items():
                    next_counts[name] = next_counts.get(name, 0) + value * take
            rec(index + 1, remaining - take, next_selected, next_counts)

    rec(0, quota, {}, {"easy_anchor": 0, "medium_case": 0, "hard_anchor": 0, "difficulty_conflict": 0})
    return pareto_plans(plans)


def sample_task1(rows: List[Dict[str, Any]], rng: random.Random) -> List[Dict[str, Any]]:
    by_group: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        group = row["_quality_group"]
        if group:
            by_group[group].append(row)

    group_plans: Dict[str, List[Dict[str, Any]]] = {}
    for group, quota in TASK1_QUOTAS.items():
        candidates = by_group[group]
        if len(candidates) < quota:
            raise RuntimeError(f"Not enough candidates for task1 group {group}")
        group_plans[group] = build_group_plans(candidates, quota, rng)

    groups = list(TASK1_QUOTAS)
    suffix_max: Dict[int, Dict[str, int]] = {}
    running = {"easy_anchor": 0, "medium_case": 0, "hard_anchor": 0, "difficulty_conflict": 0}
    for index in range(len(groups) - 1, -1, -1):
        group = groups[index]
        best_for_group = {"easy_anchor": 0, "medium_case": 0, "hard_anchor": 0, "difficulty_conflict": 0}
        for plan in group_plans[group]:
            for key, value in plan["counts"].items():
                best_for_group[key] = max(best_for_group[key], value)
        running = add_counts(running, best_for_group)
        suffix_max[index] = dict(running)
    suffix_max[len(groups)] = {"easy_anchor": 0, "medium_case": 0, "hard_anchor": 0, "difficulty_conflict": 0}

    target = {"easy_anchor": 40, "medium_case": 30, "hard_anchor": 10, "difficulty_conflict": 20}
    chosen_plan_map: Dict[str, Dict[str, Any]] = {}

    def feasible(index: int, counts: Dict[str, int]) -> bool:
        if index == len(groups):
            return all(counts.get(key, 0) >= value for key, value in target.items())
        remaining_best = suffix_max[index]
        for key, value in target.items():
            if counts.get(key, 0) + remaining_best.get(key, 0) < value:
                return False
        group = groups[index]
        for plan in group_plans[group]:
            next_counts = add_counts(counts, plan["counts"])
            if feasible(index + 1, next_counts):
                chosen_plan_map[group] = plan
                return True
        return False

    if not feasible(0, {"easy_anchor": 0, "medium_case": 0, "hard_anchor": 0, "difficulty_conflict": 0}):
        best_possible = {group: group_plans[group][0]["counts"] if group_plans[group] else {} for group in groups}
        raise RuntimeError(f"Could not satisfy task1 difficulty constraints. Best frontier heads: {best_possible}")

    chosen: List[Dict[str, Any]] = []
    for group in groups:
        rows_by_pair: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
        for row in by_group[group]:
            rows_by_pair[(row["_gemini_difficulty"], row["_deepseek_difficulty"])].append(row)
        for pair_rows in rows_by_pair.values():
            rng.shuffle(pair_rows)
        selected_counts = chosen_plan_map[group]["selected_counts"]
        for pair, count in selected_counts.items():
            chosen.extend(rows_by_pair[pair][:count])

    rng.shuffle(chosen)
    if not satisfies_task1_constraints(chosen):
        raise RuntimeError(
            f"Internal error: task1 selection failed constraints with counts {task1_constraint_counts(chosen)}"
        )
    return chosen


def sample_task2(
    rows: List[Dict[str, Any]], exclude_keys: set[Tuple[str, str, str, str]], rng: random.Random
) -> List[Dict[str, Any]]:
    by_group: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("reasoning_type") != "multi-sentence":
            continue
        group = row["_inferential_group"]
        if not group:
            continue
        key = sample_key(row)
        if key in exclude_keys:
            continue
        by_group[group].append(row)

    chosen: List[Dict[str, Any]] = []
    used = set(exclude_keys)
    for group, quota in TASK2_QUOTAS.items():
        candidates = by_group[group]
        rng.shuffle(candidates)
        picked = []
        for row in candidates:
            key = sample_key(row)
            if key in used:
                continue
            picked.append(row)
            used.add(key)
            if len(picked) == quota:
                break
        if len(picked) != quota:
            raise RuntimeError(f"Not enough candidates for task2 group {group}")
        chosen.extend(picked)
    rng.shuffle(chosen)
    return chosen


def distribution(rows: Sequence[Dict[str, Any]], field: str) -> Dict[str, int]:
    return dict(Counter(str(row.get(field, "")) for row in rows))


def group_distribution(rows: Sequence[Dict[str, Any]], field: str) -> Dict[str, int]:
    return dict(Counter(str(row[field]) for row in rows))


def main() -> None:
    ensure_dirs()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)

    joint_rows = build_joint_rows()
    task1_rows = sample_task1(joint_rows, rng)
    task1_keys = {sample_key(row) for row in task1_rows}
    task2_rows = sample_task2(joint_rows, task1_keys, rng)

    task1_annotation = [
        build_task1_annotation_row(f"DJ_T1_{i:03d}", row, row["_quality_group"])
        for i, row in enumerate(task1_rows, start=1)
    ]
    task1_gemini_key = [
        build_task1_model_key(f"DJ_T1_{i:03d}", row, "gemini") for i, row in enumerate(task1_rows, start=1)
    ]
    task1_deepseek_key = [
        build_task1_model_key(f"DJ_T1_{i:03d}", row, "deepseek") for i, row in enumerate(task1_rows, start=1)
    ]

    task2_annotation = [
        build_task2_annotation_row(f"DJ_T2_{i:03d}", row, row["_inferential_group"])
        for i, row in enumerate(task2_rows, start=1)
    ]
    task2_gemini_key = [
        build_task2_model_key(f"DJ_T2_{i:03d}", row, "gemini") for i, row in enumerate(task2_rows, start=1)
    ]
    task2_deepseek_key = [
        build_task2_model_key(f"DJ_T2_{i:03d}", row, "deepseek") for i, row in enumerate(task2_rows, start=1)
    ]

    write_jsonl(OUTPUT_DIR / "task1_quality_difficulty_100.jsonl", task1_annotation)
    write_jsonl(OUTPUT_DIR / "task1_quality_difficulty_100_key_gemini31_flash_lite.jsonl", task1_gemini_key)
    write_jsonl(OUTPUT_DIR / "task1_quality_difficulty_100_key_deepseek_v4_flash.jsonl", task1_deepseek_key)

    write_jsonl(OUTPUT_DIR / "task2_inferential_validity_50.jsonl", task2_annotation)
    write_jsonl(OUTPUT_DIR / "task2_inferential_validity_50_key_gemini31_flash_lite.jsonl", task2_gemini_key)
    write_jsonl(OUTPUT_DIR / "task2_inferential_validity_50_key_deepseek_v4_flash.jsonl", task2_deepseek_key)

    report = {
        "seed": SEED,
        "output_dir": str(OUTPUT_DIR),
        "source_gemini": str(QA_CANONICAL_JUDGED_GEMINI31_FLASH_LITE_CONTEXT_CLEANED),
        "source_deepseek": str(QA_CANONICAL_JUDGED_CONTEXT_CLEANED),
        "joint_rows": len(joint_rows),
        "task1": {
            "rows": len(task1_rows),
            "quality_group_distribution": group_distribution(task1_rows, "_quality_group"),
            "difficulty_constraint_counts": task1_constraint_counts(task1_rows),
            "gemini_quality_distribution": distribution(task1_gemini_key, "quality_band_ref"),
            "deepseek_quality_distribution": distribution(task1_deepseek_key, "quality_band_ref"),
            "gemini_difficulty_distribution": distribution(task1_gemini_key, "difficulty_band_ref"),
            "deepseek_difficulty_distribution": distribution(task1_deepseek_key, "difficulty_band_ref"),
        },
        "task2": {
            "rows": len(task2_rows),
            "inferential_group_distribution": group_distribution(task2_rows, "_inferential_group"),
            "gemini_inferential_distribution": distribution(task2_gemini_key, "inferential_validity_band_ref"),
            "deepseek_inferential_distribution": distribution(task2_deepseek_key, "inferential_validity_band_ref"),
        },
    }
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
