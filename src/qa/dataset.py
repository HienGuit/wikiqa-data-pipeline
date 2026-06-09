"""Dataset selection and merge utilities for QA generation."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from src.config import (
    CHUNKS_TOPUP_MANIFEST,
    FILTERED_CHUNKS,
    INTERIM_DIR,
    QA_ANNOTATED_FLASH,
    QA_ANNOTATED_FLASH_REJECTS,
    QA_ANNOTATION_FLASH_EXTRACTION_DIR,
    QA_ANNOTATION_FLASH_MULTI_DIR,
    QA_ANNOTATION_FULL_FLASH_SUMMARY,
    QA_CANONICAL,
    QA_CANONICAL_ANNOTATED,
    QA_FULL_RUN_SUMMARY,
    QA_INFERENTIAL_USABLE_ONLY,
    QA_RAW,
    QA_RAW_REJECTS,
    QA_REPORTS_DIR,
    QA_SHARDS_DIR,
    QA_SPLIT_READY,
    QA_TOPUP_RUN_DIR,
    QA_TOPUP_RUN_DIR_ROUND2,
    QA_WITH_TOPUP_ROUND2,
    QA_WITH_TOPUP_ROUND2_REJECTS,
    QA_WITH_TOPUP_ROUND2_SUMMARY,
    SAMPLED_CHUNKS,
    ensure_dirs,
)
from src.qa.validators import count_sentences

WORD_RE = re.compile(r"\w+", re.UNICODE)

LOGIC_MARKERS = (
    "vì",
    "do đó",
    "do vậy",
    "bởi vậy",
    "khiến",
    "nên",
    "nhờ",
    "dẫn đến",
)
COMPARISON_MARKERS = (
    "trong khi",
    "khác với",
    "so với",
    "hơn",
    "kém",
    "tương tự",
    "ngược lại",
)
TEMPORAL_MARKERS = (
    "sau đó",
    "trước đó",
    "tiếp theo",
    "cuối cùng",
    "ban đầu",
    "đồng thời",
    "về sau",
)
AGGREGATION_MARKERS = (
    "bao gồm",
    "gồm",
    "ngoài ra",
    "đồng thời",
    "cùng với",
    "mặt khác",
    "bên cạnh đó",
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def load_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    file_path = Path(path)
    if not file_path.exists():
        return []
    with file_path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def count_markers(text: str, markers: Tuple[str, ...]) -> int:
    lowered = (text or "").lower()
    return sum(1 for marker in markers if marker in lowered)


def lexical_diversity(text: str) -> float:
    tokens = WORD_RE.findall((text or "").lower())
    if not tokens:
        return 0.0
    return len(set(tokens)) / len(tokens)


def char_count(row: Dict[str, Any]) -> int:
    if isinstance(row.get("char_count"), int):
        return int(row["char_count"])
    return len(str(row.get("text", "")))


def stable_hash(seed: int, chunk_id: str) -> int:
    value = f"{seed}:{chunk_id}"
    total = 0
    for char in value:
        total = (total * 131 + ord(char)) % 1_000_000_007
    return total


def domain_quota(domains: List[str], target_total: int) -> Dict[str, int]:
    base = target_total // len(domains)
    remainder = target_total % len(domains)
    quotas = {domain: base for domain in domains}
    for domain in domains[:remainder]:
        quotas[domain] += 1
    return quotas


def inferential_score(row: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
    text = str(row.get("text", ""))
    section = str(row.get("section", "") or "")
    context_above = str(row.get("context_above", "") or "")
    sentence_count = count_sentences(text)
    chars = char_count(row)
    lex_div = lexical_diversity(text)
    logic_hits = count_markers(text, LOGIC_MARKERS)
    comparison_hits = count_markers(text, COMPARISON_MARKERS)
    temporal_hits = count_markers(text, TEMPORAL_MARKERS)
    aggregation_hits = count_markers(text, AGGREGATION_MARKERS)

    score = 0
    if sentence_count >= 3:
        score += 2
    if sentence_count >= 4:
        score += 2
    if chars >= 600:
        score += 1
    if chars >= 850:
        score += 1
    if context_above:
        score += 1
    if section:
        score += 1
    if int(row.get("chunk_index", 0) or 0) >= 1:
        score += 1
    score += min(logic_hits + comparison_hits + temporal_hits + aggregation_hits, 4)
    if "\n" in text:
        score += 1
    if lex_div >= 0.45:
        score += 1

    meta = {
        "sentence_count": sentence_count,
        "char_count": chars,
        "logic_hits": logic_hits,
        "comparison_hits": comparison_hits,
        "temporal_hits": temporal_hits,
        "aggregation_hits": aggregation_hits,
        "lexical_diversity": round(lex_div, 4),
        "has_context_above": bool(context_above),
        "is_intro": not bool(section),
    }
    return score, meta


def select_rows(
    rows: List[Dict[str, Any]], quotas: Dict[str, int], seed: int
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    grouped: dict[str, list[tuple[tuple[Any, ...], Dict[str, Any], Dict[str, Any]]]] = defaultdict(list)
    debug_summary: dict[str, Any] = {"domain_candidates": {}, "domain_selected": {}, "top_examples": {}}

    for row in rows:
        score, meta = inferential_score(row)
        chunk_id = str(row.get("chunk_id", ""))
        tie = stable_hash(seed, chunk_id)
        sort_key = (
            -score,
            -meta["sentence_count"],
            -meta["char_count"],
            -(meta["logic_hits"] + meta["comparison_hits"] + meta["temporal_hits"] + meta["aggregation_hits"]),
            -int(meta["has_context_above"]),
            tie,
            chunk_id,
        )
        grouped[str(row.get("domain", ""))].append((sort_key, row, meta))

    selected: List[Dict[str, Any]] = []
    for domain, quota in quotas.items():
        ranked = sorted(grouped.get(domain, []), key=lambda item: item[0])
        debug_summary["domain_candidates"][domain] = len(ranked)
        picks = ranked[:quota]
        debug_summary["domain_selected"][domain] = len(picks)
        debug_summary["top_examples"][domain] = [
            {
                "chunk_id": str(row.get("chunk_id", "")),
                "title": row.get("title", ""),
                "section": row.get("section", ""),
                "score": -sort_key[0],
                "sentence_count": meta["sentence_count"],
                "char_count": meta["char_count"],
            }
            for sort_key, row, meta in picks[:5]
        ]
        selected.extend(row for _, row, _ in picks)
    return selected, debug_summary


def shard_ids(input_dir: Path) -> List[str]:
    ids = []
    for path in sorted(input_dir.glob("qa_batch_shard_*.jsonl")):
        if path.name.endswith("_rejects.jsonl"):
            continue
        ids.append(path.stem.replace("qa_batch_shard_", ""))
    return ids


def split_projection(count: int) -> Dict[str, int]:
    train_count = int(round(count * 0.8))
    val_count = int(round(count * 0.1))
    return {
        "train_80": train_count,
        "val_10": val_count,
        "test_10": count - train_count - val_count,
    }


def ensure_parent_dirs(*paths: Path) -> None:
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)


def extend_unique_pairs(
    target_rows: List[Dict[str, Any]],
    incoming_rows: List[Dict[str, Any]],
    seen_pairs: set[tuple[str, str]],
    error_prefix: str,
) -> None:
    for row in incoming_rows:
        pair = (str(row.get("chunk_id", "")), str(row.get("reasoning_type", "")))
        if pair in seen_pairs:
            raise SystemExit(f"{error_prefix}: {pair}")
        seen_pairs.add(pair)
    target_rows.extend(incoming_rows)


def collect_domain_type_counts(rows: List[Dict[str, Any]]) -> tuple[Counter, Counter, dict[str, Counter]]:
    type_counts = Counter()
    domain_counts = Counter()
    domain_type_counts: dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        reasoning_type = str(row.get("reasoning_type", ""))
        domain = str(row.get("domain", ""))
        type_counts[reasoning_type] += 1
        domain_counts[domain] += 1
        domain_type_counts[domain][reasoning_type] += 1
    return type_counts, domain_counts, domain_type_counts


def build_domain_balance(
    domain_type_counts: dict[str, Counter], include_split_projection: bool = False
) -> Dict[str, Dict[str, Any]]:
    domain_balance: Dict[str, Dict[str, Any]] = {}
    for domain, counts in sorted(domain_type_counts.items()):
        extraction_count = counts.get("extraction", 0)
        multi_count = counts.get("multi-sentence", 0)
        item: Dict[str, Any] = {
            "extraction": extraction_count,
            "multi_sentence": multi_count,
            "inferential_deficit": max(0, extraction_count - multi_count),
        }
        if include_split_projection:
            item["split_projection_80_10_10"] = split_projection(multi_count)
        domain_balance[domain] = item
    return domain_balance


def write_summary(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def run_select_topup(args: argparse.Namespace) -> None:
    ensure_dirs()

    input_path = Path(args.input)
    exclude_paths = [Path(value) for value in args.exclude]
    output_path = Path(args.output)
    manifest_path = Path(args.manifest)
    ensure_parent_dirs(output_path, manifest_path)

    started_at = utc_now_iso()
    source_rows = load_jsonl(input_path)
    excluded_ids = set()
    exclude_sizes = {}
    for exclude_path in exclude_paths:
        exclude_rows = load_jsonl(exclude_path)
        exclude_sizes[str(exclude_path.resolve())] = len(exclude_rows)
        excluded_ids.update(str(row.get("chunk_id", "")) for row in exclude_rows)

    filter_reasons = Counter()
    candidates: List[Dict[str, Any]] = []
    per_domain_after_filter = Counter()

    for row in source_rows:
        chunk_id = str(row.get("chunk_id", ""))
        domain = str(row.get("domain", ""))
        text = str(row.get("text", ""))
        chars = char_count(row)
        sentences = count_sentences(text)

        if chunk_id in excluded_ids:
            filter_reasons["already_in_sampled_pool"] += 1
            continue
        if chars < args.min_char_count:
            filter_reasons["below_min_char_count"] += 1
            continue
        if sentences < args.min_sentences:
            filter_reasons["below_min_sentences"] += 1
            continue

        candidates.append(row)
        per_domain_after_filter[domain] += 1

    domains = sorted(per_domain_after_filter)
    if not domains:
        raise SystemExit("Khong tim thay candidate nao sau khi loc.")

    quotas = domain_quota(domains, args.target_total)
    selected, debug_summary = select_rows(candidates, quotas, args.seed)

    if len(selected) < args.target_total:
        raise SystemExit(
            f"Chi chon duoc {len(selected)} chunk, nho hon target {args.target_total}. "
            "Hay giam quota hoac noi dieu kien loc."
        )
    if len({str(row.get("chunk_id", "")) for row in selected}) != len(selected):
        raise SystemExit("Top-up selection produced duplicate chunk_id values.")

    write_jsonl(output_path, selected)

    selected_counter = Counter(str(row.get("domain", "")) for row in selected)
    if selected_counter != Counter(quotas):
        raise SystemExit(f"Top-up selection did not satisfy quotas: {dict(selected_counter)} vs {quotas}")

    manifest = {
        "purpose": args.purpose,
        "input_path": str(input_path.resolve()),
        "exclude_paths": [str(path.resolve()) for path in exclude_paths],
        "exclude_sizes": exclude_sizes,
        "output_path": str(output_path.resolve()),
        "seed": args.seed,
        "target_total": args.target_total,
        "min_char_count": args.min_char_count,
        "min_sentences": args.min_sentences,
        "filter_reasons": dict(filter_reasons),
        "domains": domains,
        "domain_quotas": quotas,
        "candidate_pool_size": len(candidates),
        "candidate_pool_by_domain": dict(per_domain_after_filter),
        "selected_count": len(selected),
        "selected_by_domain": dict(selected_counter),
        "selection_debug": debug_summary,
        "avg_char_count_selected": round(sum(char_count(row) for row in selected) / len(selected), 2),
        "intro_count_selected": sum(1 for row in selected if not str(row.get("section", "") or "")),
        "with_context_above_selected": sum(1 for row in selected if str(row.get("context_above", "") or "").strip()),
        "started_at_utc": started_at,
        "finished_at_utc": utc_now_iso(),
    }
    write_summary(manifest_path, manifest)

    print(f"Selected {len(selected):,} top-up chunk(s) -> {output_path}")
    print(f"Manifest written -> {manifest_path}")


def run_merge_main(args: argparse.Namespace) -> None:
    ensure_dirs()

    input_dir = Path(args.input_dir)
    merged_output = Path(args.merged_output)
    reject_output = Path(args.reject_output)
    summary_output = Path(args.summary_output)
    ensure_parent_dirs(merged_output, reject_output, summary_output)

    shard_keys = shard_ids(input_dir)
    if not shard_keys:
        raise SystemExit(f"No shard outputs found in {input_dir}")

    accepted_rows: List[Dict[str, Any]] = []
    rejected_rows: List[Dict[str, Any]] = []
    per_shard: Dict[str, Dict[str, Any]] = {}
    reject_errors = Counter()
    seen_pairs: set[tuple[str, str]] = set()

    for key in shard_keys:
        shard_name = f"qa_batch_shard_{key}"
        valid_path = input_dir / f"{shard_name}.jsonl"
        reject_path = input_dir / f"{shard_name}_rejects.jsonl"
        report_path = input_dir / f"{shard_name}_report.json"

        valid_rows = load_jsonl(valid_path)
        reject_rows = load_jsonl(reject_path)
        extend_unique_pairs(accepted_rows, valid_rows, seen_pairs, "Duplicate accepted pair detected during merge")
        rejected_rows.extend(reject_rows)
        reject_errors.update(row["error"] for row in reject_rows)

        shard_type_counts = Counter(str(row.get("reasoning_type", "")) for row in valid_rows)
        report_data = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}
        per_shard[key] = {
            "accepted": len(valid_rows),
            "rejected": len(reject_rows),
            "type_counts": dict(shard_type_counts),
            "report": report_data,
        }

    write_jsonl(merged_output, accepted_rows)
    write_jsonl(reject_output, rejected_rows)

    type_counts, domain_counts, domain_type_counts = collect_domain_type_counts(accepted_rows)
    extraction_count = type_counts.get("extraction", 0)
    inferential_count = type_counts.get("multi-sentence", 0)

    summary = {
        "shard_count": len(shard_keys),
        "accepted_total": len(accepted_rows),
        "rejected_total": len(rejected_rows),
        "type_counts": dict(type_counts),
        "domain_counts": dict(domain_counts),
        "top_reject_errors": reject_errors.most_common(20),
        "inferential_balance": {
            "extraction": extraction_count,
            "multi_sentence": inferential_count,
            "inferential_deficit_to_balance": max(0, extraction_count - inferential_count),
        },
        "domain_balance": build_domain_balance(domain_type_counts),
        "per_shard": per_shard,
        "merged_output": str(merged_output.resolve()),
        "reject_output": str(reject_output.resolve()),
    }
    write_summary(summary_output, summary)

    print(f"Merged accepted -> {merged_output}")
    print(f"Merged rejects  -> {reject_output}")
    print(f"Summary         -> {summary_output}")
    print(f"Accepted total  : {len(accepted_rows)}")
    print(f"Rejected total  : {len(rejected_rows)}")


def run_merge_topup(args: argparse.Namespace) -> None:
    ensure_dirs()

    base_accepted = load_jsonl(Path(args.base_accepted))
    base_rejects = load_jsonl(Path(args.base_rejects))

    topup_dirs = [Path(value) for value in args.topup_dirs]
    topup_accepted: List[Dict[str, Any]] = []
    topup_rejects: List[Dict[str, Any]] = []
    per_batch_topup: Dict[str, Dict[str, Any]] = {}
    seen_pairs = {(str(row.get("chunk_id", "")), str(row.get("reasoning_type", ""))) for row in base_accepted}

    for topup_dir in topup_dirs:
        batch_key = topup_dir.name
        per_shard_topup: Dict[str, Dict[str, Any]] = {}
        batch_accepted = 0
        batch_rejected = 0

        for path in sorted(topup_dir.glob("qa_topup_inferential_*.jsonl")):
            if path.name.endswith("_rejects.jsonl"):
                continue
            shard_key = path.stem.replace("qa_topup_inferential_", "")
            reject_path = topup_dir / f"qa_topup_inferential_{shard_key}_rejects.jsonl"
            report_path = topup_dir / f"qa_topup_inferential_{shard_key}_report.json"
            accepted_rows = load_jsonl(path)
            rejected_rows = load_jsonl(reject_path)

            extend_unique_pairs(
                topup_accepted, accepted_rows, seen_pairs, "Duplicate accepted pair detected while merging top-up"
            )
            topup_rejects.extend(rejected_rows)
            batch_accepted += len(accepted_rows)
            batch_rejected += len(rejected_rows)
            per_shard_topup[shard_key] = {
                "accepted": len(accepted_rows),
                "rejected": len(rejected_rows),
                "report": json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {},
            }

        per_batch_topup[batch_key] = {
            "accepted": batch_accepted,
            "rejected": batch_rejected,
            "per_shard": per_shard_topup,
            "source_dir": str(topup_dir.resolve()),
        }

    merged_accepted = base_accepted + topup_accepted
    merged_rejects = base_rejects + topup_rejects

    merged_output = Path(args.merged_output)
    reject_output = Path(args.reject_output)
    summary_output = Path(args.summary_output)
    ensure_parent_dirs(merged_output, reject_output, summary_output)

    write_jsonl(merged_output, merged_accepted)
    write_jsonl(reject_output, merged_rejects)

    type_counts, domain_counts, domain_type_counts = collect_domain_type_counts(merged_accepted)
    reject_errors = Counter(row["error"] for row in merged_rejects)
    extraction_count = type_counts.get("extraction", 0)
    inferential_count = type_counts.get("multi-sentence", 0)
    min_multi = min((counts.get("multi-sentence", 0) for counts in domain_type_counts.values()), default=0)

    summary = {
        "base_batch": {"accepted": len(base_accepted), "rejected": len(base_rejects)},
        "topup_batch": {
            "accepted": len(topup_accepted),
            "rejected": len(topup_rejects),
            "per_batch": per_batch_topup,
        },
        "merged": {
            "accepted_total": len(merged_accepted),
            "rejected_total": len(merged_rejects),
            "type_counts": dict(type_counts),
            "domain_counts": dict(domain_counts),
            "top_reject_errors": reject_errors.most_common(20),
        },
        "inferential_balance": {
            "extraction": extraction_count,
            "multi_sentence": inferential_count,
            "inferential_deficit_to_balance": max(0, extraction_count - inferential_count),
        },
        "domain_balance": build_domain_balance(domain_type_counts, include_split_projection=True),
        "split_readiness": {
            "min_multi_sentence_per_domain": min_multi,
            "supports_balanced_domain_x_reasoning_stratification": bool(min_multi >= 200),
            "note": "Danh gia nay dua tren so multi-sentence toi thieu moi domain sau top-up.",
        },
        "merged_output": str(merged_output.resolve()),
        "reject_output": str(reject_output.resolve()),
    }
    write_summary(summary_output, summary)

    print(f"Merged accepted -> {merged_output}")
    print(f"Merged rejects  -> {reject_output}")
    print(f"Summary         -> {summary_output}")


def run_merge_annotation(args: argparse.Namespace) -> None:
    ensure_dirs()

    input_dirs = [Path(value) for value in args.input_dirs]
    merged_output = Path(args.merged_output)
    reject_output = Path(args.reject_output)
    summary_output = Path(args.summary_output)
    ensure_parent_dirs(merged_output, reject_output, summary_output)

    accepted_rows: List[Dict[str, Any]] = []
    rejected_rows: List[Dict[str, Any]] = []
    per_batch: Dict[str, Dict[str, Any]] = {}
    bucket_quality = Counter()
    bucket_difficulty = Counter()
    bucket_inferential = Counter()
    bucket_detected = Counter()
    by_reasoning: defaultdict[str, Dict[str, int]] = defaultdict(lambda: {"samples": 0, "accepted": 0, "rejected": 0})
    seen_pairs: set[tuple[str, str]] = set()

    for input_dir in input_dirs:
        batch_key = input_dir.name
        batch_accept = 0
        batch_reject = 0
        per_shard: Dict[str, Dict[str, Any]] = {}

        report_paths = sorted(input_dir.glob("qa_annotation_*_report.json"))
        if not report_paths:
            report_paths = sorted(input_dir.glob("qa_judge_*_report.json"))
        for report_path in report_paths:
            prefix = "qa_annotation_" if report_path.name.startswith("qa_annotation_") else "qa_judge_"
            shard_key = report_path.stem.replace(prefix, "").replace("_report", "")
            accepted_path = input_dir / f"{prefix}{shard_key}.jsonl"
            rejected_path = input_dir / f"{prefix}{shard_key}_rejects.jsonl"

            report = json.loads(report_path.read_text(encoding="utf-8"))
            accepted = load_jsonl(accepted_path)
            rejected = load_jsonl(rejected_path)

            extend_unique_pairs(
                accepted_rows,
                accepted,
                seen_pairs,
                "Duplicate annotation pair detected during merge",
            )
            rejected_rows.extend(rejected)
            batch_accept += len(accepted)
            batch_reject += len(rejected)

            reasoning_filter = str(report.get("reasoning_filter", ""))
            by_reasoning[reasoning_filter]["samples"] += int(report.get("sample_count", 0))
            by_reasoning[reasoning_filter]["accepted"] += len(accepted)
            by_reasoning[reasoning_filter]["rejected"] += len(rejected)

            bucket_counts = report.get("bucket_counts", {})
            bucket_detected.update(bucket_counts.get("detected_reasoning_type", {}))
            bucket_quality.update(bucket_counts.get("quality_band", {}))
            bucket_difficulty.update(bucket_counts.get("difficulty_band", {}))
            bucket_inferential.update(bucket_counts.get("inferential_validity_band", {}))

            per_shard[shard_key] = {
                "accepted": len(accepted),
                "rejected": len(rejected),
                "report": report,
            }

        per_batch[batch_key] = {
            "accepted": batch_accept,
            "rejected": batch_reject,
            "per_shard": per_shard,
            "source_dir": str(input_dir.resolve()),
        }

    write_jsonl(merged_output, accepted_rows)
    write_jsonl(reject_output, rejected_rows)

    summary = {
        "accepted_total": len(accepted_rows),
        "rejected_total": len(rejected_rows),
        "input_dirs": [str(path.resolve()) for path in input_dirs],
        "per_batch": per_batch,
        "by_reasoning_filter": dict(by_reasoning),
        "bucket_counts": {
            "detected_reasoning_type": dict(bucket_detected),
            "quality_band": dict(bucket_quality),
            "difficulty_band": dict(bucket_difficulty),
            "inferential_validity_band": dict(bucket_inferential),
        },
        "merged_output": str(merged_output.resolve()),
        "reject_output": str(reject_output.resolve()),
        "finished_at_utc": utc_now_iso(),
    }
    write_summary(summary_output, summary)

    print(f"Merged annotation accepted -> {merged_output}")
    print(f"Merged annotation rejects  -> {reject_output}")
    print(f"Annotation summary         -> {summary_output}")


def run_refresh_derived(args: argparse.Namespace) -> None:
    ensure_dirs()

    canonical_rows = load_jsonl(Path(args.canonical_input))
    canonical_by_pair = {
        (str(row.get("chunk_id", "")), str(row.get("reasoning_type", ""))): row for row in canonical_rows
    }

    payload_fields = (
        "domain",
        "title",
        "section",
        "context",
        "succinct_context",
        "reasoning_type",
        "reasoning_log",
        "ablation_test_log",
        "question",
        "answer",
        "is_valid",
        "error",
    )

    targets = [
        ("annotated", Path(args.annotated_path)),
        ("filtered", Path(args.filtered_path)),
        ("inferential_usable_only", Path(args.inferential_path)),
    ]
    report: Dict[str, Any] = {
        "canonical_input": str(Path(args.canonical_input).resolve()),
        "targets": {},
        "refreshed_at_utc": utc_now_iso(),
    }

    for label, path in targets:
        rows = load_jsonl(path)
        updated = []
        replaced = 0
        missing_pairs = 0
        for row in rows:
            pair = (str(row.get("chunk_id", "")), str(row.get("reasoning_type", "")))
            base = canonical_by_pair.get(pair)
            if not base:
                missing_pairs += 1
                updated.append(row)
                continue
            merged = dict(row)
            for field in payload_fields:
                if field in base:
                    merged[field] = base[field]
            updated.append(merged)
            replaced += 1

        write_jsonl(path, updated)
        report["targets"][label] = {
            "path": str(path.resolve()),
            "rows": len(rows),
            "replaced": replaced,
            "missing_pairs": missing_pairs,
        }

    write_summary(Path(args.report_output), report)
    print(f"Refreshed derived artifacts report -> {args.report_output}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Dataset utilities for QA generation.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    topup = subparsers.add_parser("select-topup", help="Select reproducible inferential top-up chunks.")
    topup.add_argument("--input", default=str(FILTERED_CHUNKS), help="Input chunk pool JSONL.")
    topup.add_argument(
        "--exclude",
        nargs="+",
        default=[str(SAMPLED_CHUNKS)],
        help="One or more existing chunk JSONL files to exclude by chunk_id.",
    )
    topup.add_argument(
        "--output",
        default=str(INTERIM_DIR / "chunks_topup_inferential.jsonl"),
        help="Output JSONL for selected top-up chunks.",
    )
    topup.add_argument("--manifest", default=str(CHUNKS_TOPUP_MANIFEST), help="Manifest JSON path.")
    topup.add_argument(
        "--purpose", default="inferential_topup_chunk_selection", help="Purpose string saved into the manifest."
    )
    topup.add_argument("--seed", type=int, default=42, help="Seed de on dinh tie-break.")
    topup.add_argument("--target-total", type=int, default=800, help="Tong so chunk can them.")
    topup.add_argument("--min-char-count", type=int, default=450, help="Nguong toi thieu cho do dai chunk.")
    topup.add_argument("--min-sentences", type=int, default=3, help="So cau toi thieu cho chunk inferential.")
    topup.set_defaults(handler=run_select_topup)

    merge_main = subparsers.add_parser("merge-main", help="Merge QA shard outputs.")
    merge_main.add_argument("--input-dir", default=str(QA_SHARDS_DIR), help="Directory containing shard outputs.")
    merge_main.add_argument("--merged-output", default=str(QA_RAW), help="Path for merged accepted samples.")
    merge_main.add_argument("--reject-output", default=str(QA_RAW_REJECTS), help="Path for merged rejected samples.")
    merge_main.add_argument("--summary-output", default=str(QA_FULL_RUN_SUMMARY), help="Path for merged summary JSON.")
    merge_main.set_defaults(handler=run_merge_main)

    merge_topup = subparsers.add_parser("merge-topup", help="Merge main QA batch with inferential top-up.")
    merge_topup.add_argument("--base-accepted", default=str(QA_RAW), help="Accepted JSONL from main batch.")
    merge_topup.add_argument("--base-rejects", default=str(QA_RAW_REJECTS), help="Rejected JSONL from main batch.")
    merge_topup.add_argument(
        "--topup-dirs",
        nargs="+",
        default=[str(QA_TOPUP_RUN_DIR), str(QA_TOPUP_RUN_DIR_ROUND2)],
        help="One or more directories containing inferential top-up shard outputs.",
    )
    merge_topup.add_argument("--merged-output", default=str(QA_WITH_TOPUP_ROUND2), help="Merged accepted output.")
    merge_topup.add_argument(
        "--reject-output", default=str(QA_WITH_TOPUP_ROUND2_REJECTS), help="Merged rejects output."
    )
    merge_topup.add_argument(
        "--summary-output", default=str(QA_WITH_TOPUP_ROUND2_SUMMARY), help="Merged summary output."
    )
    merge_topup.set_defaults(handler=run_merge_topup)

    merge_annotation = subparsers.add_parser(
        "merge-annotation",
        help="Merge external annotation shard outputs into one consolidated artifact.",
    )
    merge_annotation.add_argument(
        "--input-dirs",
        nargs="+",
        default=[
            str(QA_ANNOTATION_FLASH_EXTRACTION_DIR),
            str(QA_ANNOTATION_FLASH_MULTI_DIR),
        ],
        help="One or more directories containing annotation shard outputs.",
    )
    merge_annotation.add_argument(
        "--merged-output", default=str(QA_ANNOTATED_FLASH), help="Merged accepted annotation output."
    )
    merge_annotation.add_argument(
        "--reject-output", default=str(QA_ANNOTATED_FLASH_REJECTS), help="Merged rejected annotation output."
    )
    merge_annotation.add_argument(
        "--summary-output", default=str(QA_ANNOTATION_FULL_FLASH_SUMMARY), help="Merged annotation summary output."
    )
    merge_annotation.set_defaults(handler=run_merge_annotation)

    refresh = subparsers.add_parser(
        "refresh-derived", help="Refresh annotated/filtered downstream artifacts from the canonical QA dataset."
    )
    refresh.add_argument("--canonical-input", default=str(QA_CANONICAL), help="Canonical QA dataset after repairs.")
    refresh.add_argument(
        "--annotated-path",
        default=str(QA_CANONICAL_ANNOTATED),
        help="Annotated dataset to refresh in place.",
    )
    refresh.add_argument("--judged-path", dest="annotated_path", help=argparse.SUPPRESS)
    refresh.add_argument(
        "--filtered-path", default=str(QA_SPLIT_READY), help="Filtered-for-split dataset to refresh in place."
    )
    refresh.add_argument(
        "--inferential-path",
        default=str(QA_INFERENTIAL_USABLE_ONLY),
        help="Inferential usable subset to refresh in place.",
    )
    refresh.add_argument(
        "--report-output",
        default=str(QA_REPORTS_DIR / "qa_refresh_derived_report.json"),
        help="Refresh report output.",
    )
    refresh.set_defaults(handler=run_refresh_derived)

    return parser


# Backward-compatible alias for historical imports.
run_merge_judge = run_merge_annotation


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "merge-judge":
        sys.argv[1] = "merge-annotation"
    args = build_arg_parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
