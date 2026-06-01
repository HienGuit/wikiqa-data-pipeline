"""Batch runners for smoke, full generation, inferential top-up, and judge."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Set, Tuple

from src.config import (
    INTERIM_DIR,
    QA_JUDGE_RUN_DIR,
    QA_REPAIR_SUCCINCT_DIR,
    QA_SHARDS_DIR,
    QA_TOPUP_RUN_DIR,
    QA_WITH_TOPUP_ROUND2,
    SAMPLED_CHUNKS,
    TOPUP_CHUNKS_ROUND1,
    TOPUP_CHUNKS_ROUND2,
    ensure_dirs,
)
from src.qa.generator import QAGenerator
from src.qa.prompts import UNIFIED_JUDGE_FEW_SHOT, UNIFIED_JUDGE_SYSTEM_PROMPT, UNIFIED_JUDGE_USER_TEMPLATE
from src.qa.provider import DeepSeekJSONProvider, OpenRouterJSONProvider
from src.qa.validators import validate_succinct_context

REQUIRED_FIELDS = ("chunk_id", "title", "domain", "text")

DEFAULT_DEEPSEEK_MODEL = "deepseek-v4-flash"
DEFAULT_OPENROUTER_MODEL = "google/gemini-3-flash-preview"
DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_OPENROUTER_APP_TITLE = "wikiqa-data-pipeline"

JUDGE_BUCKET_FIELDS = {
    "detected_reasoning_type": {"literal", "inferential"},
    "quality_band": {"weak", "usable", "strong"},
    "difficulty_band": {"easy", "medium", "hard"},
    "inferential_validity_band": {"weak", "usable", "strong"},
}
JUDGE_SCORE_BUCKET_KEYS = (
    ("detected_reasoning_type", "detected_reasoning"),
    ("quality_band", "quality"),
    ("difficulty_band", "difficulty"),
    ("inferential_validity_band", "inferential_validity"),
)


def load_chunks(path: str | Path) -> List[Dict[str, Any]]:
    file_path = Path(path)
    with file_path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def append_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_report(path: Path, report: Dict[str, Any]) -> None:
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def write_samples(path: str | Path, samples: List[Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(asdict(sample), ensure_ascii=False) + "\n")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def configure_stdout() -> None:
    sys.stdout.reconfigure(encoding="utf-8")


def require_positive_int(value: int, flag_name: str, minimum: int = 1) -> None:
    if value < minimum:
        raise SystemExit(f"{flag_name} must be >= {minimum}.")


def ensure_api_key(api_key: str, env_name: str) -> str:
    if api_key:
        return api_key
    env_value = os.getenv(env_name, "")
    if not env_value:
        raise SystemExit(f"Missing API key. Pass --api-key or set {env_name}.")
    return env_value


def validate_chunk_schema(chunk: Dict[str, Any]) -> None:
    missing = [field for field in REQUIRED_FIELDS if not chunk.get(field)]
    if missing:
        raise SystemExit(f"Chunk missing required fields {missing}: {chunk!r}")


def shard_rows(rows: List[Dict[str, Any]], shard_index: int, shard_size: int) -> List[Dict[str, Any]]:
    start = shard_index * shard_size
    return rows[start : start + shard_size]


def load_processed_values(*paths: Path, extractor: Callable[[Dict[str, Any]], Any]) -> Set[Any]:
    processed: Set[Any] = set()
    for path in paths:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                value = extractor(row)
                if value in processed:
                    raise SystemExit(f"Duplicate processed value detected in existing outputs: {value}")
                processed.add(value)
    return processed


def sample_pair(row: Dict[str, Any]) -> Tuple[str, str]:
    return (str(row.get("chunk_id", "")), str(row.get("reasoning_type", "")))


def load_processed_pairs(*paths: Path) -> Set[Tuple[str, str]]:
    return load_processed_values(*paths, extractor=sample_pair)


def load_processed_chunk_ids(*paths: Path) -> Set[str]:
    return load_processed_values(*paths, extractor=lambda row: str(row.get("chunk_id", "")))


def load_processed_judge_pairs(*paths: Path) -> Set[Tuple[str, str]]:
    return load_processed_values(*paths, extractor=sample_pair)


def build_batch_paths(output_dir: Path, prefix: str, shard_index: int) -> Tuple[Path, Path, Path]:
    shard_name = f"{prefix}_{shard_index + 1:04d}"
    return (
        output_dir / f"{shard_name}.jsonl",
        output_dir / f"{shard_name}_rejects.jsonl",
        output_dir / f"{shard_name}_report.json",
    )


def flush_jsonl_buffers(
    valid_path: Path,
    reject_path: Path,
    valid_buffer: List[Dict[str, Any]],
    reject_buffer: List[Dict[str, Any]],
) -> None:
    if valid_buffer:
        append_jsonl(valid_path, valid_buffer)
        valid_buffer.clear()
    if reject_buffer:
        append_jsonl(reject_path, reject_buffer)
        reject_buffer.clear()


def choose_chunks(rows: List[Dict[str, Any]], n: int, seed: int) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    with_context = [row for row in rows if (row.get("context_above") or "").strip()]
    without_context = [row for row in rows if not (row.get("context_above") or "").strip()]

    chosen: List[Dict[str, Any]] = []
    if n >= 1 and without_context:
        chosen.append(rng.choice(without_context))
    if n >= 2 and with_context:
        contextual = rng.choice(with_context)
        if contextual.get("chunk_id") not in {row.get("chunk_id") for row in chosen}:
            chosen.append(contextual)

    if len(chosen) < n:
        remaining = [row for row in rows if row.get("chunk_id") not in {item.get("chunk_id") for item in chosen}]
        rng.shuffle(remaining)
        chosen.extend(remaining[: max(0, n - len(chosen))])

    return chosen[:n]


def candidate_order(rows: List[Dict[str, Any]], n: int, seed: int) -> List[Dict[str, Any]]:
    initial = choose_chunks(rows, n, seed)
    initial_ids = {row.get("chunk_id") for row in initial}
    remaining = [row for row in rows if row.get("chunk_id") not in initial_ids]
    random.Random(seed + 1).shuffle(remaining)
    return initial + remaining


def print_sample(chunk: Dict[str, Any], sample: Any) -> None:
    print("=" * 80)
    print(f"chunk_id       : {chunk.get('chunk_id')}")
    print(f"title          : {chunk.get('title')}")
    print(f"section        : {chunk.get('section')}")
    print(f"chunk_index    : {chunk.get('chunk_index')}")
    print(f"has_context    : {bool((chunk.get('context_above') or '').strip())}")
    print(f"text_len       : {len(chunk.get('text') or '')}")
    print(f"text_preview   : {(chunk.get('text') or '')[:240].replace(chr(10), ' ')}")
    if (chunk.get("context_above") or "").strip():
        print(f"context_len    : {len(chunk['context_above'])}")
        print(f"context_preview: {chunk['context_above'][:240].replace(chr(10), ' ')}")
    print("-" * 80)
    print(f"type           : {sample.reasoning_type}")
    print(f"is_valid       : {sample.is_valid}")
    print(f"error          : {sample.error}")
    if sample.reasoning_log:
        print(f"reasoning_log  : {sample.reasoning_log}")
    if sample.ablation_test_log:
        print(f"ablation_log   : {sample.ablation_test_log}")
    print(f"succinct_ctx   : {sample.succinct_context}")
    print(f"stored_context : {sample.context[:280].replace(chr(10), ' ')}")
    print(f"question       : {sample.question}")
    print(f"answer         : {sample.answer}")


def collect_valid_samples(
    generator: QAGenerator,
    candidates: List[Dict[str, Any]],
    reasoning_type: str,
    target: int,
    max_attempts: int,
) -> tuple[List[Any], Counter]:
    accepted: List[Any] = []
    rejected: Counter = Counter()
    for chunk in candidates[:max_attempts]:
        sample = generator.generate_one_by_type(chunk, reasoning_type)
        if sample.is_valid:
            accepted.append(sample)
            print_sample(chunk, sample)
            if len(accepted) == target:
                break
        else:
            rejected[sample.error] += 1
            print(f"Skipped {reasoning_type} candidate {chunk.get('chunk_id')}: {sample.error}")
    return accepted, rejected


def judge_sort_key(row: Dict[str, Any]) -> Tuple[int, str]:
    reasoning = str(row.get("reasoning_type", ""))
    chunk_id = str(row.get("chunk_id", ""))
    inferential_first = 0 if reasoning == "multi-sentence" else 1
    return (inferential_first, chunk_id)


def validate_judge_buckets(raw: Dict[str, Any]) -> Tuple[bool, str]:
    for field, allowed_values in JUDGE_BUCKET_FIELDS.items():
        value = raw.get(field)
        if not isinstance(value, str):
            return False, f"invalid_{field}_type"
        if value not in allowed_values:
            return False, f"invalid_{field}_value"
    return True, ""


def build_judge_prompt(row: Dict[str, Any]) -> Tuple[str, str]:
    return (
        UNIFIED_JUDGE_SYSTEM_PROMPT,
        UNIFIED_JUDGE_USER_TEMPLATE.format(
            reasoning_type=str(row.get("reasoning_type", "")),
            title=row.get("title", ""),
            succinct_context=row.get("succinct_context", ""),
            ablation_test_log=row.get("ablation_test_log", ""),
            context=row.get("context", ""),
            question=row.get("question", ""),
            answer=row.get("answer", ""),
        )
        + "\n\n"
        + UNIFIED_JUDGE_FEW_SHOT,
    )


def resolve_judge_api_key(args: argparse.Namespace) -> str:
    env_name = "OPENROUTER_API_KEY" if args.provider == "openrouter" else "DEEPSEEK_API_KEY"
    return ensure_api_key(args.api_key, env_name)


def resolve_judge_model(args: argparse.Namespace) -> str:
    if args.provider == "openrouter" and args.model == DEFAULT_DEEPSEEK_MODEL:
        return DEFAULT_OPENROUTER_MODEL
    return args.model


def build_judge_provider(args: argparse.Namespace):
    if args.provider == "openrouter":
        return OpenRouterJSONProvider(
            api_key=args.api_key,
            model_name=args.model,
            rpm_limit=args.rpm_limit,
            timeout=args.timeout,
            base_url=args.base_url,
            service_tier=args.service_tier,
            http_referer=args.http_referer,
            app_title=args.app_title,
        )
    return DeepSeekJSONProvider(
        api_key=args.api_key,
        model_name=args.model,
        rpm_limit=args.rpm_limit,
        timeout=args.timeout,
    )


def build_judge_bucket_counts(score_buckets: Dict[str, List[str]]) -> Dict[str, Dict[str, int]]:
    return {field_name: dict(Counter(score_buckets[bucket_key])) for field_name, bucket_key in JUDGE_SCORE_BUCKET_KEYS}


def load_existing_judge_state(valid_path: Path, reject_path: Path) -> Tuple[int, Counter, Dict[str, List[str]]]:
    accepted_rows = load_chunks(valid_path) if valid_path.exists() else []
    rejected_rows = load_chunks(reject_path) if reject_path.exists() else []
    rejected = Counter(str(row.get("error", "unknown")) for row in rejected_rows)
    score_buckets: Dict[str, List[str]] = {bucket_key: [] for _, bucket_key in JUDGE_SCORE_BUCKET_KEYS}
    for row in accepted_rows:
        for field_name, bucket_key in JUDGE_SCORE_BUCKET_KEYS:
            if row.get(field_name):
                score_buckets[bucket_key].append(str(row[field_name]))
    return len(accepted_rows), rejected, score_buckets


def build_generation_report(
    *,
    mode: str,
    model: str,
    rpm_limit: int,
    input_path: str | Path,
    valid_path: Path,
    reject_path: Path,
    shard_index: int,
    shard_size: int,
    chunk_count: int,
    accepted: int,
    rejected: Counter,
    started_at: str,
    max_validation_retries: int,
    reasoning_type: str | None = None,
) -> Dict[str, Any]:
    report = {
        "mode": mode,
        "shard_index": shard_index,
        "shard_size": shard_size,
        "chunk_count": chunk_count,
        "model": model,
        "rpm_limit": rpm_limit,
        "max_validation_retries": max_validation_retries,
        "input_path": str(Path(input_path).resolve()),
        "valid_path": str(valid_path.resolve()),
        "reject_path": str(reject_path.resolve()),
        "accepted_samples": accepted,
        "rejected_samples": sum(rejected.values()),
        "rejected_by_error": dict(rejected),
        "started_at_utc": started_at,
        "finished_at_utc": utc_now_iso(),
    }
    if reasoning_type:
        report["reasoning_type"] = reasoning_type
    return report


def build_chunk_lookup(source_paths: List[str | Path]) -> Dict[str, Dict[str, Any]]:
    chunk_by_id: Dict[str, Dict[str, Any]] = {}
    for source_path in source_paths:
        path = Path(source_path)
        if not path.exists():
            continue
        for chunk in load_chunks(path):
            validate_chunk_schema(chunk)
            chunk_id = str(chunk.get("chunk_id", ""))
            if chunk_id in chunk_by_id:
                continue
            chunk_by_id[chunk_id] = chunk
    return chunk_by_id


def collect_succinct_repair_targets(
    input_path: str | Path,
    chunk_sources: List[str | Path],
    reasoning_type: str,
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows = load_chunks(input_path)
    chunk_by_id = build_chunk_lookup(chunk_sources)
    targets: List[Dict[str, Any]] = []
    stats = Counter()
    by_reasoning = Counter()

    for row in rows:
        row_reasoning = str(row.get("reasoning_type", ""))
        if reasoning_type != "all" and row_reasoning != reasoning_type:
            continue
        if not (row.get("succinct_context") or "").strip():
            continue

        chunk_id = str(row.get("chunk_id", ""))
        chunk = chunk_by_id.get(chunk_id)
        if not chunk:
            stats["missing_chunk_source"] += 1
            continue

        ok, err = validate_succinct_context(
            row,
            str(chunk.get("text", "")),
            str(chunk.get("context_above", "") or ""),
        )
        if ok:
            continue

        stats[err] += 1
        by_reasoning[row_reasoning] += 1
        targets.append({"row": row, "chunk": chunk, "original_error": err})

    targets.sort(key=lambda item: (str(item["row"].get("reasoning_type", "")), str(item["row"].get("chunk_id", ""))))
    manifest = {
        "input_path": str(Path(input_path).resolve()),
        "chunk_sources": [str(Path(path).resolve()) for path in chunk_sources],
        "reasoning_filter": reasoning_type,
        "target_count": len(targets),
        "error_counts": dict(stats),
        "by_reasoning_type": dict(by_reasoning),
    }
    return targets, manifest


def run_generation_batch(
    args: argparse.Namespace,
    *,
    mode: str,
    output_prefix: str,
    progress_label: str,
    processed_mode: str,
    reasoning_type: str | None = None,
) -> None:
    configure_stdout()
    ensure_dirs()
    args.api_key = ensure_api_key(args.api_key, "DEEPSEEK_API_KEY")
    require_positive_int(args.shard_index, "--shard-index", minimum=0)
    require_positive_int(args.shard_size, "--shard-size")
    require_positive_int(args.flush_every, "--flush-every")

    started_at = utc_now_iso()
    rows = load_chunks(args.input)
    for row in rows:
        validate_chunk_schema(row)
    shard = shard_rows(rows, args.shard_index, args.shard_size)
    if not shard:
        raise SystemExit("Shard is empty. Check --shard-index and --shard-size.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    valid_path, reject_path, report_path = build_batch_paths(output_dir, output_prefix, args.shard_index)

    if processed_mode == "pair":
        processed_pairs = load_processed_pairs(valid_path, reject_path)
        processed_label = f"already processed pairs={len(processed_pairs)}"
    else:
        processed_pairs = None
        processed_chunk_ids = load_processed_chunk_ids(valid_path, reject_path)
        processed_label = f"already processed chunks={len(processed_chunk_ids)}"

    generator = QAGenerator(
        api_key=args.api_key,
        model_name=args.model,
        rpm_limit=args.rpm_limit,
        max_validation_retries=args.max_validation_retries,
    )

    print(f"Loaded {len(rows):,} chunks from {args.input}")
    print(f"Running {progress_label} shard {args.shard_index + 1} with {len(shard)} chunk(s); {processed_label}")
    print(f"Valid output : {valid_path}")
    print(f"Reject output: {reject_path}")
    print(f"Report file  : {report_path}")

    accepted = 0
    rejected = Counter()
    valid_buffer: List[Dict[str, Any]] = []
    reject_buffer: List[Dict[str, Any]] = []

    for chunk_offset, chunk in enumerate(shard, start=1):
        chunk_id = str(chunk.get("chunk_id", ""))

        if processed_mode == "pair":
            assert processed_pairs is not None
            samples = generator.generate(chunk)
            for sample in samples:
                pair = (chunk_id, sample.reasoning_type)
                if pair in processed_pairs:
                    continue
                processed_pairs.add(pair)
                if sample.is_valid:
                    accepted += 1
                    valid_buffer.append(asdict(sample))
                else:
                    rejected[sample.error] += 1
                    reject_buffer.append(
                        {
                            "chunk_id": sample.chunk_id,
                            "title": sample.title,
                            "reasoning_type": sample.reasoning_type,
                            "error": sample.error,
                        }
                    )
        else:
            if chunk_id in processed_chunk_ids:
                continue
            processed_chunk_ids.add(chunk_id)
            sample = generator.generate_one_by_type(chunk, reasoning_type or "multi-sentence")
            if sample.is_valid:
                accepted += 1
                valid_buffer.append(asdict(sample))
            else:
                rejected[sample.error] += 1
                reject_buffer.append(
                    {
                        "chunk_id": sample.chunk_id,
                        "title": sample.title,
                        "domain": sample.domain,
                        "error": sample.error,
                    }
                )

        if chunk_offset % args.flush_every == 0 or chunk_offset == len(shard):
            flush_jsonl_buffers(valid_path, reject_path, valid_buffer, reject_buffer)
            print(f"Progress {chunk_offset}/{len(shard)} chunks | accepted={accepted} | rejected={sum(rejected.values())}")

    report = build_generation_report(
        mode=mode,
        model=args.model,
        rpm_limit=args.rpm_limit,
        input_path=args.input,
        valid_path=valid_path,
        reject_path=reject_path,
        shard_index=args.shard_index,
        shard_size=args.shard_size,
        chunk_count=len(shard),
        accepted=accepted,
        rejected=rejected,
        started_at=started_at,
        max_validation_retries=args.max_validation_retries,
        reasoning_type=reasoning_type,
    )
    write_report(report_path, report)
    print("Done.")


def run_judge(args: argparse.Namespace) -> None:
    configure_stdout()
    ensure_dirs()
    args.api_key = resolve_judge_api_key(args)
    args.model = resolve_judge_model(args)
    require_positive_int(args.shard_index, "--shard-index", minimum=0)
    require_positive_int(args.shard_size, "--shard-size")
    require_positive_int(args.flush_every, "--flush-every")

    started_at = utc_now_iso()
    rows = load_chunks(args.input)
    if args.reasoning_type != "all":
        rows = [row for row in rows if str(row.get("reasoning_type", "")) == args.reasoning_type]
    rows = sorted(rows, key=judge_sort_key)
    shard = shard_rows(rows, args.shard_index, args.shard_size)
    if not shard:
        raise SystemExit("Judge shard is empty. Check filters, --shard-index, and --shard-size.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    valid_path, reject_path, report_path = build_batch_paths(output_dir, "qa_judge", args.shard_index)
    processed_pairs = load_processed_judge_pairs(valid_path, reject_path)
    provider = build_judge_provider(args)

    print(f"Loaded {len(rows):,} QA sample(s) from {args.input} after filter={args.reasoning_type}")
    print(f"Running judge shard {args.shard_index + 1} with {len(shard)} sample(s); already processed pairs={len(processed_pairs)}")
    print(f"Valid output : {valid_path}")
    print(f"Reject output: {reject_path}")
    print(f"Report file  : {report_path}")

    accepted, rejected, score_buckets = load_existing_judge_state(valid_path, reject_path)
    valid_buffer: List[Dict[str, Any]] = []
    reject_buffer: List[Dict[str, Any]] = []

    for idx, row in enumerate(shard, start=1):
        pair = sample_pair(row)
        if pair in processed_pairs:
            continue
        processed_pairs.add(pair)

        system_prompt, user_prompt = build_judge_prompt(row)
        try:
            raw = provider.generate_json(system_prompt, user_prompt)
            is_valid, error = validate_judge_buckets(raw)
            if not is_valid:
                rejected[error] += 1
                reject_buffer.append(
                    {
                        "chunk_id": pair[0],
                        "reasoning_type": pair[1],
                        "title": row.get("title", ""),
                        "error": error,
                        "raw_output": raw,
                    }
                )
            else:
                accepted += 1
                valid_buffer.append(
                    {
                        "chunk_id": pair[0],
                        "domain": row.get("domain", ""),
                        "title": row.get("title", ""),
                        "reasoning_type": pair[1],
                        "question": row.get("question", ""),
                        "answer": row.get("answer", ""),
                        "detected_reasoning_type": raw["detected_reasoning_type"],
                        "quality_band": raw["quality_band"],
                        "difficulty_band": raw["difficulty_band"],
                        "inferential_validity_band": raw["inferential_validity_band"],
                    }
                )
                score_buckets["detected_reasoning"].append(raw["detected_reasoning_type"])
                score_buckets["quality"].append(raw["quality_band"])
                score_buckets["difficulty"].append(raw["difficulty_band"])
                score_buckets["inferential_validity"].append(raw["inferential_validity_band"])
        except Exception as exc:
            rejected["request_failed"] += 1
            reject_buffer.append(
                {
                    "chunk_id": pair[0],
                    "reasoning_type": pair[1],
                    "title": row.get("title", ""),
                    "error": "request_failed",
                    "raw_output": str(exc),
                }
            )

        if idx % args.flush_every == 0 or idx == len(shard):
            flush_jsonl_buffers(valid_path, reject_path, valid_buffer, reject_buffer)
            print(f"Progress {idx}/{len(shard)} sample(s) | accepted={accepted} | rejected={sum(rejected.values())}")

    report = {
        "mode": "judge",
        "reasoning_filter": args.reasoning_type,
        "shard_index": args.shard_index,
        "shard_size": args.shard_size,
        "sample_count": len(shard),
        "provider": args.provider,
        "model": args.model,
        "service_tier": args.service_tier if args.provider == "openrouter" else None,
        "rpm_limit": args.rpm_limit,
        "timeout": args.timeout,
        "input_path": str(Path(args.input).resolve()),
        "valid_path": str(valid_path.resolve()),
        "reject_path": str(reject_path.resolve()),
        "accepted_samples": accepted,
        "rejected_samples": sum(rejected.values()),
        "rejected_by_error": dict(rejected),
        "bucket_counts": build_judge_bucket_counts(score_buckets),
        "started_at_utc": started_at,
        "finished_at_utc": utc_now_iso(),
    }
    write_report(report_path, report)
    print("Done.")


def run_smoke(args: argparse.Namespace) -> None:
    configure_stdout()
    args.api_key = ensure_api_key(args.api_key, "DEEPSEEK_API_KEY")
    require_positive_int(args.n, "--n")

    rows = load_chunks(args.input)
    candidates = candidate_order(rows, args.n, args.seed)
    max_candidates = args.max_candidates or args.n * 5

    generator = QAGenerator(
        api_key=args.api_key,
        model_name=args.model,
        rpm_limit=args.rpm_limit,
        max_validation_retries=args.max_validation_retries,
    )

    print(f"Loaded {len(rows):,} chunks from {args.input}")
    print(f"Collecting {args.n} valid sample(s) per type with model={args.model}; max_candidates={max_candidates}")

    extraction, extraction_rejected = collect_valid_samples(generator, candidates, "extraction", args.n, max_candidates)
    inferential, inferential_rejected = collect_valid_samples(generator, candidates, "multi-sentence", args.n, max_candidates)
    all_samples = extraction + inferential

    write_samples(args.output, all_samples)
    print("=" * 80)
    print(f"Accepted extraction    : {len(extraction)}/{args.n}")
    print(f"Accepted multi-sentence: {len(inferential)}/{args.n}")
    print(f"Rejected extraction    : {dict(extraction_rejected)}")
    print(f"Rejected multi-sentence: {dict(inferential_rejected)}")
    print(f"Saved {len(all_samples)} sample(s) to {args.output}")
    if len(extraction) < args.n or len(inferential) < args.n:
        raise SystemExit("Could not fill both QA quotas with valid candidates.")


def run_full(args: argparse.Namespace) -> None:
    run_generation_batch(
        args,
        mode="full",
        output_prefix="qa_batch_shard",
        progress_label="full",
        processed_mode="pair",
    )


def run_topup(args: argparse.Namespace) -> None:
    run_generation_batch(
        args,
        mode="inferential_topup",
        output_prefix="qa_topup_inferential",
        progress_label="inferential",
        processed_mode="chunk",
        reasoning_type="multi-sentence",
    )


def run_repair_succinct(args: argparse.Namespace) -> None:
    configure_stdout()
    ensure_dirs()
    args.api_key = ensure_api_key(args.api_key, "DEEPSEEK_API_KEY")
    require_positive_int(args.shard_index, "--shard-index", minimum=0)
    require_positive_int(args.shard_size, "--shard-size")
    require_positive_int(args.flush_every, "--flush-every")

    started_at = utc_now_iso()
    targets, target_manifest = collect_succinct_repair_targets(args.input, args.chunk_sources, args.reasoning_type)
    shard = shard_rows(targets, args.shard_index, args.shard_size)
    if not shard:
        raise SystemExit("Repair shard is empty. Check filters, --shard-index, and --shard-size.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "repair_target_manifest.json"
    if not manifest_path.exists():
        write_report(
            manifest_path,
            {
                **target_manifest,
                "created_at_utc": utc_now_iso(),
            },
        )

    valid_path, reject_path, report_path = build_batch_paths(output_dir, "qa_repair_succinct", args.shard_index)
    processed_pairs = load_processed_pairs(valid_path, reject_path)
    generator = QAGenerator(
        api_key=args.api_key,
        model_name=args.model,
        rpm_limit=args.rpm_limit,
        max_validation_retries=args.max_validation_retries,
    )

    print(f"Loaded {target_manifest['target_count']:,} repair target(s) from {args.input}")
    print(f"Running succinct repair shard {args.shard_index + 1} with {len(shard)} target(s); already processed pairs={len(processed_pairs)}")
    print(f"Valid output : {valid_path}")
    print(f"Reject output: {reject_path}")
    print(f"Report file  : {report_path}")

    accepted = 0
    rejected = Counter()
    original_errors = Counter()
    valid_buffer: List[Dict[str, Any]] = []
    reject_buffer: List[Dict[str, Any]] = []

    for idx, item in enumerate(shard, start=1):
        row = item["row"]
        chunk = item["chunk"]
        original_error = str(item["original_error"])
        pair = sample_pair(row)
        if pair in processed_pairs:
            continue
        processed_pairs.add(pair)
        original_errors[original_error] += 1

        sample = generator.generate_one_by_type(chunk, pair[1])
        if sample.is_valid:
            accepted += 1
            valid_buffer.append(asdict(sample))
        else:
            rejected[sample.error] += 1
            reject_buffer.append(
                {
                    "chunk_id": sample.chunk_id,
                    "title": sample.title,
                    "reasoning_type": sample.reasoning_type,
                    "error": sample.error,
                    "original_error": original_error,
                    "original_question": row.get("question", ""),
                    "original_answer": row.get("answer", ""),
                }
            )

        if idx % args.flush_every == 0 or idx == len(shard):
            flush_jsonl_buffers(valid_path, reject_path, valid_buffer, reject_buffer)
            print(f"Progress {idx}/{len(shard)} targets | accepted={accepted} | rejected={sum(rejected.values())}")

    report = {
        "mode": "repair_succinct",
        "shard_index": args.shard_index,
        "shard_size": args.shard_size,
        "target_count_total": target_manifest["target_count"],
        "target_count_shard": len(shard),
        "reasoning_filter": args.reasoning_type,
        "model": args.model,
        "rpm_limit": args.rpm_limit,
        "max_validation_retries": args.max_validation_retries,
        "input_path": str(Path(args.input).resolve()),
        "chunk_sources": [str(Path(path).resolve()) for path in args.chunk_sources],
        "valid_path": str(valid_path.resolve()),
        "reject_path": str(reject_path.resolve()),
        "accepted_samples": accepted,
        "rejected_samples": sum(rejected.values()),
        "rejected_by_error": dict(rejected),
        "original_error_counts_in_shard": dict(original_errors),
        "started_at_utc": started_at,
        "finished_at_utc": utc_now_iso(),
    }
    write_report(report_path, report)
    print("Done.")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="QA batch runners.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    smoke = subparsers.add_parser("smoke", help="Smoke-test QA generation.")
    smoke.add_argument("--input", default=str(SAMPLED_CHUNKS), help="Path to chunk JSONL file.")
    smoke.add_argument("--api-key", default=os.getenv("DEEPSEEK_API_KEY", ""), help="DeepSeek API key.")
    smoke.add_argument("--n", type=int, default=2, help="Target number of valid samples per reasoning type.")
    smoke.add_argument("--seed", type=int, default=42, help="Random seed for chunk selection.")
    smoke.add_argument("--model", default=DEFAULT_DEEPSEEK_MODEL, help="DeepSeek model name.")
    smoke.add_argument("--rpm-limit", type=int, default=120, help="Soft throttle in requests per minute for the local client.")
    smoke.add_argument("--max-candidates", type=int, default=None, help="Maximum candidate chunks to try per reasoning type. Defaults to n * 5.")
    smoke.add_argument("--max-validation-retries", type=int, default=1, help="Retries for malformed/invalid generations before replacing a candidate.")
    smoke.add_argument("--output", default="tests/chunk_for_tests/qa_smoke_test_output.jsonl", help="JSONL file to save generated QA samples.")
    smoke.set_defaults(handler=run_smoke)

    full = subparsers.add_parser("full", help="Run full shard generation.")
    full.add_argument("--input", default=str(SAMPLED_CHUNKS), help="Chunk JSONL input.")
    full.add_argument("--api-key", default=os.getenv("DEEPSEEK_API_KEY", ""), help="DeepSeek API key.")
    full.add_argument("--model", default=DEFAULT_DEEPSEEK_MODEL, help="Model name.")
    full.add_argument("--rpm-limit", type=int, default=120, help="Soft throttle for local client.")
    full.add_argument("--max-validation-retries", type=int, default=1, help="Retries per generation.")
    full.add_argument("--shard-index", type=int, default=0, help="0-based shard index.")
    full.add_argument("--shard-size", type=int, default=800, help="Chunks per shard.")
    full.add_argument("--output-dir", default=str(QA_SHARDS_DIR), help="Directory for shard outputs.")
    full.add_argument("--flush-every", type=int, default=20, help="Flush files every N chunks.")
    full.set_defaults(handler=run_full)

    topup = subparsers.add_parser("topup", help="Run inferential-only top-up generation.")
    topup.add_argument("--input", default=str(TOPUP_CHUNKS_ROUND1), help="Top-up chunk JSONL input.")
    topup.add_argument("--api-key", default=os.getenv("DEEPSEEK_API_KEY", ""), help="DeepSeek API key.")
    topup.add_argument("--model", default=DEFAULT_DEEPSEEK_MODEL, help="Model name.")
    topup.add_argument("--rpm-limit", type=int, default=120, help="Soft throttle for local client.")
    topup.add_argument("--max-validation-retries", type=int, default=1, help="Retries per generation.")
    topup.add_argument("--shard-index", type=int, default=0, help="0-based shard index.")
    topup.add_argument("--shard-size", type=int, default=100, help="Chunks per shard.")
    topup.add_argument("--output-dir", default=str(QA_TOPUP_RUN_DIR), help="Directory for inferential top-up outputs.")
    topup.add_argument("--flush-every", type=int, default=20, help="Flush files every N chunks.")
    topup.set_defaults(handler=run_topup)

    repair = subparsers.add_parser("repair-succinct", help="Repair rows whose succinct_context is incomplete.")
    repair.add_argument("--input", default=str(QA_WITH_TOPUP_ROUND2), help="Accepted QA JSONL input.")
    repair.add_argument(
        "--chunk-sources",
        nargs="+",
        default=[
            str(SAMPLED_CHUNKS),
            str(TOPUP_CHUNKS_ROUND1),
            str(TOPUP_CHUNKS_ROUND2),
        ],
        help="Chunk JSONL sources used to rebuild original chunk context.",
    )
    repair.add_argument("--api-key", default=os.getenv("DEEPSEEK_API_KEY", ""), help="DeepSeek API key.")
    repair.add_argument("--model", default=DEFAULT_DEEPSEEK_MODEL, help="Model name.")
    repair.add_argument("--rpm-limit", type=int, default=120, help="Soft throttle for local client.")
    repair.add_argument("--max-validation-retries", type=int, default=5, help="Retries per generation.")
    repair.add_argument("--reasoning-type", choices=["all", "extraction", "multi-sentence"], default="all", help="Repair all or only one reasoning type.")
    repair.add_argument("--shard-index", type=int, default=0, help="0-based shard index.")
    repair.add_argument("--shard-size", type=int, default=300, help="Targets per shard.")
    repair.add_argument("--output-dir", default=str(QA_REPAIR_SUCCINCT_DIR), help="Directory for repair outputs.")
    repair.add_argument("--flush-every", type=int, default=20, help="Flush files every N targets.")
    repair.set_defaults(handler=run_repair_succinct)

    judge = subparsers.add_parser("judge", help="Run LLM-as-judge on generated QA samples.")
    judge.add_argument("--input", default=str(QA_WITH_TOPUP_ROUND2), help="Accepted QA JSONL input.")
    judge.add_argument("--provider", choices=["deepseek", "openrouter"], default="deepseek", help="Judge API provider.")
    judge.add_argument("--api-key", default="", help="API key. Defaults to DEEPSEEK_API_KEY or OPENROUTER_API_KEY by provider.")
    judge.add_argument("--model", default=DEFAULT_DEEPSEEK_MODEL, help="Judge model name.")
    judge.add_argument("--base-url", default=DEFAULT_OPENROUTER_BASE_URL, help="OpenRouter-compatible base URL.")
    judge.add_argument("--service-tier", choices=["auto", "default", "flex", "priority", "scale"], default=None, help="OpenRouter service tier.")
    judge.add_argument("--http-referer", default="", help="Optional OpenRouter HTTP-Referer header.")
    judge.add_argument("--app-title", default=DEFAULT_OPENROUTER_APP_TITLE, help="Optional OpenRouter app title header.")
    judge.add_argument("--rpm-limit", type=int, default=120, help="Soft throttle for local client.")
    judge.add_argument("--timeout", type=int, default=120, help="HTTP timeout in seconds.")
    judge.add_argument("--reasoning-type", choices=["all", "extraction", "multi-sentence"], default="multi-sentence", help="Judge all QA or only one reasoning type.")
    judge.add_argument("--shard-index", type=int, default=0, help="0-based shard index.")
    judge.add_argument("--shard-size", type=int, default=400, help="Samples per shard.")
    judge.add_argument("--output-dir", default=str(QA_JUDGE_RUN_DIR), help="Directory for judge outputs.")
    judge.add_argument("--flush-every", type=int, default=20, help="Flush files every N samples.")
    judge.set_defaults(handler=run_judge)

    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
