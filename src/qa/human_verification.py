"""Shared helpers for human-verification bundle assembly and alignment."""

from __future__ import annotations

import json
from collections import Counter
from json import JSONDecoder
from pathlib import Path
from typing import Any, Dict, Iterable, List


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_jsonl_robust(path: Path) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    rows: List[Dict[str, Any]] = []
    repairs: List[Dict[str, Any]] = []
    decoder = JSONDecoder()

    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
                continue
            except json.JSONDecodeError:
                pass

            try:
                parsed, end = decoder.raw_decode(line)
                rows.append(parsed)
                repairs.append(
                    {
                        "line_number": line_number,
                        "repair_type": "trailing_garbage_trimmed",
                        "trailing_fragment": line[end:].strip(),
                    }
                )
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Could not parse {path} at line {line_number}: {exc}") from exc

    return rows, repairs


def load_task1_from_export(path: Path) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    rows = load_json(path)
    annotator1_rows: List[Dict[str, Any]] = []
    annotator2_rows: List[Dict[str, Any]] = []
    annotator_meta: List[Dict[str, Any]] = []

    for item in rows:
        data = dict(item.get("data", {}))
        annotations = sorted(
            list(item.get("annotations", [])),
            key=lambda ann: (
                str(ann.get("created_at", "")),
                int(ann.get("id", 0) or 0),
            ),
        )
        if len(annotations) < 2:
            raise RuntimeError(f"Task 1 export row {data.get('sample_id')} has fewer than 2 annotations.")

        if len(annotator_meta) < 2:
            annotator_meta = [{"slot": f"annotator{idx}"} for idx in (1, 2)]

        parsed_rows = []
        for annotation in annotations[:2]:
            parsed = {
                **data,
                "human_quality_band": "",
                "human_difficulty_band": "",
                "notes": "",
            }
            for result in annotation.get("result", []):
                from_name = result.get("from_name")
                choices = result.get("value", {}).get("choices", [])
                if not choices:
                    continue
                choice = str(choices[0]).strip().lower()
                if from_name == "quality_band":
                    parsed["human_quality_band"] = choice
                elif from_name == "difficulty_band":
                    parsed["human_difficulty_band"] = choice
            parsed_rows.append(parsed)

        annotator1_rows.append(parsed_rows[0])
        annotator2_rows.append(parsed_rows[1])

    return annotator1_rows, annotator2_rows, {"source": str(path), "annotator_slots": annotator_meta}


def sample_id_set(rows: Iterable[Dict[str, Any]]) -> set[str]:
    return {str(row.get("sample_id", "")) for row in rows}


def rows_by_sample_id(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {str(row.get("sample_id", "")): row for row in rows}


def annotation_distribution(rows: List[Dict[str, Any]], field: str) -> Dict[str, int]:
    return dict(sorted(Counter(str(row.get(field, "")) for row in rows).items()))


def validate_bundle_alignment(
    *,
    label: str,
    task_rows: List[Dict[str, Any]],
    gemini_key_rows: List[Dict[str, Any]],
    annotator1_rows: List[Dict[str, Any]],
    annotator2_rows: List[Dict[str, Any]],
) -> Dict[str, Any]:
    task_ids = sample_id_set(task_rows)
    gemini_ids = sample_id_set(gemini_key_rows)
    annotator1_ids = sample_id_set(annotator1_rows)
    annotator2_ids = sample_id_set(annotator2_rows)

    return {
        "task": label,
        "task_rows": len(task_rows),
        "gemini_key_rows": len(gemini_key_rows),
        "annotator1_rows": len(annotator1_rows),
        "annotator2_rows": len(annotator2_rows),
        "all_ids_match": task_ids == gemini_ids == annotator1_ids == annotator2_ids,
        "missing_in_gemini_key": sorted(task_ids - gemini_ids),
        "missing_in_annotator1": sorted(task_ids - annotator1_ids),
        "missing_in_annotator2": sorted(task_ids - annotator2_ids),
    }


def strip_task_payload_fields(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "sample_id": row.get("sample_id"),
        "task": row.get("task"),
        "bucket_group": row.get("bucket_group"),
        "chunk_id": row.get("chunk_id"),
        "title": row.get("title"),
        "domain": row.get("domain"),
        "section": row.get("section"),
        "reasoning_type": row.get("reasoning_type"),
        "context": row.get("context"),
        "question": row.get("question"),
        "answer": row.get("answer"),
    }


def strip_task1_annotation_fields(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "human_quality_band": row.get("human_quality_band", ""),
        "human_difficulty_band": row.get("human_difficulty_band", ""),
        "notes": row.get("notes", ""),
    }


def strip_task2_annotation_fields(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "human_inferential_validity_band": row.get("human_inferential_validity_band", ""),
        "notes": row.get("notes", ""),
    }


def strip_task1_key_fields(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "judge_model": row.get("judge_model", ""),
        "status": row.get("status", ""),
        "quality_band_ref": row.get("quality_band_ref", ""),
        "difficulty_band_ref": row.get("difficulty_band_ref", ""),
    }


def strip_task2_key_fields(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "judge_model": row.get("judge_model", ""),
        "status": row.get("status", ""),
        "inferential_validity_band_ref": row.get("inferential_validity_band_ref", ""),
    }


def build_combined_task1_rows(
    *,
    task_rows: List[Dict[str, Any]],
    gemini_key_rows: List[Dict[str, Any]],
    annotator1_rows: List[Dict[str, Any]],
    annotator2_rows: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    gemini_by_id = rows_by_sample_id(gemini_key_rows)
    annotator1_by_id = rows_by_sample_id(annotator1_rows)
    annotator2_by_id = rows_by_sample_id(annotator2_rows)

    return [
        {
            **strip_task_payload_fields(row),
            "gemini_key": strip_task1_key_fields(gemini_by_id[str(row.get("sample_id", ""))]),
            "annotator1": strip_task1_annotation_fields(annotator1_by_id[str(row.get("sample_id", ""))]),
            "annotator2": strip_task1_annotation_fields(annotator2_by_id[str(row.get("sample_id", ""))]),
        }
        for row in task_rows
    ]


def build_combined_task2_rows(
    *,
    task_rows: List[Dict[str, Any]],
    gemini_key_rows: List[Dict[str, Any]],
    annotator1_rows: List[Dict[str, Any]],
    annotator2_rows: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    gemini_by_id = rows_by_sample_id(gemini_key_rows)
    annotator1_by_id = rows_by_sample_id(annotator1_rows)
    annotator2_by_id = rows_by_sample_id(annotator2_rows)

    return [
        {
            **strip_task_payload_fields(row),
            "gemini_key": strip_task2_key_fields(gemini_by_id[str(row.get("sample_id", ""))]),
            "annotator1": strip_task2_annotation_fields(annotator1_by_id[str(row.get("sample_id", ""))]),
            "annotator2": strip_task2_annotation_fields(annotator2_by_id[str(row.get("sample_id", ""))]),
        }
        for row in task_rows
    ]
