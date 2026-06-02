"""Entity lookup helpers for feature engineering."""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Iterable

try:
    from underthesea import ner
except Exception:  # pragma: no cover - optional dependency
    ner = None


def normalize_entity_name(name: str) -> str:
    """Normalize entity names for robust dictionary matching."""

    text = " ".join(str(name or "").strip().split())
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    text = text.replace("đ", "d").replace("Đ", "D").lower()
    text = re.sub(r"[^0-9a-z\s]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def load_entity_db(path: str | Path) -> dict:
    """Load entity_db JSONL and prepare lookup structures."""

    file_path = Path(path)
    records: list[dict] = []
    by_normalized: dict[str, dict] = {}
    with file_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            normalized = record.get("entity_name_normalized") or normalize_entity_name(record.get("entity_name", ""))
            record["entity_name_normalized"] = normalized
            records.append(record)
            if normalized and normalized not in by_normalized:
                by_normalized[normalized] = record

    search_items = sorted(by_normalized.items(), key=lambda item: len(item[0]), reverse=True)
    return {
        "records": records,
        "by_normalized": by_normalized,
        "search_items": search_items,
    }


def _merge_bio_tags(ner_result: list) -> list[str]:
    """Merge Underthesea BIO-style outputs into entity phrases."""

    merged: list[str] = []
    current: list[str] = []
    current_label = None

    for token, *_rest, label in ner_result:
        if label == "O":
            if current:
                merged.append(" ".join(current))
                current = []
                current_label = None
            continue

        if label.startswith("B-"):
            if current:
                merged.append(" ".join(current))
            current = [token]
            current_label = label[2:]
            continue

        if label.startswith("I-") and current and current_label == label[2:]:
            current.append(token)
            continue

        if current:
            merged.append(" ".join(current))
        current = [token]
        current_label = label[2:] if "-" in label else label

    if current:
        merged.append(" ".join(current))
    return merged


def _match_entities_from_dictionary(text: str, entity_db: dict) -> list[str]:
    normalized_text = normalize_entity_name(text)
    if not normalized_text:
        return []

    matches: list[str] = []
    occupied: list[tuple[int, int]] = []
    padded = f" {normalized_text} "
    for normalized_name, record in entity_db["search_items"]:
        needle = f" {normalized_name} "
        start = padded.find(needle)
        if start < 0:
            continue
        end = start + len(needle)
        if any(not (end <= taken_start or start >= taken_end) for taken_start, taken_end in occupied):
            continue
        occupied.append((start, end))
        matches.append(record["entity_name"])
    return matches


def extract_entities(text: str, entity_db: dict, use_ner: bool = True) -> list[str]:
    """Extract canonical entity names from text using dictionary + NER fallback."""

    dictionary_matches = _match_entities_from_dictionary(text, entity_db)
    if dictionary_matches or not use_ner or ner is None:
        return dictionary_matches

    try:
        ner_result = ner(text)
    except Exception:  # pragma: no cover - library/runtime variability
        return []

    merged = _merge_bio_tags(ner_result)
    matched: list[str] = []
    for phrase in merged:
        normalized = normalize_entity_name(phrase)
        if normalized in entity_db["by_normalized"]:
            matched.append(entity_db["by_normalized"][normalized]["entity_name"])
            continue
        phrase_matches = _match_entities_from_dictionary(phrase, entity_db)
        matched.extend(phrase_matches)

    # Deduplicate while preserving order.
    seen = set()
    deduped = []
    for item in matched:
        if item in seen:
            continue
        seen.add(item)
        deduped.append(item)
    return deduped


def iter_unique_titles(rows: Iterable[dict]) -> list[str]:
    titles = {str(row.get("title", "")).strip() for row in rows if str(row.get("title", "")).strip()}
    return sorted(titles)
