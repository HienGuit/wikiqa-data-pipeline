"""Build structural and knowledge feature matrices for the QA dataset."""

from __future__ import annotations

import math
import re
import statistics

import pandas as pd

from eda.utils.answer_typing import AnswerTypeResult, annotate_answer_types
from src.features.entity_utils import extract_entities, normalize_entity_name

try:
    from underthesea import word_tokenize
except Exception:  # pragma: no cover - optional dependency
    word_tokenize = None


QUESTION_TYPE_PATTERNS = [
    ("who", re.compile(r"^(ai|nguoi nao)\b")),
    ("what", re.compile(r"^(gi|cai gi|dieu gi|loai nao)\b")),
    ("when", re.compile(r"\b(khi nao|nam nao|thoi gian nao|bao gio)\b")),
    ("where", re.compile(r"\b(o dau|noi nao|dia diem nao)\b")),
    ("why", re.compile(r"\b(tai sao|vi sao|ly do gi)\b")),
    ("how", re.compile(r"\b(nhu the nao|bang cach nao|the nao)\b")),
    ("how_many", re.compile(r"\b(bao nhieu|so luong|may)\b")),
]
KNOWLEDGE_WEIGHTS = {
    "page_views": 0.30,
    "site_links": 0.20,
    "wiki_count": 0.20,
    "statements": 0.20,
    "references": 0.10,
}


def fold_text(text: str) -> str:
    return normalize_entity_name(text)


def tokenize_vi(text: str) -> list[str]:
    cleaned = " ".join(str(text or "").split())
    if not cleaned:
        return []
    if word_tokenize is None:
        return fold_text(cleaned).split()
    tokenized = word_tokenize(cleaned, format="text")
    return [token for token in fold_text(tokenized).split() if token]


def split_sentences(text: str) -> list[str]:
    cleaned = " ".join(str(text or "").split())
    if not cleaned:
        return []
    parts = re.split(r"(?<=[\.\?!…])\s+", cleaned)
    return [part.strip() for part in parts if part.strip()]


def safe_ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator) / float(denominator)


def question_type(question: str) -> str:
    folded = fold_text(question)
    for label, pattern in QUESTION_TYPE_PATTERNS:
        if pattern.search(folded):
            return label
    return "other"


def build_rank_maps(entity_records: list[dict]) -> dict[str, dict[str, float]]:
    rank_maps: dict[str, dict[str, float]] = {}
    for metric in KNOWLEDGE_WEIGHTS:
        values = [(record["entity_name_normalized"], record.get(metric)) for record in entity_records]
        valid = [(name, float(value)) for name, value in values if value is not None and not pd.isna(value)]
        if not valid:
            rank_maps[metric] = {}
            continue
        series = pd.Series({name: value for name, value in valid}).rank(method="average", pct=True)
        rank_maps[metric] = {name: float(rank) for name, rank in series.items()}
    return rank_maps


def _global_metric_defaults(rank_maps: dict[str, dict[str, float]]) -> dict[str, float]:
    defaults = {}
    for metric, mapping in rank_maps.items():
        defaults[metric] = float(statistics.mean(mapping.values())) if mapping else math.nan
    return defaults


def _resolve_entity_record(
    row: pd.Series,
    entity_db: dict,
) -> tuple[dict | None, str]:
    answer_entities = extract_entities(str(row.get("answer", "")), entity_db)
    if answer_entities:
        normalized = normalize_entity_name(answer_entities[0])
        return entity_db["by_normalized"].get(normalized), "answer_entity"

    question_entities = extract_entities(str(row.get("question", "")), entity_db)
    if question_entities:
        normalized = normalize_entity_name(question_entities[0])
        return entity_db["by_normalized"].get(normalized), "question_entity"

    section = str(row.get("section", "")).strip()
    if section:
        normalized = normalize_entity_name(section)
        if normalized in entity_db["by_normalized"]:
            return entity_db["by_normalized"][normalized], "wiki_section"

    title = str(row.get("title", "")).strip()
    if title:
        normalized = normalize_entity_name(title)
        if normalized in entity_db["by_normalized"]:
            return entity_db["by_normalized"][normalized], "wiki_title"

    return None, "global_mean"


def _compute_knowledge_features(
    row: pd.Series,
    entity_db: dict,
    rank_maps: dict[str, dict[str, float]],
    global_defaults: dict[str, float],
) -> dict:
    entity_record, source = _resolve_entity_record(row, entity_db)
    normalized_name = entity_record["entity_name_normalized"] if entity_record else None

    metric_ranks = {}
    available_weights = 0.0
    weighted_sum = 0.0

    for metric, weight in KNOWLEDGE_WEIGHTS.items():
        if normalized_name and normalized_name in rank_maps[metric]:
            rank_value = rank_maps[metric][normalized_name]
        else:
            rank_value = global_defaults.get(metric, math.nan)
        metric_ranks[f"{metric}_rank"] = rank_value
        if not pd.isna(rank_value):
            weighted_sum += weight * rank_value
            available_weights += weight

    popularity_score = weighted_sum / available_weights if available_weights > 0 else math.nan
    knowledge_difficulty = 1.0 - popularity_score if not pd.isna(popularity_score) else math.nan

    return {
        "popularity_source": source,
        "knowledge_difficulty": knowledge_difficulty,
        **metric_ranks,
    }


def _answer_type_lookup(answer_map: dict[str, AnswerTypeResult], answer: str) -> AnswerTypeResult:
    key = " ".join(str(answer or "").split())
    return answer_map.get(key, AnswerTypeResult(answer_type="Other", method="fallback"))


def build_feature_frame(dataset: pd.DataFrame, entity_db: dict) -> pd.DataFrame:
    answer_map, _answer_metadata = annotate_answer_types(dataset["answer"].tolist(), device=-1, batch_size=32)
    rank_maps = build_rank_maps(entity_db["records"])
    global_defaults = _global_metric_defaults(rank_maps)

    rows: list[dict] = []
    for idx, row in dataset.reset_index(drop=True).iterrows():
        context = str(row.get("context", "") or "")
        question = str(row.get("question", "") or "")
        answer = str(row.get("answer", "") or "")

        q_tokens = tokenize_vi(question)
        a_tokens = tokenize_vi(answer)
        ctx_tokens = tokenize_vi(context)
        ctx_sentences = split_sentences(context)

        answer_start = context.find(answer)
        answer_position = safe_ratio(answer_start if answer_start >= 0 else 0, len(context))

        sentence_idx = 0
        if answer and ctx_sentences:
            for sent_idx, sentence in enumerate(ctx_sentences):
                if answer in sentence:
                    sentence_idx = sent_idx
                    break
        answer_sentence_ratio = safe_ratio(sentence_idx, max(1, len(ctx_sentences)))

        q_set = set(q_tokens)
        ctx_set = set(ctx_tokens)
        answer_type_result = _answer_type_lookup(answer_map, answer)

        base = {
            "row_id": idx,
            "chunk_id": row.get("chunk_id"),
            "title": row.get("title"),
            "domain": row.get("domain"),
            "section": row.get("section"),
            "reasoning_type": row.get("reasoning_type"),
            "final_reasoning_bucket": row.get("final_reasoning_bucket"),
            "quality_band": row.get("quality_band"),
            "difficulty_band": row.get("difficulty_band"),
            "inferential_validity_band": row.get("inferential_validity_band"),
            "q_length": len(q_tokens),
            "a_length": len(a_tokens),
            "ctx_length": len(ctx_tokens),
            "ctx_sentence_count": len(ctx_sentences),
            "answer_position_ratio": answer_position,
            "answer_sentence_index_ratio": answer_sentence_ratio,
            "answer_density": safe_ratio(len(a_tokens), max(1, len(ctx_tokens))),
            "lexical_overlap_ratio": safe_ratio(len(q_set & ctx_set), max(1, len(q_set))),
            "ttr_question": safe_ratio(len(q_set), max(1, len(q_tokens))),
            "question_type": question_type(question),
            "answer_type": answer_type_result.answer_type,
        }
        base.update(_compute_knowledge_features(row, entity_db, rank_maps, global_defaults))
        rows.append(base)

    return pd.DataFrame(rows)
