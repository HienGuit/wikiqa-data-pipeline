"""Answer-type detection helpers reused by EDA and feature engineering."""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable

from transformers import pipeline

try:
    from underthesea import word_tokenize
except Exception:  # pragma: no cover - optional dependency
    word_tokenize = None


DEFAULT_NER_MODEL = "quocanh944/phoBERT-ner"
ANSWER_TYPE_ORDER = [
    "Date / Time",
    "Number / Quantity",
    "Proper Noun - Person",
    "Proper Noun - Location",
    "Proper Noun - Organization",
    "Common Noun Phrase",
    "Clause / Verb Phrase",
    "Other",
]

DATE_PATTERNS = [
    re.compile(r"\bngay\s+\d{1,2}(\s+thang\s+\d{1,2})?(\s+nam\s+\d{2,4})?\b", re.IGNORECASE),
    re.compile(r"\bthang\s+\d{1,2}(\s+nam\s+\d{2,4})?\b", re.IGNORECASE),
    re.compile(r"\bnam\s+\d{3,4}\b", re.IGNORECASE),
    re.compile(r"\bthe\s+ky\s+[xivlcdm0-9]+\b", re.IGNORECASE),
    re.compile(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b"),
]
QUANTITY_PATTERN = re.compile(
    r"\b\d+(?:[.,]\d+)?\s*(%|‰|km|m|cm|mm|kg|g|mg|ha|m2|km2|usd|vnd|dong|trieu|ty|nguoi|nam|thang|ngay|gio|phut|giay)\b",
    re.IGNORECASE,
)
ACRONYM_PATTERN = re.compile(r"^[A-ZĐ]{2,}(?:\s+[A-ZĐ]{2,})*$")
ORG_KEYWORDS = {
    "dang",
    "bo",
    "truong",
    "dai_hoc",
    "dai",
    "hoc",
    "hoc_vien",
    "uy_ban",
    "quoc_hoi",
    "lien_hop_quoc",
    "unesco",
    "cong_ty",
    "to_chuc",
    "hiep_hoi",
    "ngan_hang",
    "vien",
}
LOC_KEYWORDS = {
    "tinh",
    "thanh_pho",
    "thanh",
    "pho",
    "song",
    "nui",
    "bien",
    "vinh",
    "ho",
    "quan",
    "huyen",
    "xa",
    "dao",
    "chau",
    "luc",
}
VERB_CUES = {
    "la",
    "duoc",
    "bi",
    "gom",
    "gay",
    "giup",
    "muon",
    "nham",
    "de",
    "vi",
    "do",
    "khien",
    "thong_nhat",
    "thong",
}
CLAUSE_PREFIXES = ("vi ", "do ", "de ", "nham ", "khi ", "sau khi ", "truoc khi ")


@dataclass
class AnswerTypeResult:
    answer_type: str
    method: str


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def strip_accents(text: str) -> str:
    normalized = unicodedata.normalize("NFD", text)
    without_marks = "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")
    return without_marks.replace("đ", "d").replace("Đ", "D")


def fold_text(text: str) -> str:
    return strip_accents(normalize_space(text)).lower()


def looks_like_date_time(text: str) -> bool:
    folded = fold_text(text)
    return any(pattern.search(folded) for pattern in DATE_PATTERNS)


def looks_like_number_quantity(text: str) -> bool:
    return bool(QUANTITY_PATTERN.search(fold_text(text)))


def looks_like_clause(text: str) -> bool:
    lowered = fold_text(text)
    tokens = lowered.split()
    if lowered.startswith(CLAUSE_PREFIXES):
        return True
    return len(tokens) >= 4 and any(token in VERB_CUES for token in tokens)


def looks_like_common_noun_phrase(text: str) -> bool:
    lowered = fold_text(text)
    tokens = lowered.split()
    if not tokens or len(tokens) > 7:
        return False
    return not any(token in VERB_CUES for token in tokens)


def is_title_token(token: str) -> bool:
    letters = [ch for ch in token if ch.isalpha()]
    return bool(letters) and letters[0].isupper()


def titlecase_tokens(text: str) -> list[str]:
    return [token for token in normalize_space(text).split() if is_title_token(token)]


def heuristic_named_entity_type(text: str) -> AnswerTypeResult | None:
    cleaned = normalize_space(text)
    normalized = fold_text(cleaned).replace(" ", "_")
    title_tokens = titlecase_tokens(cleaned)

    if ACRONYM_PATTERN.match(cleaned) or any(keyword in normalized for keyword in ORG_KEYWORDS):
        return AnswerTypeResult("Proper Noun - Organization", "heuristic_proper_noun")
    if any(keyword in normalized for keyword in LOC_KEYWORDS):
        return AnswerTypeResult("Proper Noun - Location", "heuristic_proper_noun")
    if 1 <= len(title_tokens) <= 4 and len(title_tokens) == len(cleaned.split()):
        return AnswerTypeResult("Proper Noun - Person", "heuristic_proper_noun")
    return None


@lru_cache(maxsize=4)
def load_phobert_ner(model_name: str = DEFAULT_NER_MODEL, device: int = -1):
    return pipeline(
        "token-classification",
        model=model_name,
        tokenizer=model_name,
        aggregation_strategy="simple",
        framework="pt",
        device=device,
    )


def map_entity_group(label: str) -> str | None:
    normalized = label.upper()
    if normalized.endswith("PER") or "PERSON" in normalized:
        return "Proper Noun - Person"
    if normalized.endswith("LOC") or "LOCATION" in normalized:
        return "Proper Noun - Location"
    if normalized.endswith("ORG") or "ORGANIZATION" in normalized:
        return "Proper Noun - Organization"
    return None


def detect_named_entity_type_from_entities(text: str, entities: list[dict]) -> AnswerTypeResult | None:
    cleaned = normalize_space(text)
    if not cleaned or not entities:
        return None

    groups = []
    covered_chars = 0
    compact_text = re.sub(r"\s+", "", cleaned)
    for entity in entities:
        mapped = map_entity_group(str(entity.get("entity_group", "")))
        if not mapped:
            continue
        span = normalize_space(str(entity.get("word", "")))
        if not span:
            continue
        covered_chars += len(re.sub(r"\s+", "", span))
        groups.append(mapped)

    if not groups:
        return None

    dominant, count = Counter(groups).most_common(1)[0]
    coverage = covered_chars / max(1, len(compact_text))
    if dominant == "Proper Noun - Organization" and ACRONYM_PATTERN.match(cleaned):
        return AnswerTypeResult(dominant, "phobert_ner")
    if count == len(groups) and coverage >= 0.45:
        return AnswerTypeResult(dominant, "phobert_ner")
    return None


def detect_named_entity_type(text: str, ner_pipe) -> AnswerTypeResult | None:
    cleaned = normalize_space(text)
    ner_input = word_tokenize(cleaned, format="text") if word_tokenize is not None else cleaned
    entities = ner_pipe(ner_input)
    return detect_named_entity_type_from_entities(cleaned, entities)


def classify_answer(text: str, ner_pipe=None) -> AnswerTypeResult:
    cleaned = normalize_space(text)
    if not cleaned:
        return AnswerTypeResult("Other", "empty")
    if looks_like_date_time(cleaned):
        return AnswerTypeResult("Date / Time", "regex")
    if looks_like_number_quantity(cleaned):
        return AnswerTypeResult("Number / Quantity", "regex")
    if ACRONYM_PATTERN.match(cleaned):
        return AnswerTypeResult("Proper Noun - Organization", "heuristic_acronym")
    if ner_pipe is not None:
        ner_result = detect_named_entity_type(cleaned, ner_pipe)
        if ner_result is not None:
            return ner_result
    heuristic_entity = heuristic_named_entity_type(cleaned)
    if heuristic_entity is not None:
        return heuristic_entity
    if looks_like_clause(cleaned):
        return AnswerTypeResult("Clause / Verb Phrase", "heuristic")
    if looks_like_common_noun_phrase(cleaned):
        return AnswerTypeResult("Common Noun Phrase", "heuristic")
    return AnswerTypeResult("Other", "fallback")


def annotate_answer_types(
    answers: Iterable[str],
    *,
    model_name: str = DEFAULT_NER_MODEL,
    device: int = -1,
    batch_size: int = 32,
) -> tuple[dict[str, AnswerTypeResult], dict[str, str]]:
    unique_answers = sorted({normalize_space(answer) for answer in answers if normalize_space(answer)})
    regex_first: dict[str, AnswerTypeResult] = {}
    unresolved: list[str] = []

    for answer in unique_answers:
        result = classify_answer(answer, ner_pipe=None)
        if result.method == "regex":
            regex_first[answer] = result
        else:
            unresolved.append(answer)

    metadata = {
        "model_name": model_name,
        "ner_status": "not_used",
        "segmenter": "underthesea" if word_tokenize is not None else "none",
        "device": device,
        "batch_size": batch_size,
    }
    ner_pipe = None
    if unresolved:
        try:
            ner_pipe = load_phobert_ner(model_name, device)
            metadata["ner_status"] = "loaded"
        except Exception as exc:  # pragma: no cover
            metadata["ner_status"] = f"failed:{type(exc).__name__}"

    results = dict(regex_first)
    if ner_pipe is None:
        for answer in unresolved:
            results[answer] = classify_answer(answer, ner_pipe=None)
    else:
        ner_inputs = [
            word_tokenize(answer, format="text") if word_tokenize is not None else answer for answer in unresolved
        ]
        ner_outputs = ner_pipe(ner_inputs, batch_size=batch_size)
        for answer, entities in zip(unresolved, ner_outputs):
            ner_result = detect_named_entity_type_from_entities(answer, entities)
            if ner_result is not None:
                results[answer] = ner_result
            else:
                heuristic_entity = heuristic_named_entity_type(answer)
                if heuristic_entity is not None:
                    results[answer] = heuristic_entity
                elif looks_like_clause(answer):
                    results[answer] = AnswerTypeResult("Clause / Verb Phrase", "heuristic")
                elif looks_like_common_noun_phrase(answer):
                    results[answer] = AnswerTypeResult("Common Noun Phrase", "heuristic")
                else:
                    results[answer] = AnswerTypeResult("Other", "fallback")

    metadata["method_counts"] = dict(Counter(result.method for result in results.values()))
    return results, metadata
