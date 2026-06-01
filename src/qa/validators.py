"""Validation helpers and validators for QA generation."""

from __future__ import annotations

import re
from typing import Any, Dict, Tuple

SENTENCE_RE = re.compile(r"[.!?…]+(?:[\]\)\"']+)?(?=\s|$)")
DECIMAL_DOT_RE = re.compile(r"(?<=\d)\.(?=\d)")
PURE_NUMBER_RE = re.compile(r"^\d+(?:[.,]\d+)?$")
QUESTION_REQUEST_RE = re.compile(r"\b(?:nào|gì|ai)\b|bao nhiêu|khi nào")
ABSOLUTE_DATE_RE = re.compile(
    r"^(?:ngày\s+\d{1,2}\s+tháng\s+\d{1,2}\s+năm\s+\d{3,4}|\d{1,2}/\d{1,2}/\d{2,4})$",
    re.IGNORECASE,
)
TRAILING_UNIT_RE = re.compile(
    r"^(?:\s|/|\d|[.,])*(?:%|phần trăm|người|ca|lần|năm|tháng|ngày|giờ|phút|km|m|mét|ha)\b",
    re.IGNORECASE,
)
TERMINAL_PUNCT_RE = re.compile(r"[.!?…\"\)\]”']$")
HANGING_SUFFIX_RE = re.compile(
    r"(?:,\s*|:\s*|;\s*|\b(?:nơi|khi|sau khi|trong khi|nhưng|và|hoặc|để|rằng|vốn là|trong đó)\s*)$",
    re.IGNORECASE,
)

AMBIGUOUS_ANSWERS = {
    "anh",
    "bà",
    "cô",
    "họ",
    "nó",
    "ông",
    "người này",
    "nhân vật này",
    "tổ chức này",
    "chỗ ấy",
    "nơi ấy",
    "chỗ đó",
    "nơi đó",
    "đó",
    "đây",
}

DOCUMENT_REFERENCE_PHRASES = (
    "theo đoạn văn",
    "theo đoạn trích",
    "trong đoạn văn",
    "trong đoạn trích",
    "trong bài này",
    "theo bài viết",
    "theo văn bản",
    "theo ngữ cảnh",
    "trong văn bản",
    "trong nội dung",
    "thông tin trên",
)

DURATION_QUESTION_CUES = ("bao lâu", "khoảng thời gian", "mất bao lâu")

UNSUPPORTED_INFERENTIAL_RELATIONS = (
    "sử dụng",
    "dùng để",
    "tận dụng",
    "liên quan đến",
    "để thực hiện",
    "nhờ",
    "khiến",
)

QUESTION_STOPWORDS = {
    "ai",
    "bao",
    "các",
    "cho",
    "của",
    "gì",
    "khi",
    "khác",
    "là",
    "một",
    "nào",
    "này",
    "những",
    "người",
    "ở",
    "theo",
    "trong",
    "và",
    "việc",
    "với",
}


def count_sentences(text: str) -> int:
    without_numeric_dots = DECIMAL_DOT_RE.sub("", text or "")
    return len(SENTENCE_RE.findall(without_numeric_dots))


def normalize_span(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", (text or "").strip().lower())
    return re.sub(r"[\"'“”‘’.,;:!?()\[\]{}…-]+", "", cleaned)


def normalized_tokens(text: str) -> list[str]:
    return re.findall(r"\w+", normalize_span(text))


def content_tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"\w+", (text or "").lower())
        if len(token) > 1 and token not in QUESTION_STOPWORDS
    }


def is_contiguous_span(span: str, context: str) -> bool:
    if span in context:
        return True
    normalized_span = re.sub(r"\s+", " ", span).strip()
    normalized_context = re.sub(r"\s+", " ", context)
    return bool(normalized_span) and normalized_span in normalized_context


def is_complete_succinct_context(text: str) -> bool:
    succinct = (text or "").strip()
    if not succinct:
        return False
    if HANGING_SUFFIX_RE.search(succinct):
        return False
    if TERMINAL_PUNCT_RE.search(succinct):
        return True
    if count_sentences(succinct) >= 1 and not succinct.endswith(","):
        return True
    return False


def validate_succinct_context(
    qa: Dict[str, Any],
    context: str,
    context_above: str,
) -> Tuple[bool, str]:
    if not context_above:
        return True, ""

    succinct = (qa.get("succinct_context") or "").strip()
    if not succinct:
        return False, "missing_succinct_context"
    if len(succinct) > 500 or count_sentences(succinct) > 2:
        return False, "succinct_context_too_long"
    if not is_complete_succinct_context(succinct):
        return False, "succinct_context_incomplete"
    if (qa.get("answer") or "").strip() in succinct:
        return False, "answer_leaked_in_succinct_context"
    return True, ""


def validate_common(
    qa: Dict[str, Any],
    context: str,
    title: str = "",
    context_above: str = "",
) -> Tuple[bool, str]:
    answer = (qa.get("answer") or "").strip()
    question = (qa.get("question") or "").strip()

    if not answer or not question:
        return False, "missing_field"
    if answer not in context:
        if context_above and answer in context_above:
            return False, "answer_from_context_above"
        return False, "answer_not_in_context"
    if len(answer) > 250:
        return False, "answer_too_long"
    if len(question) < 15:
        return False, "question_too_short"

    question_lower = question.lower()
    if any(phrase in question_lower for phrase in DOCUMENT_REFERENCE_PHRASES):
        return False, "question_references_source_document"
    if answer.lower() in question_lower:
        return False, "question_leaks_answer"
    if normalize_span(answer) == normalize_span(title):
        return False, "answer_matches_title"
    if normalize_span(answer) in AMBIGUOUS_ANSWERS:
        return False, "ambiguous_answer"

    if PURE_NUMBER_RE.fullmatch(answer):
        start = context.find(answer)
        trailing = context[start + len(answer) : start + len(answer) + 24] if start >= 0 else ""
        if TRAILING_UNIT_RE.match(trailing):
            return False, "incomplete_numeric_answer"

    if any(cue in question_lower for cue in DURATION_QUESTION_CUES) and ABSOLUTE_DATE_RE.fullmatch(answer.lower()):
        return False, "answer_type_mismatch"

    return validate_succinct_context(qa, context, context_above)


def validate_extraction(
    qa: Dict[str, Any],
    context: str,
    title: str = "",
    context_above: str = "",
) -> Tuple[bool, str]:
    if not (qa.get("reasoning_log") or "").strip():
        return False, "missing_reasoning_log"

    is_valid, error = validate_common(qa, context, title, context_above)
    if not is_valid:
        return is_valid, error

    answer = (qa.get("answer") or "").strip()
    question = (qa.get("question") or "").strip().lower()

    if count_sentences(answer) > 1:
        return False, "answer_too_long"
    if len(answer) > 120 or answer.count(",") >= 4:
        return False, "answer_too_long"
    if " và " in question and (len(answer) > 80 or answer.count(",") >= 2):
        return False, "question_requests_multiple_answers"
    if re.search(r"\b(?:những|các)\b.+\bnào\b", question) and not any(
        marker in answer for marker in (",", ";", " và ")
    ):
        return False, "question_requests_multiple_answers"
    return True, ""


def question_uses_evidence(qa: Dict[str, Any], title: str = "") -> bool:
    evidence = (qa.get("evidence_span") or "").strip()
    question_tokens = content_tokens(qa.get("question") or "") - content_tokens(title)
    answer_tokens = content_tokens(qa.get("answer") or "")
    sentences = re.findall(r".*?[.!?…]+(?:[\]\)\"']+)?(?=\s|$)", evidence)
    answer_sentences = [sentence for sentence in sentences if (qa.get("answer") or "") in sentence]
    if not answer_sentences:
        return False
    answer_sentence_tokens = set().union(*(content_tokens(sentence) for sentence in answer_sentences))
    support_tokens = question_tokens - answer_tokens - answer_sentence_tokens
    return any(
        sentence not in answer_sentences and len(support_tokens & content_tokens(sentence)) >= 3
        for sentence in sentences
    )


def validate_multi_sentence(
    qa: Dict[str, Any],
    context: str,
    title: str = "",
    context_above: str = "",
) -> Tuple[bool, str]:
    if not (qa.get("ablation_test_log") or "").strip():
        return False, "missing_ablation_test_log"

    is_valid, error = validate_common(qa, context, title, context_above)
    if not is_valid:
        return is_valid, error

    evidence = (qa.get("evidence_span") or "").strip()
    if not evidence:
        return False, "missing_evidence_span"
    if not is_contiguous_span(evidence, context):
        return False, "evidence_not_in_context"
    if qa["answer"] not in evidence:
        return False, "answer_not_in_evidence"
    if count_sentences(evidence) < 2:
        return False, "multi_sentence_evidence_too_short"
    if len(QUESTION_REQUEST_RE.findall(qa["question"].lower())) > 1:
        return False, "question_requests_multiple_answers"

    evidence_lower = evidence.lower()
    question_lower = qa["question"].lower()
    if any(cue in question_lower and cue not in evidence_lower for cue in UNSUPPORTED_INFERENTIAL_RELATIONS):
        return False, "question_adds_unsupported_relation"
    if not question_uses_evidence(qa, title):
        return False, "multi_sentence_question_uses_one_sentence"
    return True, ""
