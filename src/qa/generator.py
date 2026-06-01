"""Orchestration layer for QA generation."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

from src.qa.prompts import (
    EXTRACTION_CONTEXTUAL_FEW_SHOT,
    EXTRACTION_CONTEXTUAL_USER_TEMPLATE,
    EXTRACTION_FEW_SHOT,
    EXTRACTION_SYSTEM_PROMPT,
    EXTRACTION_USER_TEMPLATE,
    MULTI_CONTEXTUAL_FEW_SHOT,
    MULTI_CONTEXTUAL_USER_TEMPLATE,
    MULTI_FEW_SHOT,
    MULTI_SYSTEM_PROMPT,
    MULTI_USER_TEMPLATE,
)
from src.qa.provider import DeepSeekJSONProvider
from src.qa.validators import validate_extraction, validate_multi_sentence

Validator = Callable[[Dict[str, Any], str, str, str], Tuple[bool, str]]
REQUIRED_CHUNK_FIELDS = ("chunk_id", "title", "domain", "text")


@dataclass(frozen=True)
class PromptConfig:
    system_prompt: str
    user_template: str
    contextual_user_template: str
    few_shot: str
    contextual_few_shot: str
    validator: Validator


PROMPT_CONFIGS: Dict[str, PromptConfig] = {
    "extraction": PromptConfig(
        system_prompt=EXTRACTION_SYSTEM_PROMPT,
        user_template=EXTRACTION_USER_TEMPLATE,
        contextual_user_template=EXTRACTION_CONTEXTUAL_USER_TEMPLATE,
        few_shot=EXTRACTION_FEW_SHOT,
        contextual_few_shot=EXTRACTION_CONTEXTUAL_FEW_SHOT,
        validator=validate_extraction,
    ),
    "multi-sentence": PromptConfig(
        system_prompt=MULTI_SYSTEM_PROMPT,
        user_template=MULTI_USER_TEMPLATE,
        contextual_user_template=MULTI_CONTEXTUAL_USER_TEMPLATE,
        few_shot=MULTI_FEW_SHOT,
        contextual_few_shot=MULTI_CONTEXTUAL_FEW_SHOT,
        validator=validate_multi_sentence,
    ),
}

RETRY_HINTS = {
    "answer_not_in_context": "Answer must be an exact span from Current Context.",
    "answer_from_context_above": "Do not take the answer from Context Above.",
    "answer_matches_title": "Choose a more specific answer span, not the page title.",
    "answer_too_long": "Choose one compact exact span, not a long copied list.",
    "question_leaks_answer": "Rewrite the question so it does not reveal the answer verbatim.",
    "question_references_source_document": "Question must stand alone, without phrases like 'the passage above'.",
    "missing_succinct_context": "Return a short succinct_context when Context Above is provided.",
    "succinct_context_too_long": "Keep succinct_context short, at most two sentences.",
    "succinct_context_incomplete": "succinct_context must be a complete sentence or two complete sentences, not a trailing clause or fragment.",
    "answer_leaked_in_succinct_context": "succinct_context must not contain the answer span.",
    "evidence_not_in_context": "evidence_span must be copied contiguously from Current Context.",
    "answer_not_in_evidence": "evidence_span must contain the answer verbatim.",
    "multi_sentence_evidence_too_short": "evidence_span must include at least two complete sentences.",
    "multi_sentence_question_uses_one_sentence": "Rewrite the question so it truly needs multiple sentences.",
    "question_requests_multiple_answers": "Ask for one target fact, not several answers at once.",
    "question_adds_unsupported_relation": "Do not invent a relation that is not explicit in the evidence.",
    "incomplete_numeric_answer": "If the answer is numeric, keep the attached unit or qualifier when needed.",
    "answer_type_mismatch": "Make the answer type match the question type.",
}


@dataclass
class QASample:
    chunk_id: str
    domain: str
    title: str
    section: str
    context: str
    succinct_context: str
    reasoning_type: str
    reasoning_log: str
    ablation_test_log: str
    question: str
    answer: str
    is_valid: bool
    error: str


class QAGenerator:
    """Generate QA samples from chunks with optional contextual prefixes."""

    def __init__(
        self,
        api_key: str,
        model_name: str = "deepseek-v4-flash",
        rpm_limit: int = 120,
        base_url: str = "https://api.deepseek.com",
        timeout: int = 120,
        max_validation_retries: int = 5,
    ) -> None:
        self.max_validation_retries = max(0, max_validation_retries)
        self.log = logging.getLogger(self.__class__.__name__)
        self.provider = DeepSeekJSONProvider(
            api_key=api_key,
            model_name=model_name,
            rpm_limit=rpm_limit,
            base_url=base_url,
            timeout=timeout,
        )

    def _generate_json(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        return self.provider.generate_json(system_prompt, user_prompt)

    @staticmethod
    def _validate_chunk_input(chunk: Dict[str, Any]) -> Tuple[bool, str]:
        for field in REQUIRED_CHUNK_FIELDS:
            value = chunk.get(field)
            if value is None or (isinstance(value, str) and not value.strip()):
                return False, f"missing_chunk_{field}"
        return True, ""

    @staticmethod
    def _get_context_above(chunk: Dict[str, Any]) -> str:
        return str(chunk.get("context_above") or "").strip()

    @staticmethod
    def _stored_context(chunk: Dict[str, Any], succinct_context: str) -> str:
        text = chunk["text"]
        return f"{succinct_context}\n\n{text}" if succinct_context else text

    def _build_sample(
        self,
        chunk: Dict[str, Any],
        reasoning_type: str,
        raw: Dict[str, Any],
        is_valid: bool,
        error: str,
    ) -> QASample:
        succinct_context = str(raw.get("succinct_context") or "").strip()
        return QASample(
            chunk_id=str(chunk.get("chunk_id", "")),
            domain=str(chunk.get("domain", "")),
            title=str(chunk.get("title", "")),
            section=str(chunk.get("section", "")),
            context=self._stored_context(chunk, succinct_context),
            succinct_context=succinct_context,
            reasoning_type=reasoning_type,
            reasoning_log=str(raw.get("reasoning_log") or "").strip(),
            ablation_test_log=str(raw.get("ablation_test_log") or "").strip(),
            question=str(raw.get("question") or "").strip(),
            answer=str(raw.get("answer") or "").strip(),
            is_valid=is_valid,
            error=error,
        )

    def _build_error_sample(self, chunk: Dict[str, Any], reasoning_type: str, error: str) -> QASample:
        return self._build_sample(chunk, reasoning_type, {}, False, error)

    def _retry_hint(self, error: str) -> str:
        return RETRY_HINTS.get(error, "Return a valid JSON object that follows the required schema.")

    def _build_user_prompt(
        self,
        *,
        config: PromptConfig,
        chunk: Dict[str, Any],
        context_above: Optional[str],
    ) -> str:
        fields = {"title": chunk.get("title", ""), "context": chunk["text"]}
        if context_above:
            return config.contextual_user_template.format(
                few_shot=config.contextual_few_shot,
                context_above=context_above,
                **fields,
            )
        return config.user_template.format(few_shot=config.few_shot, **fields)

    def _build_reasoning_config(
        self, chunk: Dict[str, Any], reasoning_type: str, context_above: Optional[str]
    ) -> Tuple[str, str, Validator]:
        if reasoning_type not in PROMPT_CONFIGS:
            raise ValueError(f"Unsupported reasoning_type: {reasoning_type}")
        config = PROMPT_CONFIGS[reasoning_type]
        return (
            config.system_prompt,
            self._build_user_prompt(config=config, chunk=chunk, context_above=context_above),
            config.validator,
        )

    def _generate_one(
        self,
        *,
        chunk: Dict[str, Any],
        reasoning_type: str,
        context_above: Optional[str],
        system_prompt: str,
        user_prompt: str,
        validator: Validator,
    ) -> QASample:
        prompt = user_prompt
        last_error = "generation_failed"
        rejected_output = ""

        for attempt in range(self.max_validation_retries + 1):
            try:
                raw = self._generate_json(system_prompt, prompt)
                if not isinstance(raw, dict):
                    last_error = "invalid_json_shape"
                elif raw.get("error") == "insufficient_context":
                    return self._build_error_sample(chunk, reasoning_type, "insufficient_context")
                else:
                    is_valid, error = validator(
                        raw,
                        chunk["text"],
                        str(chunk.get("title", "")),
                        context_above or "",
                    )
                    sample = self._build_sample(chunk, reasoning_type, raw, is_valid, error)
                    if is_valid or attempt == self.max_validation_retries:
                        return sample
                    last_error = error
                    rejected_output = json.dumps(raw, ensure_ascii=False)
            except json.JSONDecodeError:
                last_error = "json_parse_error"
            except Exception as exc:  # pragma: no cover
                self.log.error("%s: %s generation failed: %s", chunk.get("chunk_id", ""), reasoning_type, exc)
                return self._build_error_sample(chunk, reasoning_type, "request_failed")

            if attempt < self.max_validation_retries:
                prompt = (
                    f"{user_prompt}\n\nThe previous attempt was rejected with `{last_error}`. "
                    f"Rejected output: {rejected_output or '{no valid JSON}'}. "
                    f"{self._retry_hint(last_error)}"
                )

        return self._build_error_sample(chunk, reasoning_type, last_error)

    def _generate_reasoning_sample(
        self, chunk: Dict[str, Any], reasoning_type: str, context_above: Optional[str] = None
    ) -> QASample:
        is_valid_chunk, chunk_error = self._validate_chunk_input(chunk)
        if not is_valid_chunk:
            return self._build_error_sample(chunk, reasoning_type, chunk_error)

        system_prompt, user_prompt, validator = self._build_reasoning_config(
            chunk,
            reasoning_type,
            context_above,
        )
        return self._generate_one(
            chunk=chunk,
            reasoning_type=reasoning_type,
            context_above=context_above,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            validator=validator,
        )

    def generate_one_by_type(self, chunk: Dict[str, Any], reasoning_type: str) -> QASample:
        return self._generate_reasoning_sample(
            chunk,
            reasoning_type,
            context_above=self._get_context_above(chunk) or None,
        )

    def generate(self, chunk: Dict[str, Any]) -> list[QASample]:
        context_above = self._get_context_above(chunk) or None
        return [
            self._generate_reasoning_sample(chunk, reasoning_type, context_above=context_above)
            for reasoning_type in ("extraction", "multi-sentence")
        ]


if __name__ == "__main__":
    raise SystemExit("src.qa.generator is import-only. Instantiate src.qa.QAGenerator instead.")
