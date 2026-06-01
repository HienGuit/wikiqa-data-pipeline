"""Public API for the QA generation subsystem."""

from src.qa.generator import QAGenerator, QASample
from src.qa.validators import count_sentences, validate_extraction, validate_multi_sentence

__all__ = [
    "QAGenerator",
    "QASample",
    "count_sentences",
    "validate_extraction",
    "validate_multi_sentence",
]
