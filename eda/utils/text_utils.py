"""Small text-quality helpers used by EDA notebooks."""

from __future__ import annotations

import re


def has_html(text: str) -> bool:
    """Return True if the text still contains HTML-like tags."""

    return bool(re.search(r"<[^>]+>", text))


def has_wiki_template(text: str) -> bool:
    """Return True if the text still contains common wiki markup."""

    return "{{" in text or "[[" in text


def ends_with_punct(text: str) -> bool:
    """Return True if the text ends with sentence punctuation."""

    return text.strip().endswith((".", "!", "?", "…"))
