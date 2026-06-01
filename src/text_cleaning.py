"""Shared text-cleaning helpers for article and QA context normalization."""

from __future__ import annotations

import html
import re
import unicodedata

CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
MULTI_BLANK_RE = re.compile(r"\n{3,}")
DOUBLE_QUOTES_RE = re.compile(r'"{2,}')
DISPLAYSTYLE_RE = re.compile(r"\{\\displaystyle.*?\}", re.DOTALL)
LATEX_COMMAND_RE = re.compile(
    r"\\(?:frac|left|right|cdot|sum|int|sqrt|alpha|beta|gamma|delta|times|pm|approx)\b(?:\{.*?\})*",
    re.IGNORECASE,
)
GENERIC_LATEX_TOKEN_RE = re.compile(r"\b\S*(?:\\|[_^{}])\S*\b")
MATH_ONLY_LINE_OPERATOR_RE = re.compile(r"[+\-*/=^]")
BULLET_LINE_RE = re.compile(r"^\s*(?:[-*•]|\d+[.)]|[A-Za-z][.)])\s+")
SENTENCE_END_RE = re.compile(r'[.?!…:;"”"]$')
TOKEN_MATH_CHARS = set("\\{}^_()[]=-+*/")


def normalize_unicode(text: str) -> str:
    return unicodedata.normalize("NFC", text or "")


def unescape_text(text: str) -> str:
    cleaned = html.unescape(text or "")
    replacements = {
        '\\"': '"',
        "\\'": "'",
        "\\n": "\n",
        "\\r": "\n",
        "\\t": " ",
    }
    for source, target in replacements.items():
        cleaned = cleaned.replace(source, target)
    return cleaned


def strip_control_chars(text: str) -> str:
    cleaned = (text or "").replace("\r\n", "\n").replace("\r", "\n").replace("\t", " ")
    return CONTROL_CHARS_RE.sub("", cleaned)


def merge_broken_lines(text: str) -> str:
    lines = text.split("\n")
    if not lines:
        return ""

    merged: list[str] = [lines[0].strip()]
    for line in lines[1:]:
        stripped = line.strip()
        if not stripped:
            merged.append("")
            continue

        previous = merged[-1] if merged else ""
        if (
            previous
            and previous.strip()
            and not SENTENCE_END_RE.search(previous.strip())
            and not BULLET_LINE_RE.match(stripped)
        ):
            merged[-1] = f"{previous.rstrip()} {stripped}"
        else:
            merged.append(stripped)

    cleaned = "\n".join(merged)
    cleaned = re.sub(r"\n[ \t]+\n", "\n\n", cleaned)
    return MULTI_BLANK_RE.sub("\n\n", cleaned).strip()


def _is_math_heavy_token(token: str) -> bool:
    if not token:
        return False
    if re.fullmatch(r"\(?\d+[.)]?\)?", token):
        return False
    math_chars = sum(1 for char in token if char in TOKEN_MATH_CHARS)
    return math_chars / max(len(token), 1) > 0.5


def remove_math_markup(text: str) -> str:
    cleaned = DISPLAYSTYLE_RE.sub(" ", text or "")
    cleaned = LATEX_COMMAND_RE.sub(" ", cleaned)
    cleaned = GENERIC_LATEX_TOKEN_RE.sub(
        lambda match: " " if _is_math_heavy_token(match.group(0)) else match.group(0),
        cleaned,
    )

    kept_lines: list[str] = []
    for raw_line in cleaned.splitlines():
        line = raw_line.strip()
        if not line:
            kept_lines.append("")
            continue

        operator_count = len(MATH_ONLY_LINE_OPERATOR_RE.findall(line))
        slash_count = line.count("\\")
        natural_token_count = len(re.findall(r"[A-Za-zÀ-ỹ]{2,}", line))
        if len(line) <= 120 and (operator_count >= 3 or slash_count >= 2) and natural_token_count <= 3:
            continue

        tokens = line.split()
        tokens = [token for token in tokens if not _is_math_heavy_token(token)]
        kept_lines.append(" ".join(tokens).strip())

    return "\n".join(kept_lines)


def normalize_quotes_and_whitespace(text: str) -> str:
    cleaned = DOUBLE_QUOTES_RE.sub('"', text or "")
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r" *\n *", "\n", cleaned)
    cleaned = MULTI_BLANK_RE.sub("\n\n", cleaned)
    return cleaned.strip()


def clean_article_text(text: str) -> str:
    cleaned = normalize_unicode(text)
    cleaned = unescape_text(cleaned)
    cleaned = strip_control_chars(cleaned)
    cleaned = remove_math_markup(cleaned)
    cleaned = merge_broken_lines(cleaned)
    cleaned = normalize_quotes_and_whitespace(cleaned)
    return cleaned


def clean_short_text(text: str) -> str:
    cleaned = normalize_unicode(text)
    cleaned = unescape_text(cleaned)
    cleaned = strip_control_chars(cleaned)
    cleaned = normalize_quotes_and_whitespace(cleaned)
    return cleaned
