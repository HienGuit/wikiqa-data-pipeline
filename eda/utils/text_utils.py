import re


def has_html(text: str) -> bool:
    """Kiểm tra text có còn thẻ HTML hay không."""
    return bool(re.search(r"<[^>]+>", text))


def has_wiki_template(text: str) -> bool:
    """Kiểm tra text có còn template hoặc wiki markup cơ bản hay không."""
    return "{{" in text or "[[" in text


def ends_with_punct(text: str) -> bool:
    """Kiểm tra text có kết thúc bằng dấu câu hay không."""
    return text.strip().endswith((".", "!", "?", "…"))
