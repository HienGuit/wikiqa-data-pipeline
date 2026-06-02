"""Semantic chunking for Vietnamese Wikipedia articles."""

from __future__ import annotations

import argparse
import json
import logging
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Generator, List

from src.config import RAW_CHUNKS, RAW_PAGES, ensure_dirs

from .text_cleaning import clean_article_text


@dataclass
class ChunkConfig:
    """Configuration for article chunking."""

    chunk_size: int = 1000
    chunk_overlap: int = 150
    min_chunk_size: int = 200
    max_chunk_size: int = 1500
    include_section_headers: bool = True
    skip_short_articles: bool = True
    min_article_chars: int = 500


@dataclass
class Chunk:
    """Chunk record persisted to JSONL."""

    chunk_id: str
    pageid: int
    title: str
    section: str
    text: str
    char_count: int
    chunk_index: int
    total_chunks: int
    position_pct: float
    url: str
    domain: str
    scope: str
    context_above: str = ""


SECTION_RE = re.compile(r"^(={2,4})\s*(.+?)\s*\1\s*$", re.MULTILINE)
PLAIN_SECTION_LINE_RE = re.compile(r"^[^\W\d_].{0,79}$", re.UNICODE)
SENTENCE_SPLIT_RE = re.compile(
    r'(?<=[.!?…])\s+(?=[A-ZÀ-Ỹ"“‘\'])',
    re.UNICODE,
)
MULTI_BLANK_RE = re.compile(r"\n{3,}")
SKIP_SECTION_PREFIX_RE = re.compile(
    r"^(Xem thêm|Đọc thêm|Liên kết ngoài|Tham khảo|Chú thích|Ghi chú)\s*[:\-–—]?\s*",
    flags=re.IGNORECASE,
)

SKIP_SECTION_NAMES = {
    "xem thêm",
    "đọc thêm",
    "liên kết ngoài",
    "tham khảo",
    "chú thích",
    "ghi chú",
}


def clean_text(text: str) -> str:
    """Normalize whitespace and remove obvious divider noise."""

    cleaned = MULTI_BLANK_RE.sub("\n\n", text or "")
    cleaned = cleaned.replace("\t", " ").replace("\r", "")
    cleaned = re.sub(r"^\s*[-=]{5,}\s*$", "", cleaned, flags=re.MULTILINE)
    return cleaned.strip()


def normalize_section_name(section: str) -> str:
    return re.sub(r"\s+", " ", (section or "").strip().lower())


def should_skip_section(section: str) -> bool:
    return normalize_section_name(section) in SKIP_SECTION_NAMES


def looks_like_plaintext_section_header(line: str) -> bool:
    candidate = (line or "").strip()
    if not candidate:
        return False
    if not PLAIN_SECTION_LINE_RE.match(candidate):
        return False
    if len(candidate.split()) > 8:
        return False
    if re.search(r"[.!?…:;]$", candidate):
        return False
    if re.search(r"^\W", candidate):
        return False
    return True


def clean_chunk_text(text: str) -> str:
    cleaned = (text or "").strip()
    cleaned = SKIP_SECTION_PREFIX_RE.sub("", cleaned)
    cleaned = MULTI_BLANK_RE.sub("\n\n", cleaned)
    return cleaned.strip()


def pre_clean_text(text: str) -> str:
    """Remove leftover wiki artifacts before chunking."""

    cleaned = clean_article_text(text)
    cleaned = re.sub(r"\{\|.*?\|\}", "", cleaned, flags=re.DOTALL)
    cleaned = re.sub(r"^\s*[|!].*$", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"\[\[(File|Image|Hình|Tập\s*tin):.*?\]\]", "", cleaned, flags=re.IGNORECASE | re.DOTALL)
    cleaned = re.sub(r"\{\{.*?\}\}", "", cleaned, flags=re.DOTALL)
    cleaned = re.sub(r"<[^>]+>", "", cleaned)
    cleaned = MULTI_BLANK_RE.sub("\n\n", cleaned)
    return cleaned.strip()


def split_into_paragraphs(text: str) -> List[str]:
    return [paragraph.strip() for paragraph in (text or "").split("\n\n") if paragraph.strip()]


def split_into_sentences(paragraph: str) -> List[str]:
    return [sentence.strip() for sentence in SENTENCE_SPLIT_RE.split(paragraph or "") if sentence.strip()]


def hard_split(text: str, max_size: int) -> List[str]:
    words = (text or "").split()
    parts: List[str] = []
    current: List[str] = []
    current_len = 0

    for word in words:
        current.append(word)
        current_len += len(word) + 1
        if current_len >= max_size:
            parts.append(" ".join(current))
            current = []
            current_len = 0

    if current:
        parts.append(" ".join(current))
    return parts


def fix_chunk_ending(text: str) -> str:
    """Trim incomplete trailing clause when no sentence-ending punctuation exists."""

    cleaned = (text or "").rstrip()
    if re.search(r"[.!?…]$", cleaned):
        return cleaned

    match = re.search(r"[.!?…](?=[^.!?…]*$)", cleaned)
    if match:
        return cleaned[: match.end()].rstrip()
    return cleaned


def is_noisy_chunk(text: str, max_special_ratio: float = 0.15) -> bool:
    normal = len(re.findall(r"[\w\sÀ-ỹ.,!?;:()\-\'\"]", text or ""))
    special = len(text or "") - normal
    return (special / max(len(text or ""), 1)) > max_special_ratio


class VietnameseWikiChunker:
    """Chunk Vietnamese Wikipedia text while preserving article structure."""

    def __init__(self, config: ChunkConfig | None = None) -> None:
        self.cfg = config or ChunkConfig()
        self.log = logging.getLogger("Chunker")

    def extract_sections(self, text: str) -> List[Dict[str, str]]:
        """Return article sections as ``{\"header\": str, \"body\": str}``."""

        cleaned_text = clean_text(text)
        matches = list(SECTION_RE.finditer(cleaned_text))
        sections: List[Dict[str, str]] = []

        if not matches:
            return self.extract_plaintext_sections(cleaned_text)

        intro = cleaned_text[: matches[0].start()].strip()
        if intro:
            sections.append({"header": "", "body": clean_text(intro)})

        for index, match in enumerate(matches):
            header = match.group(2).strip()
            start = match.end()
            end = matches[index + 1].start() if index + 1 < len(matches) else len(cleaned_text)
            body = clean_text(cleaned_text[start:end])
            if body:
                sections.append({"header": header, "body": body})

        return sections

    def extract_plaintext_sections(self, text: str) -> List[Dict[str, str]]:
        paragraphs = split_into_paragraphs(text)
        if not paragraphs:
            return []

        sections: List[Dict[str, str]] = []
        current_header = ""
        current_parts: List[str] = []

        def flush_section() -> None:
            body = clean_text("\n\n".join(current_parts))
            if body:
                sections.append({"header": current_header, "body": body})

        for index, paragraph in enumerate(paragraphs):
            lines = [line.strip() for line in paragraph.splitlines() if line.strip()]
            if not lines:
                continue

            first_line = lines[0]
            inline_header = looks_like_plaintext_section_header(first_line) and len(lines) > 1
            standalone_header = (
                looks_like_plaintext_section_header(first_line) and len(lines) == 1 and index + 1 < len(paragraphs)
            )

            if inline_header or standalone_header:
                flush_section()
                current_header = first_line
                current_parts = []
                if inline_header:
                    remainder = "\n".join(lines[1:]).strip()
                    if remainder:
                        current_parts.append(remainder)
                continue

            current_parts.append(paragraph)

        flush_section()
        return sections

    def pack_chunks(self, pieces: List[str], section_header: str) -> List[str]:
        """Pack paragraphs or sentences into chunks with overlap."""

        del section_header
        cfg = self.cfg
        raw_chunks: List[str] = []
        current_parts: List[str] = []
        current_len = 0

        for piece in pieces:
            piece_len = len(piece) + 1

            if len(piece) > cfg.max_chunk_size:
                if current_parts:
                    raw_chunks.append(fix_chunk_ending(" ".join(current_parts)))
                    current_parts = []
                    current_len = 0
                for sub_piece in hard_split(piece, cfg.chunk_size):
                    raw_chunks.append(fix_chunk_ending(sub_piece))
                continue

            if current_len + piece_len > cfg.chunk_size and current_parts:
                raw_chunks.append(fix_chunk_ending(" ".join(current_parts)))
                overlap_parts: List[str] = []
                overlap_len = 0
                for part in reversed(current_parts):
                    candidate_len = overlap_len + len(part) + 1
                    if candidate_len > cfg.chunk_overlap:
                        break
                    overlap_parts.insert(0, part)
                    overlap_len = candidate_len
                current_parts = overlap_parts
                current_len = overlap_len

            current_parts.append(piece)
            current_len += piece_len

        if current_parts:
            raw_chunks.append(fix_chunk_ending(" ".join(current_parts)))

        return raw_chunks

    def chunk_section(self, header: str, body: str) -> List[str]:
        """Prefer paragraph boundaries, then sentence boundaries, then hard split."""

        pieces: List[str] = []
        for paragraph in split_into_paragraphs(body):
            if len(paragraph) <= self.cfg.chunk_size:
                pieces.append(paragraph)
            else:
                sentences = split_into_sentences(paragraph)
                pieces.extend(sentences if sentences else [paragraph])
        return self.pack_chunks(pieces, header)

    def chunk_record(self, record: Dict[str, Any]) -> List[Chunk]:
        """Chunk a single page record into QA-ready article chunks."""

        cfg = self.cfg
        text = pre_clean_text(str(record.get("text", "")))
        pageid = int(record.get("pageid", 0) or 0)
        title = str(record.get("title", ""))
        url = str(record.get("url", ""))
        domain = str(record.get("domain", ""))
        scope = str(record.get("scope", ""))

        if cfg.skip_short_articles and len(text) < cfg.min_article_chars:
            self.log.debug("Skipping short article '%s' (%s chars)", title, len(text))
            return []

        raw_texts: List[tuple[str, str]] = []
        for section in self.extract_sections(text):
            if should_skip_section(section["header"]):
                self.log.debug("Skipping low-signal section '%s'", section["header"])
                continue
            for chunk_text in self.chunk_section(section["header"], section["body"]):
                raw_texts.append((section["header"], chunk_text))

        chunks: List[Chunk] = []
        total_raw = len(raw_texts)

        for index, (section_header, chunk_text) in enumerate(raw_texts):
            cleaned_text = clean_chunk_text(chunk_text)
            char_count = len(cleaned_text)
            if char_count < cfg.min_chunk_size:
                continue
            if is_noisy_chunk(cleaned_text):
                self.log.debug("Skipping noisy chunk: %s...", cleaned_text[:60])
                continue

            chunks.append(
                Chunk(
                    chunk_id=f"{pageid}_{index:04d}",
                    pageid=pageid,
                    title=title,
                    section=section_header,
                    text=cleaned_text,
                    char_count=char_count,
                    chunk_index=index,
                    total_chunks=total_raw,
                    position_pct=round(index / max(total_raw - 1, 1), 4),
                    url=url,
                    domain=domain,
                    scope=scope,
                )
            )

        real_total = len(chunks)
        for chunk in chunks:
            chunk.total_chunks = real_total

        intro_text = ""
        for candidate in chunks:
            if candidate.section == "":
                intro_text = candidate.text
                break

        for index, chunk in enumerate(chunks):
            if index == 0:
                chunk.context_above = ""
                continue
            context_parts: List[str] = []
            previous_text = chunks[index - 1].text
            if intro_text:
                context_parts.append(intro_text)
            if previous_text and previous_text != intro_text:
                context_parts.append(previous_text)
            chunk.context_above = "\n\n".join(context_parts).strip()

        return chunks


def iter_records(path: str | Path) -> Generator[Dict[str, Any], None, None]:
    """Yield JSON objects from a JSONL file."""

    file_path = Path(path)
    with file_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                yield json.loads(stripped)
            except json.JSONDecodeError as error:
                logging.getLogger("ChunkerIO").warning("Line %s has invalid JSON: %s", line_number, error)


def run_chunking(
    input_path: str | Path,
    output_path: str | Path,
    config: ChunkConfig | None = None,
    log_every: int = 500,
) -> Dict[str, int]:
    """Run chunking over the full JSONL article corpus."""

    source = Path(input_path)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    chunker = VietnameseWikiChunker(config)
    logger = logging.getLogger("ChunkerRunner")
    cfg = chunker.cfg

    logger.info(
        "Starting chunking from %s with chunk_size=%s overlap=%s min=%s max=%s",
        source,
        cfg.chunk_size,
        cfg.chunk_overlap,
        cfg.min_chunk_size,
        cfg.max_chunk_size,
    )

    temp_path = destination.with_suffix(destination.suffix + ".tmp")
    total_articles = 0
    total_chunks = 0
    skipped_articles = 0

    with temp_path.open("w", encoding="utf-8") as handle:
        for record in iter_records(source):
            total_articles += 1
            chunks = chunker.chunk_record(record)

            if not chunks:
                skipped_articles += 1
            else:
                total_chunks += len(chunks)
                for chunk in chunks:
                    handle.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")

            if total_articles % log_every == 0:
                logger.info(
                    "Processed %s articles | %s chunks | skipped %s",
                    total_articles,
                    total_chunks,
                    skipped_articles,
                )

    temp_path.replace(destination)

    logger.info("=" * 55)
    logger.info("Chunking complete")
    logger.info("Articles processed : %s", total_articles)
    logger.info("Articles skipped   : %s", skipped_articles)
    logger.info("Chunks written     : %s", total_chunks)
    if total_articles - skipped_articles > 0:
        average = total_chunks / (total_articles - skipped_articles)
        logger.info("Average chunks/article: %.1f", average)
    logger.info("Output             : %s", destination)
    logger.info("=" * 55)

    return {"articles": total_articles, "chunks": total_chunks, "skipped": skipped_articles}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Vietnamese Wikipedia semantic chunker",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", default=str(RAW_PAGES), help="Input article JSONL")
    parser.add_argument("--output", default=str(RAW_CHUNKS), help="Output chunk JSONL")
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--chunk-overlap", type=int, default=150)
    parser.add_argument("--min-chunk", type=int, default=200)
    parser.add_argument("--max-chunk", type=int, default=1500)
    parser.add_argument("--no-headers", action="store_true", help="Disable section-header prefixes in chunk assembly")
    parser.add_argument("--log-every", type=int, default=500)
    return parser


def main() -> None:
    ensure_dirs()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    args = build_arg_parser().parse_args()
    config = ChunkConfig(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        min_chunk_size=args.min_chunk,
        max_chunk_size=args.max_chunk,
        include_section_headers=not args.no_headers,
    )
    run_chunking(args.input, args.output, config=config, log_every=args.log_every)


if __name__ == "__main__":
    main()
