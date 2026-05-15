"""
chunker.py — Vietnamese Wikipedia Semantic Chunker
====================================================
Thiết kế dành riêng cho bộ dữ liệu Wikipedia tiếng Việt (JSONL),
phục vụ huấn luyện LLM và hệ thống RAG.

Chiến lược chunking theo thứ tự ưu tiên:
  1. Ưu tiên ranh giới đoạn văn (\\n\\n)
  2. Fallback: ranh giới câu (dấu . ! ? …)
  3. Fallback cuối: ranh giới từ

Khi chạy: python chunker.py --input wiki_pages_content.jsonl --output wiki_chunks.jsonl
Mỗi chunk được làm giàu metadata: domain, scope, url, tiêu đề,
vị trí tương đối, độ phủ toàn bài.
"""

import json
import re
import os
import logging
import argparse
from typing import List, Dict, Generator, Any
from dataclasses import dataclass, asdict

# ──────────────────────────────────────────────
# Cấu hình mặc định
# ──────────────────────────────────────────────
@dataclass
class ChunkConfig:
    # Kích thước chunk tính theo ký tự
    chunk_size: int = 1000          # Target size mỗi chunk
    chunk_overlap: int = 150        # Overlap để giữ ngữ cảnh liên tục
    min_chunk_size: int = 200       # Bỏ qua chunk quá ngắn
    max_chunk_size: int = 1500      # Giới hạn trên tránh chunk khổng lồ

    # Hành vi
    include_section_headers: bool = True   # Giữ tiêu đề section trong chunk
    skip_short_articles: bool = True       # Bỏ qua bài quá ngắn
    min_article_chars: int = 500


# ──────────────────────────────────────────────
# Các regex dùng chung — biên dịch sẵn
# ──────────────────────────────────────────────
# Tiêu đề section Wikipedia (dòng == … == hoặc === … ===)
_SECTION_RE   = re.compile(r'^(={2,4})\s*(.+?)\s*\1\s*$', re.MULTILINE)

# Câu kết thúc: . ! ? … theo sau bởi dấu cách hoặc xuống dòng hoặc EOF
_SENTENCE_END = re.compile(r'(?<=[.!?…])\s+(?=[A-ZÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚÝĂĐƠƯẠẢẤẦẨẪẬẮẰẲẴẶẸẺẼẾỀỂỄỆỈỊỌỎỐỒỔỖỘỚỜỞỠỢỤỦỨỪỬỮỰỲỴỶỸ\"\'])', re.UNICODE)

# Nhiều dòng trắng → chuẩn hoá
_MULTI_BLANK  = re.compile(r'\n{3,}')

# Loại bỏ dòng chỉ có dấu = (header markers sót lại)
_HEADER_LINE  = re.compile(r'^={2,}.*={2,}$', re.MULTILINE)


# ──────────────────────────────────────────────
# Data model cho một Chunk
# ──────────────────────────────────────────────
@dataclass
class Chunk:
    chunk_id:     str    # "{pageid}_{idx:04d}"
    pageid:       int
    title:        str
    section:      str    # Tiêu đề section hiện tại (rỗng nếu intro)
    text:         str    # Nội dung chunk
    char_count:   int
    chunk_index:  int    # Vị trí chunk trong bài (0-based)
    total_chunks: int    # Tổng số chunk của bài (cập nhật sau)
    position_pct: float  # Vị trí tương đối 0.0–1.0
    url:          str
    domain:       str
    scope:        str


# ──────────────────────────────────────────────
# Tiện ích xử lý văn bản
# ──────────────────────────────────────────────

def _clean_text(text: str) -> str:
    """Chuẩn hoá khoảng trắng và loại ký tự rác."""
    text = _MULTI_BLANK.sub('\n\n', text)
    text = text.replace('\t', ' ').replace('\r', '')
    # Loại bỏ dòng chỉ toàn dấu gạch ngang (divider)
    text = re.sub(r'^\s*[-=]{5,}\s*$', '', text, flags=re.MULTILINE)
    return text.strip()


def _split_into_paragraphs(text: str) -> List[str]:
    """Tách văn bản theo đoạn (\\n\\n). Lọc đoạn trống."""
    return [p.strip() for p in text.split('\n\n') if p.strip()]


def _split_into_sentences(paragraph: str) -> List[str]:
    """Tách đoạn văn thành câu dựa trên regex tiếng Việt."""
    parts = _SENTENCE_END.split(paragraph)
    return [s.strip() for s in parts if s.strip()]


def _hard_split(text: str, max_size: int) -> List[str]:
    """Cắt cứng theo từ khi không có ranh giới tự nhiên."""
    words = text.split()
    chunks, current = [], []
    length = 0
    for word in words:
        current.append(word)
        length += len(word) + 1
        if length >= max_size:
            chunks.append(' '.join(current))
            current, length = [], 0
    if current:
        chunks.append(' '.join(current))
    return chunks


# ──────────────────────────────────────────────
# Chunker chính
# ──────────────────────────────────────────────

class VietnameseWikiChunker:
    """
    Chunker ngữ nghĩa cho văn bản Wikipedia tiếng Việt.

    Thuật toán:
        - Phân tích cấu trúc section của bài viết
        - Tách từng section thành đoạn → câu
        - Gom câu / đoạn vào chunk không vượt chunk_size
        - Thêm overlap bằng cách tái sử dụng phần cuối chunk trước
        - Gắn metadata đầy đủ cho mỗi chunk
    """

    def __init__(self, config: ChunkConfig = None):
        self.cfg = config or ChunkConfig()
        self.log = logging.getLogger("Chunker")

    # ── Bước 1: Phân tích section ──────────────────

    def _extract_sections(self, text: str) -> List[Dict[str, str]]:
        """
        Trả về list of {"header": str, "body": str}.
        Phần đầu bài (trước section đầu tiên) có header = "".
        """
        sections = []
        matches = list(_SECTION_RE.finditer(text))

        if not matches:
            return [{"header": "", "body": _clean_text(text)}]

        # Phần intro (trước section đầu)
        intro = text[:matches[0].start()].strip()
        if intro:
            sections.append({"header": "", "body": _clean_text(intro)})

        for i, m in enumerate(matches):
            header = m.group(2).strip()
            start  = m.end()
            end    = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            body   = _clean_text(text[start:end])
            if body:
                sections.append({"header": header, "body": body})

        return sections

    # ── Bước 2: Gom nội dung thành chunk ──────────

    def _pack_chunks(self, pieces: List[str], section_header: str) -> List[str]:
        """
        Gom các mảnh văn bản (đoạn hoặc câu) thành chunk kích thước phù hợp.
        Áp dụng overlap giữa các chunk liên tiếp.
        """
        cfg = self.cfg
        prefix = f"[{section_header}] " if section_header and cfg.include_section_headers else ""

        raw_chunks: List[str] = []
        current_parts: List[str] = []
        current_len = len(prefix)

        for piece in pieces:
            piece_len = len(piece) + 1  # +1 for separator

            # Nếu một mảnh đơn lẻ đã vượt max → hard-split nó
            if len(piece) > cfg.max_chunk_size:
                if current_parts:
                    raw_chunks.append(prefix + ' '.join(current_parts))
                    current_parts, current_len = [], len(prefix)
                for sub in _hard_split(piece, cfg.chunk_size):
                    raw_chunks.append(prefix + sub)
                continue

            if current_len + piece_len > cfg.chunk_size and current_parts:
                raw_chunks.append(prefix + ' '.join(current_parts))
                # Overlap: giữ lại một số mảnh cuối
                overlap_parts, overlap_len = [], 0
                for part in reversed(current_parts):
                    if overlap_len + len(part) + 1 > cfg.chunk_overlap:
                        break
                    overlap_parts.insert(0, part)
                    overlap_len += len(part) + 1
                current_parts = overlap_parts
                current_len   = len(prefix) + overlap_len

            current_parts.append(piece)
            current_len += piece_len

        if current_parts:
            raw_chunks.append(prefix + ' '.join(current_parts))

        return raw_chunks

    # ── Bước 3: Tạo chunk từ một section ──────────

    def _chunk_section(self, header: str, body: str) -> List[str]:
        """Chiến lược: đoạn → câu → hard-split theo thứ tự ưu tiên."""
        paragraphs = _split_into_paragraphs(body)
        pieces: List[str] = []

        for para in paragraphs:
            if len(para) <= self.cfg.chunk_size:
                pieces.append(para)
            else:
                # Đoạn dài → tách câu
                sentences = _split_into_sentences(para)
                pieces.extend(sentences if sentences else [para])

        return self._pack_chunks(pieces, header)

    # ── Entry point: xử lý một record JSONL ────────

    def chunk_record(self, record: Dict[str, Any]) -> List[Chunk]:
        """
        Nhận một record từ wiki_pages_content.jsonl, trả về list[Chunk].
        """
        cfg      = self.cfg
        text     = record.get("text", "")
        pageid   = record.get("pageid", 0)
        title    = record.get("title", "")
        url      = record.get("url", "")
        domain   = record.get("domain", "")
        scope    = record.get("scope", "")

        if cfg.skip_short_articles and len(text) < cfg.min_article_chars:
            self.log.debug(f"Bỏ qua bài quá ngắn: '{title}' ({len(text)} chars)")
            return []

        sections = self._extract_sections(text)
        raw_texts: List[tuple] = []  # (section_header, chunk_text)

        for sec in sections:
            for chunk_text in self._chunk_section(sec["header"], sec["body"]):
                raw_texts.append((sec["header"], chunk_text))

        chunks: List[Chunk] = []
        total  = len(raw_texts)

        for idx, (sec_header, chunk_text) in enumerate(raw_texts):
            char_count = len(chunk_text)
            if char_count < cfg.min_chunk_size:
                continue

            chunks.append(Chunk(
                chunk_id     = f"{pageid}_{idx:04d}",
                pageid       = pageid,
                title        = title,
                section      = sec_header,
                text         = chunk_text,
                char_count   = char_count,
                chunk_index  = idx,
                total_chunks = total,
                position_pct = round(idx / max(total - 1, 1), 4),
                url          = url,
                domain       = domain,
                scope        = scope,
            ))

        # Cập nhật total_chunks thực tế (sau khi lọc min_chunk_size)
        real_total = len(chunks)
        for c in chunks:
            c.total_chunks = real_total

        return chunks


# ──────────────────────────────────────────────
# I/O: Đọc JSONL → Chunk → Ghi JSONL
# ──────────────────────────────────────────────

def iter_records(path: str) -> Generator[Dict, None, None]:
    """Đọc từng dòng JSONL, yield dict."""
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                logging.getLogger("IO").warning(f"Dòng {lineno}: JSON lỗi — {e}")


def run_chunking(
    input_path:  str,
    output_path: str,
    config:      ChunkConfig = None,
    log_every:   int = 500,
):
    """
    Pipeline chính: đọc JSONL bài viết → chunk → ghi JSONL chunk.

    Args:
        input_path:  Đường dẫn file wiki_pages_content.jsonl
        output_path: Đường dẫn file output (wiki_chunks.jsonl)
        config:      ChunkConfig tùy chỉnh; dùng mặc định nếu None
        log_every:   Tần suất ghi log tiến độ
    """
    log     = logging.getLogger("Runner")
    chunker = VietnameseWikiChunker(config)
    cfg     = chunker.cfg

    log.info(f"Bắt đầu chunking: {input_path}")
    log.info(f"Config: chunk_size={cfg.chunk_size}, overlap={cfg.chunk_overlap}, "
             f"min={cfg.min_chunk_size}, max={cfg.max_chunk_size}")

    tmp_path       = output_path + ".tmp"
    total_articles = 0
    total_chunks   = 0
    skipped        = 0

    with open(tmp_path, "w", encoding="utf-8") as out_f:
        for record in iter_records(input_path):
            total_articles += 1
            chunks = chunker.chunk_record(record)

            if not chunks:
                skipped += 1
            else:
                total_chunks += len(chunks)
                for chunk in chunks:
                    out_f.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")

            if total_articles % log_every == 0:
                log.info(
                    f"  Đã xử lý {total_articles} bài | "
                    f"{total_chunks} chunks | bỏ qua {skipped}"
                )

    # Atomic replace
    os.replace(tmp_path, output_path)

    log.info("=" * 55)
    log.info(f"Hoàn tất chunking!")
    log.info(f"  Tổng bài viết xử lý : {total_articles}")
    log.info(f"  Bài bị bỏ qua        : {skipped}")
    log.info(f"  Tổng chunks tạo ra   : {total_chunks}")
    if total_articles - skipped > 0:
        avg = total_chunks / (total_articles - skipped)
        log.info(f"  Trung bình chunk/bài : {avg:.1f}")
    log.info(f"  Output               : {output_path}")
    log.info("=" * 55)

    return {"articles": total_articles, "chunks": total_chunks, "skipped": skipped}


# ──────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Vietnamese Wikipedia Semantic Chunker",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input",   default="data/wiki_pages_content.jsonl",
                   help="Đường dẫn file JSONL đầu vào")
    p.add_argument("--output",  default="data/wiki_chunks.jsonl",
                   help="Đường dẫn file JSONL đầu ra")
    p.add_argument("--chunk-size",    type=int, default=1000)
    p.add_argument("--chunk-overlap", type=int, default=150)
    p.add_argument("--min-chunk",     type=int, default=200)
    p.add_argument("--max-chunk",     type=int, default=1500)
    p.add_argument("--no-headers",    action="store_true",
                   help="Không thêm tiêu đề section vào đầu chunk")
    p.add_argument("--log-every",     type=int, default=500)
    return p


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    args   = _build_arg_parser().parse_args()
    config = ChunkConfig(
        chunk_size            = args.chunk_size,
        chunk_overlap         = args.chunk_overlap,
        min_chunk_size        = args.min_chunk,
        max_chunk_size        = args.max_chunk,
        include_section_headers = not args.no_headers,
    )

    run_chunking(
        input_path  = args.input,
        output_path = args.output,
        config      = config,
        log_every   = args.log_every,
    )
