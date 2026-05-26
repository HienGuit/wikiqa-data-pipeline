import json
import logging

from src import ChunkConfig, ContentPipeline, WikipediaCrawler, load_taxonomy, run_chunking
from src import Config


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("Main")


def main() -> None:
    log.info("Khởi động Wikipedia Data Engineering Pipeline...")
    Config.ensure_dirs()

    try:
        domains_data = load_taxonomy(Config.TAXONOMY_FILE, Config)
        log.info(f"Đã nạp thành công {len(domains_data)} domains từ taxonomy.")
    except Exception as exc:
        log.error(f"Lỗi khởi tạo: {exc}")
        return

    log.info("=" * 50)
    log.info("PHASE 1: Khởi động Crawler...")
    crawler = WikipediaCrawler(Config)
    metadata_path = crawler.run(domains_data, Config.BLACKLIST_KEYWORDS)

    log.info("=" * 50)
    log.info("PHASE 2: Khởi động Content Cleaner...")
    raw_records = []
    try:
        with metadata_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    raw_records.append(json.loads(line))
        log.info(f"Đọc được {len(raw_records)} bài viết cần fetch nội dung.")
    except FileNotFoundError:
        log.error("Không tìm thấy file metadata raw. Crawler chưa chạy thành công?")
        return

    cleaner = ContentPipeline(Config)
    cleaner.process(raw_records)

    log.info("=" * 50)
    log.info("PHASE 3: Khởi động Chunker...")
    run_chunking(
        input_path=Config.RAW_PAGES,
        output_path=Config.RAW_CHUNKS,
        config=ChunkConfig(),
    )

    log.info("=" * 50)
    log.info("Hoàn tất!")


if __name__ == "__main__":
    main()
