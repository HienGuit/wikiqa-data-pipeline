import json
import logging

from src import ChunkConfig, Config, ContentPipeline, WikipediaCrawler, load_taxonomy, run_chunking


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("Main")


def main() -> None:
    log.info("Khoi dong Wikipedia Data Engineering Pipeline...")
    Config.ensure_dirs()

    try:
        domains_data = load_taxonomy(Config.TAXONOMY_FILE, Config)
        log.info("Da nap thanh cong %s domains tu taxonomy.", len(domains_data))
    except Exception as exc:
        log.error("Loi khoi tao: %s", exc)
        return

    log.info("=" * 50)
    log.info("PHASE 1: Khoi dong Crawler...")
    crawler = WikipediaCrawler(Config)
    metadata_path = crawler.run(domains_data, Config.BLACKLIST_KEYWORDS)

    log.info("=" * 50)
    log.info("PHASE 2: Khoi dong Content Cleaner...")
    raw_records = []
    try:
        with metadata_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    raw_records.append(json.loads(line))
        log.info("Doc duoc %s bai viet can fetch noi dung.", len(raw_records))
    except FileNotFoundError:
        log.error("Khong tim thay file metadata raw. Crawler chua chay thanh cong?")
        return

    cleaner = ContentPipeline(Config)
    cleaner.process(raw_records)

    log.info("=" * 50)
    log.info("PHASE 3: Khoi dong Chunker...")
    run_chunking(
        input_path=Config.RAW_PAGES,
        output_path=Config.RAW_CHUNKS,
        config=ChunkConfig(),
    )

    log.info("=" * 50)
    log.info("Hoan tat!")


if __name__ == "__main__":
    main()
