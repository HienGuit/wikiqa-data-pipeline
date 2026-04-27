import logging
import json
from src import Config, WikipediaCrawler, ContentPipeline, load_taxonomy

# Cấu hình log hiển thị ra màn hình cực kỳ chuyên nghiệp
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger("Main")

def main():
    log.info("Khởi động Wikipedia Data Engineering Pipeline...")
    
    # 1. Đảm bảo thư mục data đã tồn tại
    Config.setup_directories()
    
    # 2. Đọc file taxonomy và cập nhật cấu hình blacklist
    try:
        domains_data = load_taxonomy(Config.TAXONOMY_FILE, Config)
        log.info(f"Đã nạp thành công {len(domains_data)} domains từ taxonomy.json")
    except Exception as e:
        log.error(f"Lỗi khởi tạo: {e}")
        return

    # 3. CHẠY CRAWLER (Thu thập Metadata)
    log.info("=" * 50)
    log.info("PHASE 1: Khởi động Crawler...")
    crawler = WikipediaCrawler(Config)
    crawler.run(domains_data, Config.BLACKLIST_KEYWORDS)
    
    # 4. CHUẨN BỊ DỮ LIỆU CHO PIPELINE
    log.info("=" * 50)
    log.info("PHASE 2: Khởi động Content Pipeline...")
    
    raw_records = []
    # Đọc lại file raw do Crawler vừa tạo ra
    try:
        with open(Config.RAW_DATA_PATH, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    raw_records.append(json.loads(line))
        log.info(f"Đọc được {len(raw_records)} bài viết cần fetch nội dung.")
    except FileNotFoundError:
        log.error("Không tìm thấy file Raw Data. Crawler chưa chạy thành công?")
        return

    # 5. CHẠY PIPELINE (Tải Plain Text & Làm sạch)
    pipeline = ContentPipeline(Config)
    pipeline.process(raw_records)
    
    log.info("=" * 50)
    log.info("Hoàn tất!")

if __name__ == "__main__":
    main()