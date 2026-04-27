import os
from typing import List

class Config:
    """Cấu hình hệ thống thu thập dữ liệu Wikipedia."""
    
    # API Settings
    API_URL = "https://vi.wikipedia.org/w/api.php"
    USER_AGENT = "WikiDataPipeline/2.0 (research; contact: your-email@gmail.com)"
    
    # Paths
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, "data")
    
    TAXONOMY_FILE = os.path.join(DATA_DIR, "taxonomy.json")
    RAW_DATA_PATH = os.path.join(DATA_DIR, "wiki_pages_raw.jsonl")
    CLEAN_DATA_PATH = os.path.join(DATA_DIR, "wiki_pages_content.jsonl")
    CHECKPOINT_PATH = os.path.join(DATA_DIR, "checkpoint.json")
    
    # Crawler Settings
    MAX_DEPTH = 3
    DELAY_CRAWL = 0.5
    TARGET_PER_DOMAIN = 850
    
    # Pipeline Settings
    MAX_WORKERS = 5  # Số luồng chạy song song
    MIN_CHARS = 1500
    MAX_CHARS = 80000
    TRUNCATE_LONG = True

    # Tránh các trang hệ thống
    STUB_PREFIXES = [
        "Bản mẫu:", "Thể loại:", "Tập tin:", "Trợ giúp:", 
        "Dự án:", "Thảo luận:", "Wikipedia:", "Cổng thông tin:"
    ]

    @classmethod
    def setup_directories(cls):
        """Tạo thư mục data nếu chưa tồn tại."""
        os.makedirs(cls.DATA_DIR, exist_ok=True)