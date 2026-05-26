"""
config.py
Single source of truth cho toan bo duong dan va hang so.
Moi script khac PHAI import tu day, khong duoc hardcode path.
"""
from pathlib import Path

import yaml


# ROOT
ROOT = Path(__file__).resolve().parents[1]

# DATA LAYERS
RAW_DIR = ROOT / 'data' / 'raw'
INTERIM_DIR = ROOT / 'data' / 'interim'
PROCESSED_DIR = ROOT / 'data' / 'processed'

# Tang 1 - Raw
TAXONOMY_FILE = RAW_DIR / 'taxonomy.json'
RAW_PAGES = RAW_DIR / 'wiki_pages_content.jsonl'
RAW_METADATA = RAW_DIR / 'wiki_pages_raw.jsonl'

# Tang 2 - Interim
RAW_CHUNKS = INTERIM_DIR / 'wiki_chunks.jsonl'
FILTERED_CHUNKS = INTERIM_DIR / 'chunks_filtered.jsonl'
SAMPLED_CHUNKS = INTERIM_DIR / 'chunks_sampled.jsonl'

# Tang 3 - Processed
QA_RAW = PROCESSED_DIR / 'qa_pairs_raw.jsonl'
QA_JUDGED = PROCESSED_DIR / 'qa_pairs_judged.jsonl'
QA_VERIFIED = PROCESSED_DIR / 'qa_pairs_verified.jsonl'
QA_FINAL = PROCESSED_DIR / 'qa_dataset_final.jsonl'

# CONFIG YAML
CONFIGS_DIR = ROOT / 'configs'

# EDA OUTPUT
FIGURES_DIR = ROOT / 'eda' / 'figures'

# API SETTINGS
API_URL = "https://vi.wikipedia.org/w/api.php"
USER_AGENT = "WikiDataPipeline/2.0 (research; contact: your-email@gmail.com)"

# CRAWLER SETTINGS
MAX_DEPTH = 3
DELAY_CRAWL = 0.5
TARGET_PER_DOMAIN = 850

# CONTENT CLEANER SETTINGS
MAX_WORKERS = 5
MIN_CHARS = 1500
MAX_CHARS = 80000
TRUNCATE_LONG = True

# BLACKLIST SETTINGS
BLACKLIST_KEYWORDS: list[str] = []
STUB_PREFIXES = [
    "Bản mẫu:",
    "Thể loại:",
    "Tập tin:",
    "Trợ giúp:",
    "Dự án:",
    "Thảo luận:",
    "Wikipedia:",
    "Cổng thông tin:",
]


def load_filter_config() -> dict:
    with (CONFIGS_DIR / "filter_config.yaml").open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_qa_gen_config() -> dict:
    with (CONFIGS_DIR / "qa_gen_config.yaml").open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    return loaded or {}


def ensure_dirs() -> None:
    for directory in [RAW_DIR, INTERIM_DIR, PROCESSED_DIR, FIGURES_DIR, CONFIGS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
