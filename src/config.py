"""Centralized paths and project-wide constants."""

from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]

# Core repo directories
RAW_DIR = ROOT / "data" / "raw"
INTERIM_DIR = ROOT / "data" / "interim"
PROCESSED_DIR = ROOT / "data" / "processed"
FINAL_DATA_DIR = ROOT / "data" / "final"
CONFIGS_DIR = ROOT / "configs"
FIGURES_DIR = ROOT / "eda" / "figures"
SCRIPTS_DIR = ROOT / "scripts"

# Processed-data sublayers
PROCESSED_DATASETS_DIR = PROCESSED_DIR / "datasets"
PROCESSED_RUNS_DIR = PROCESSED_DIR / "runs"
PROCESSED_REPORTS_DIR = PROCESSED_DIR / "reports"
PROCESSED_ARCHIVE_DIR = PROCESSED_DIR / "archive"
PROCESSED_WIKI_METRICS_DIR = PROCESSED_DIR / "wiki_metrics"
PROCESSED_FEATURES_DIR = PROCESSED_DIR / "features"

QA_RUNS_DIR = PROCESSED_RUNS_DIR / "qa"
QA_REPORTS_DIR = PROCESSED_REPORTS_DIR / "qa"
QA_ARCHIVE_DIR = PROCESSED_ARCHIVE_DIR / "qa"
QA_LEGACY_DATASETS_DIR = QA_ARCHIVE_DIR / "legacy_datasets"
QA_JUDGE_EXPORTS_DIR = QA_ARCHIVE_DIR / "judge_exports"
FEATURE_ENGINEERING_FIGURES_DIR = FIGURES_DIR / "03_feature_engineering_eda"

# Raw layer
TAXONOMY_FILE = RAW_DIR / "taxonomy.json"
RAW_PAGES = RAW_DIR / "wiki_pages_content.jsonl"
RAW_METADATA = RAW_DIR / "wiki_pages_raw.jsonl"

# Interim layer
RAW_CHUNKS = INTERIM_DIR / "wiki_chunks.jsonl"
FILTERED_CHUNKS = INTERIM_DIR / "chunks_filtered.jsonl"
SAMPLED_CHUNKS = INTERIM_DIR / "chunks_sampled.jsonl"
TOPUP_CHUNKS_ROUND1 = INTERIM_DIR / "chunks_topup_inferential.jsonl"
TOPUP_CHUNKS_ROUND2 = INTERIM_DIR / "chunks_topup_inferential_round2.jsonl"

# Canonical processed datasets
QA_CANONICAL = PROCESSED_DATASETS_DIR / "qa_pairs_canonical.jsonl"
QA_CANONICAL_REJECTS = PROCESSED_DATASETS_DIR / "qa_pairs_canonical_generation_rejects.jsonl"
QA_CANONICAL_JUDGED = PROCESSED_DATASETS_DIR / "qa_pairs_canonical_judged.jsonl"
QA_CANONICAL_JUDGED_GEMINI31_FLASH_LITE = (
    PROCESSED_DATASETS_DIR / "qa_pairs_canonical_judged_gemini31_flash_lite.jsonl"
)
QA_CANONICAL_JUDGED_DEEPSEEK_V4_FLASH = PROCESSED_DATASETS_DIR / "qa_pairs_canonical_judged_deepseek_v4_flash.jsonl"
QA_CANONICAL_CONTEXT_CLEANED = PROCESSED_DATASETS_DIR / "qa_pairs_canonical_context_cleaned.jsonl"
QA_CANONICAL_CONTEXT_CLEANING_REJECTS = PROCESSED_DATASETS_DIR / "qa_pairs_canonical_context_cleaning_rejects.jsonl"
QA_CANONICAL_JUDGED_CONTEXT_CLEANED = PROCESSED_DATASETS_DIR / "qa_pairs_canonical_judged_context_cleaned.jsonl"
QA_CANONICAL_JUDGED_GEMINI31_FLASH_LITE_CONTEXT_CLEANED = (
    PROCESSED_DATASETS_DIR / "qa_pairs_canonical_judged_gemini31_flash_lite_context_cleaned.jsonl"
)
QA_CANONICAL_JUDGED_DEEPSEEK_V4_FLASH_CONTEXT_CLEANED = (
    PROCESSED_DATASETS_DIR / "qa_pairs_canonical_judged_deepseek_v4_flash_context_cleaned.jsonl"
)
QA_CANONICAL_JUDGED_RELEASE = PROCESSED_DATASETS_DIR / "qa_pairs_canonical_judged_release.jsonl"
QA_CANONICAL_JUDGED_DEEPSEEK_V4_FLASH_RELEASE = (
    PROCESSED_DATASETS_DIR / "qa_pairs_canonical_judged_deepseek_v4_flash_release.jsonl"
)
QA_CANONICAL_JUDGED_CONTEXT_CLEANING_REJECTS = (
    PROCESSED_DATASETS_DIR / "qa_pairs_canonical_judged_context_cleaning_rejects.jsonl"
)
QA_SPLIT_READY = PROCESSED_DATASETS_DIR / "qa_pairs_split_ready.jsonl"
QA_INFERENTIAL_USABLE_ONLY = PROCESSED_DATASETS_DIR / "qa_inferential_usable_only.jsonl"
QA_THREE_WAY_ANALYSIS = PROCESSED_DATASETS_DIR / "qa_pairs_three_way_analysis.jsonl"
QA_THREE_WAY_READY = PROCESSED_DATASETS_DIR / "qa_pairs_three_way_ready.jsonl"
QA_THREE_WAY_EXTRACTION = PROCESSED_DATASETS_DIR / "qa_pairs_three_way_extraction.jsonl"
QA_THREE_WAY_BRIDGE = PROCESSED_DATASETS_DIR / "qa_pairs_three_way_bridge.jsonl"
QA_THREE_WAY_MULTI_SENTENCE = PROCESSED_DATASETS_DIR / "qa_pairs_three_way_multi_sentence.jsonl"
QA_ANNOTATION_POOL = QA_CANONICAL_JUDGED_CONTEXT_CLEANED
QA_ANNOTATION_POOL_LEGACY = PROCESSED_DATASETS_DIR / "qa_pairs_canonical_judged_cleaned.jsonl"
QA_ANNOTATION_POOL_LEGACY_REJECTS = PROCESSED_DATASETS_DIR / "qa_pairs_canonical_judged_cleaning_rejects.jsonl"

FINAL_DATA_DIR.mkdir(parents=True, exist_ok=True)
QA_TRAIN_SPLIT = FINAL_DATA_DIR / "train.jsonl"
QA_VAL_SPLIT = FINAL_DATA_DIR / "val.jsonl"
QA_TEST_SPLIT = FINAL_DATA_DIR / "test.jsonl"

# Feature engineering artifacts
ENTITY_DB = PROCESSED_WIKI_METRICS_DIR / "entity_db.jsonl"
FEATURE_MATRIX_FULL = PROCESSED_FEATURES_DIR / "feature_matrix_full.csv"
FEATURE_MATRIX_FINAL = FINAL_DATA_DIR / "feature_matrix_final.csv"
FEATURE_MATRIX_FETCH_REPORT = PROCESSED_WIKI_METRICS_DIR / "fetch_wiki_metrics_report.json"
FEATURE_MATRIX_BUILD_REPORT = PROCESSED_FEATURES_DIR / "feature_matrix_build_report.json"
FEATURE_MATRIX_EDA_REPORT = FEATURE_ENGINEERING_FIGURES_DIR / "feature_engineering_eda_report.json"

# Human verification datasets
QA_HUMAN_VERIFICATION_TASK1 = PROCESSED_DATASETS_DIR / "human_verification_task1_quality_difficulty_100.jsonl"
QA_HUMAN_VERIFICATION_TASK2 = PROCESSED_DATASETS_DIR / "human_verification_task2_inferential_validity_50.jsonl"
QA_HUMAN_VERIFICATION_TASK1_KEY = QA_REPORTS_DIR / "human_verification_task1_quality_difficulty_100_key.jsonl"
QA_HUMAN_VERIFICATION_TASK2_KEY = QA_REPORTS_DIR / "human_verification_task2_inferential_validity_50_key.jsonl"

# Historical / legacy processed datasets kept for traceability
QA_RAW = QA_LEGACY_DATASETS_DIR / "qa_pairs_raw.jsonl"
QA_RAW_REJECTS = QA_LEGACY_DATASETS_DIR / "qa_pairs_raw_rejects.jsonl"
QA_WITH_TOPUP = QA_LEGACY_DATASETS_DIR / "qa_pairs_with_topup.jsonl"
QA_WITH_TOPUP_REJECTS = QA_LEGACY_DATASETS_DIR / "qa_pairs_with_topup_rejects.jsonl"

# Judge label exports
QA_JUDGED_GEMINI31_FLASH_LITE = QA_JUDGE_EXPORTS_DIR / "qa_judge_openrouter_gemini31_flash_lite_flex_full.jsonl"
QA_JUDGED_GEMINI31_FLASH_LITE_REJECTS = (
    QA_JUDGE_EXPORTS_DIR / "qa_judge_openrouter_gemini31_flash_lite_flex_full_rejects.jsonl"
)
QA_JUDGED_FLASH_LEGACY = QA_JUDGE_EXPORTS_DIR / "qa_judge_full_flash.jsonl"
QA_JUDGED_FLASH_LEGACY_REJECTS = QA_JUDGE_EXPORTS_DIR / "qa_judge_full_flash_rejects.jsonl"

# Backward-compatible aliases used by existing code
QA_WITH_TOPUP_ROUND2 = QA_CANONICAL
QA_WITH_TOPUP_ROUND2_REJECTS = QA_CANONICAL_REJECTS
QA_WITH_TOPUP_ROUND2_JUDGED = QA_CANONICAL_JUDGED
QA_WITH_TOPUP_ROUND2_FILTERED = QA_SPLIT_READY
QA_JUDGED = QA_CANONICAL_JUDGED
QA_VERIFIED = PROCESSED_DATASETS_DIR / "qa_pairs_verified.jsonl"
QA_FINAL = PROCESSED_DATASETS_DIR / "qa_dataset_final.jsonl"

# Processed run directories
QA_SHARDS_DIR = QA_RUNS_DIR / "shards"
QA_TOPUP_RUN_DIR = QA_RUNS_DIR / "topup_round1"
QA_TOPUP_RUN_DIR_ROUND2 = QA_RUNS_DIR / "topup_round2"
QA_JUDGE_RUN_DIR = QA_RUNS_DIR / "judge_default"
QA_JUDGE_FLASH_MULTI_DIR = QA_RUNS_DIR / "judge_full_flash_multi"
QA_JUDGE_FLASH_EXTRACTION_DIR = QA_RUNS_DIR / "judge_full_flash_extraction"
QA_JUDGE_GEMINI31_FLASH_LITE_FULL_DIR = QA_RUNS_DIR / "judge_openrouter_gemini31_flash_lite_flex_full"
QA_REPAIR_SUCCINCT_DIR = QA_RUNS_DIR / "repair_succinct"

# Processed reports
CHUNKS_TOPUP_MANIFEST = QA_REPORTS_DIR / "chunks_topup_inferential_manifest.json"
CHUNKS_TOPUP_ROUND2_MANIFEST = QA_REPORTS_DIR / "chunks_topup_inferential_round2_manifest.json"
QA_FULL_RUN_SUMMARY = QA_REPORTS_DIR / "qa_full_run_summary.json"
QA_WITH_TOPUP_SUMMARY = QA_REPORTS_DIR / "qa_with_topup_summary.json"
QA_WITH_TOPUP_ROUND2_SUMMARY = QA_REPORTS_DIR / "qa_with_topup_round2_summary.json"
QA_INFERENTIAL_USABLE_ONLY_SUMMARY = QA_REPORTS_DIR / "qa_inferential_usable_only_summary.json"
QA_JUDGE_FULL_FLASH_SUMMARY = QA_REPORTS_DIR / "qa_judge_full_flash_summary.json"
QA_JUDGED_GEMINI31_FLASH_LITE_SUMMARY = (
    QA_REPORTS_DIR / "qa_judge_openrouter_gemini31_flash_lite_flex_full_summary.json"
)
QA_SUCCINCT_REPAIR_MERGE_MANIFEST = QA_REPORTS_DIR / "qa_pairs_canonical_succinct_repair_merge_manifest.json"
QA_REFRESH_DERIVED_REPORT = QA_REPORTS_DIR / "qa_refresh_derived_report.json"
QA_OFFICIAL_SELECTION_REPORT = QA_REPORTS_DIR / "qa_official_dataset_selection_report.json"
QA_CONTEXT_CLEANING_REPORT = QA_REPORTS_DIR / "qa_context_cleaning_report.json"
QA_CONTEXT_CLEANED_SYNC_REPORT = QA_REPORTS_DIR / "qa_context_cleaned_sync_report.json"
QA_DATASET_FINALIZATION_REPORT = QA_REPORTS_DIR / "qa_dataset_finalization_report.json"
QA_HUMAN_VERIFICATION_SAMPLING_REPORT = QA_REPORTS_DIR / "human_verification_sampling_report.json"
QA_ANNOTATION_POOL_COMPAT_REPORT = QA_REPORTS_DIR / "qa_annotation_pool_compat_report.json"
QA_THREE_WAY_REPORT = QA_REPORTS_DIR / "qa_three_way_split_report.json"
QA_THREE_WAY_FINAL_VALIDATION_REPORT = QA_REPORTS_DIR / "qa_three_way_final_validation_report.json"
QA_JUDGE_PROVENANCE_REPORT = QA_REPORTS_DIR / "qa_judge_provenance_report.json"
QA_JUDGE_CANONICAL_PROMOTION_REPORT = QA_REPORTS_DIR / "qa_judge_canonical_promotion_report.json"
QA_JUDGED_RELEASE_NORMALIZATION_REPORT = QA_REPORTS_DIR / "qa_judged_release_normalization_report.json"
QA_SPLIT_DISTRIBUTION_REPORT = QA_REPORTS_DIR / "qa_split_distribution_report.json"
FEATURE_PHASE1_PROVENANCE_JSON = QA_REPORTS_DIR / "feature_phase1_provenance.json"
FEATURE_PHASE1_PROVENANCE_MD = QA_REPORTS_DIR / "feature_phase1_provenance.md"
FINAL_RELEASE_MANIFEST_JSON = QA_REPORTS_DIR / "final_release_manifest.json"
FINAL_RELEASE_MANIFEST_MD = QA_REPORTS_DIR / "final_release_manifest.md"

# Archive / backups
QA_WITH_TOPUP_ROUND2_BACKUP = QA_ARCHIVE_DIR / "qa_pairs_canonical.before_succinct_repair_20260601.jsonl"

# API settings
API_URL = "https://vi.wikipedia.org/w/api.php"
USER_AGENT = "WikiDataPipeline/2.0 (research; contact: your-email@gmail.com)"

# Crawler settings
MAX_DEPTH = 3
DELAY_CRAWL = 0.5
TARGET_PER_DOMAIN = 850

# Content cleaner settings
MAX_WORKERS = 5
MIN_CHARS = 1500
MAX_CHARS = 80000
TRUNCATE_LONG = True

# Blacklist settings
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
    directories = [
        RAW_DIR,
        INTERIM_DIR,
        PROCESSED_DIR,
        PROCESSED_DATASETS_DIR,
        PROCESSED_WIKI_METRICS_DIR,
        PROCESSED_FEATURES_DIR,
        PROCESSED_RUNS_DIR,
        PROCESSED_REPORTS_DIR,
        PROCESSED_ARCHIVE_DIR,
        FINAL_DATA_DIR,
        QA_RUNS_DIR,
        QA_REPORTS_DIR,
        QA_ARCHIVE_DIR,
        QA_LEGACY_DATASETS_DIR,
        QA_JUDGE_EXPORTS_DIR,
        QA_SHARDS_DIR,
        QA_TOPUP_RUN_DIR,
        QA_TOPUP_RUN_DIR_ROUND2,
        QA_JUDGE_RUN_DIR,
        QA_JUDGE_FLASH_MULTI_DIR,
        QA_JUDGE_FLASH_EXTRACTION_DIR,
        QA_JUDGE_GEMINI31_FLASH_LITE_FULL_DIR,
        QA_REPAIR_SUCCINCT_DIR,
        FIGURES_DIR,
        FEATURE_ENGINEERING_FIGURES_DIR,
        CONFIGS_DIR,
        SCRIPTS_DIR,
    ]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
