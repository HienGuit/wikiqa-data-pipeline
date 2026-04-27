import json
import os
import logging

def load_taxonomy(filepath: str, config_class) -> dict:
    """
    Đọc file taxonomy.json và tự động cập nhật Blacklist vào Config.
    Trả về dictionary chứa danh sách các domains.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Không tìm thấy file cấu hình tại: {filepath}")
        
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    # Cập nhật tự động blacklist từ file JSON vào class Config
    if "blacklist" in data:
        # Loại bỏ trùng lặp nếu có
        config_class.BLACKLIST_KEYWORDS = list(set(config_class.BLACKLIST_KEYWORDS + data["blacklist"]))
        
    if "title_blacklist_prefixes" in data:
        config_class.STUB_PREFIXES = list(set(config_class.STUB_PREFIXES + data["title_blacklist_prefixes"]))
        
    return data.get("domains", {})