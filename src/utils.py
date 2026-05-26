import json
from pathlib import Path
from typing import Any


def load_taxonomy(filepath: str | Path, config_module: Any) -> dict:
    """
    Đọc taxonomy và cập nhật blacklist/prefix trực tiếp vào config module.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file cấu hình tại: {path}")

    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if "blacklist" in data:
        config_module.BLACKLIST_KEYWORDS = list(
            set(config_module.BLACKLIST_KEYWORDS + data["blacklist"])
        )

    if "title_blacklist_prefixes" in data:
        config_module.STUB_PREFIXES = list(
            set(config_module.STUB_PREFIXES + data["title_blacklist_prefixes"])
        )

    return data.get("domains", {})
