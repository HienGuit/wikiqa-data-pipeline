import json
import logging
from collections import deque
from pathlib import Path
from typing import Any, Dict, List

import requests


class WikipediaCrawler:
    """Trình thu thập siêu dữ liệu bài viết dựa trên taxonomy."""

    def __init__(self, config: Any):
        self.cfg = config
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": self.cfg.USER_AGENT})
        self.log = logging.getLogger("Crawler")

    def _is_excluded(self, title: str, blacklist: List[str]) -> bool:
        """Kiểm tra bài viết có thuộc diện loại trừ không."""
        if any(title.startswith(prefix) for prefix in self.cfg.STUB_PREFIXES):
            return True
        return any(keyword.lower() in title.lower() for keyword in blacklist)

    def _fetch_members(self, category: str, member_type: str = "page") -> List[Dict]:
        """Lấy danh sách thành viên của một category qua API."""
        params = {
            "action": "query",
            "format": "json",
            "list": "categorymembers",
            "cmtitle": f"Thể loại:{category}",
            "cmtype": member_type,
            "cmlimit": "500",
        }
        try:
            response = self.session.get(self.cfg.API_URL, params=params, timeout=15)
            response.raise_for_status()
            return response.json().get("query", {}).get("categorymembers", [])
        except Exception as exc:
            self.log.error(f"Lỗi API Category '{category}': {exc}")
            return []

    def run(self, domains: Dict, blacklist: List[str]) -> Path:
        """Thực thi BFS và ghi metadata ra file raw."""
        output_path = Path(self.cfg.RAW_METADATA)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open("w", encoding="utf-8") as handle:
            global_seen = set()
            for key, domain_cfg in domains.items():
                self.log.info(f"Processing Domain: {domain_cfg['label']}")
                rollover = 0

                for category_cfg in domain_cfg["categories"]:
                    target = category_cfg["quota"] + rollover
                    branch_count = 0
                    queue = deque([{"name": category_cfg["name"], "depth": 0}])
                    visited = set()

                    while queue and branch_count < target:
                        current = queue.popleft()
                        if current["name"] in visited:
                            continue
                        visited.add(current["name"])

                        pages = self._fetch_members(current["name"], "page")
                        for page in pages:
                            if branch_count >= target:
                                break
                            if page["pageid"] in global_seen:
                                continue
                            if self._is_excluded(page["title"], blacklist):
                                continue

                            global_seen.add(page["pageid"])
                            branch_count += 1
                            payload = {**page, "domain": key, "scope": category_cfg["scope"]}
                            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

                        if current["depth"] < self.cfg.MAX_DEPTH:
                            subcats = self._fetch_members(current["name"], "subcat")
                            for subcat in subcats:
                                queue.append(
                                    {
                                        "name": subcat["title"].replace("Thể loại:", ""),
                                        "depth": current["depth"] + 1,
                                    }
                                )

                    rollover = max(0, target - branch_count)

        return output_path
