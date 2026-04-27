import time
import json
import logging
from collections import deque
from typing import List, Dict, Set, Any
import requests

class WikipediaCrawler:
    """Trình thu thập siêu dữ liệu bài viết (Metadata) dựa trên Taxonomy."""

    def __init__(self, config: Any):
        self.cfg = config
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": self.cfg.USER_AGENT})
        self.log = logging.getLogger("Crawler")

    def _is_excluded(self, title: str, blacklist: List[str]) -> bool:
        """Kiểm tra bài viết có thuộc diện loại trừ không."""
        if any(title.startswith(p) for p in self.cfg.STUB_PREFIXES):
            return True
        return any(kw.lower() in title.lower() for kw in blacklist)

    def _fetch_members(self, category: str, m_type: str = "page") -> List[Dict]:
        """Lấy danh sách thành viên của một Category qua API."""
        params = {
            "action": "query", "format": "json", "list": "categorymembers",
            "cmtitle": f"Thể loại:{category}", "cmtype": m_type, "cmlimit": "500"
        }
        try:
            resp = self.session.get(self.cfg.API_URL, params=params, timeout=15)
            resp.raise_for_status()
            return resp.json().get("query", {}).get("categorymembers", [])
        except Exception as e:
            self.log.error(f"Lỗi API Category '{category}': {e}")
            return []

    def run(self, domains: Dict, blacklist: List[str]):
        """Thực thi thuật toán BFS với cơ chế Quota Rollover."""
        with open(self.cfg.RAW_DATA_PATH, "w", encoding="utf-8") as f:
            global_seen = set()
            for key, d_cfg in domains.items():
                self.log.info(f"Processing Domain: {d_cfg['label']}")
                domain_collected = set()
                rollover = 0

                for cat in d_cfg["categories"]:
                    target = cat["quota"] + rollover
                    branch_count = 0
                    queue = deque([{"name": cat["name"], "depth": 0}])
                    visited = set()

                    while queue and branch_count < target:
                        curr = queue.popleft()
                        if curr["name"] in visited: continue
                        visited.add(curr["name"])

                        # Lấy bài viết
                        pages = self._fetch_members(curr["name"], "page")
                        for p in pages:
                            if branch_count >= target: break
                            if p["pageid"] not in global_seen and not self._is_excluded(p["title"], blacklist):
                                global_seen.add(p["pageid"])
                                domain_collected.add(p["pageid"])
                                branch_count += 1
                                f.write(json.dumps({**p, "domain": key, "scope": cat["scope"]}, ensure_ascii=False) + "\n")

                        # Lấy subcat
                        if curr["depth"] < self.cfg.MAX_DEPTH:
                            subcats = self._fetch_members(curr["name"], "subcat")
                            for sc in subcats:
                                queue.append({"name": sc["title"].replace("Thể loại:", ""), "depth": curr["depth"]+1})
                    
                    rollover = max(0, target - branch_count)