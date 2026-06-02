"""Wikipedia metadata crawler based on taxonomy-defined category quotas."""

from __future__ import annotations

import json
import logging
from collections import deque
from pathlib import Path
from typing import Any, Dict, List

import requests


class WikipediaCrawler:
    """Breadth-first metadata crawler over Wikipedia categories."""

    def __init__(self, config: Any):
        self.cfg = config
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": self.cfg.USER_AGENT})
        self.log = logging.getLogger("Crawler")

    def _is_excluded(self, title: str, blacklist: List[str]) -> bool:
        if any(title.startswith(prefix) for prefix in self.cfg.STUB_PREFIXES):
            return True
        return any(keyword.lower() in title.lower() for keyword in blacklist)

    def _fetch_members(self, category: str, member_type: str = "page") -> List[Dict[str, Any]]:
        """Fetch category members from the MediaWiki API."""

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
            self.log.error("Loi API category '%s': %s", category, exc)
            return []

    def _crawl_branch(
        self,
        *,
        domain_key: str,
        category_cfg: Dict[str, Any],
        target: int,
        blacklist: List[str],
        global_seen: set[int],
        handle,
    ) -> int:
        """Crawl one category branch and return the accepted page count."""

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
                payload = {**page, "domain": domain_key, "scope": category_cfg["scope"]}
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

        return branch_count

    def run(self, domains: Dict[str, Any], blacklist: List[str]) -> Path:
        """Run the metadata crawl and save raw article metadata."""

        output_path = Path(self.cfg.RAW_METADATA)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open("w", encoding="utf-8") as handle:
            global_seen: set[int] = set()
            for domain_key, domain_cfg in domains.items():
                self.log.info("Processing domain: %s", domain_cfg["label"])
                rollover = 0

                for category_cfg in domain_cfg["categories"]:
                    target = category_cfg["quota"] + rollover
                    accepted = self._crawl_branch(
                        domain_key=domain_key,
                        category_cfg=category_cfg,
                        target=target,
                        blacklist=blacklist,
                        global_seen=global_seen,
                        handle=handle,
                    )
                    rollover = max(0, target - accepted)

        return output_path
