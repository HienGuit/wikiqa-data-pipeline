import json
import os
import shutil
import logging
import concurrent.futures
from typing import Dict, List, Any
import requests

class ContentPipeline:
    """Pipeline tải và làm sạch nội dung bài viết."""

    def __init__(self, config: Any):
        self.cfg = config
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": self.cfg.USER_AGENT})
        self.log = logging.getLogger("Pipeline")

    def fetch_extract(self, pageid: int) -> Dict:
        """Tải Plain Text sạch từ Wikipedia Extracts API."""
        params = {
            "action": "query", "format": "json", "prop": "extracts|info",
            "explaintext": True, "exsectionformat": "plain", "inprop": "url",
            "pageids": str(pageid)
        }
        try:
            resp = self.session.get(self.cfg.API_URL, params=params, timeout=20)
            page = resp.json().get("query", {}).get("pages", {}).get(str(pageid), {})
            return {"text": page.get("extract", ""), "url": page.get("fullurl", "")}
        except Exception:
            return {}

    def process(self, raw_metadata: List[Dict]):
        """Thực thi Fetch đa luồng và ghi dữ liệu an toàn."""
        tmp_path = self.cfg.CLEAN_DATA_PATH + ".tmp"
        done_ids = set()
        
        with open(tmp_path, "w", encoding="utf-8") as f:
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.cfg.MAX_WORKERS) as executor:
                future_to_id = {executor.submit(self.fetch_extract, r["pageid"]): r for r in raw_metadata}
                
                for i, future in enumerate(concurrent.futures.as_completed(future_to_id)):
                    meta = future_to_id[future]
                    content = future.result()
                    
                    if content.get("text") and len(content["text"]) >= self.cfg.MIN_CHARS:
                        text = content["text"][:self.cfg.MAX_CHARS] if self.cfg.TRUNCATE_LONG else content["text"]
                        data = {**meta, "text": text, "url": content["url"]}
                        f.write(json.dumps(data, ensure_ascii=False) + "\n")
                    
                    if (i + 1) % 100 == 0:
                        self.log.info(f"Progress: {i+1}/{len(raw_metadata)} articles processed.")

        shutil.move(tmp_path, self.cfg.CLEAN_DATA_PATH)
        self.log.info("Pipeline completed successfully.")