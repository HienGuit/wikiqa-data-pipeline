import concurrent.futures
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import requests


class ContentPipeline:
    """Pipeline tải và làm sạch nội dung bài viết."""

    def __init__(self, config: Any):
        self.cfg = config
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": self.cfg.USER_AGENT})
        self.log = logging.getLogger("ContentCleaner")

    def fetch_extract(self, pageid: int) -> Dict:
        """Tải plain text sạch từ Wikipedia Extracts API."""
        params = {
            "action": "query",
            "format": "json",
            "prop": "extracts|info",
            "explaintext": True,
            "exsectionformat": "plain",
            "inprop": "url",
            "pageids": str(pageid),
        }
        try:
            response = self.session.get(self.cfg.API_URL, params=params, timeout=20)
            page = response.json().get("query", {}).get("pages", {}).get(str(pageid), {})
            return {"text": page.get("extract", ""), "url": page.get("fullurl", "")}
        except Exception:
            return {}

    def process(self, raw_metadata: List[Dict]) -> Path:
        """Fetch đa luồng và ghi file nội dung sạch."""
        output_path = Path(self.cfg.RAW_PAGES)
        tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with tmp_path.open("w", encoding="utf-8") as handle:
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.cfg.MAX_WORKERS) as executor:
                future_to_record = {executor.submit(self.fetch_extract, row["pageid"]): row for row in raw_metadata}

                for index, future in enumerate(concurrent.futures.as_completed(future_to_record), start=1):
                    metadata = future_to_record[future]
                    content = future.result()

                    if content.get("text") and len(content["text"]) >= self.cfg.MIN_CHARS:
                        text = (
                            content["text"][: self.cfg.MAX_CHARS]
                            if self.cfg.TRUNCATE_LONG
                            else content["text"]
                        )
                        payload = {**metadata, "text": text, "url": content["url"]}
                        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

                    if index % 100 == 0:
                        self.log.info(f"Progress: {index}/{len(raw_metadata)} articles processed.")

        tmp_path.replace(output_path)
        self.log.info("Content cleaner completed successfully.")
        return output_path
