"""Fetch Wikipedia/Wikidata metrics for unique QA titles."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import aiohttp
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import (  # noqa: E402
    API_URL,
    ENTITY_DB,
    FEATURE_MATRIX_FETCH_REPORT,
    QA_THREE_WAY_READY,
    USER_AGENT,
    ensure_dirs,
)
from src.features.entity_utils import normalize_entity_name  # noqa: E402

PAGEVIEWS_URL = (
    "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/"
    "vi.wikipedia.org/all-access/user/{title}/daily/{start}/{end}"
)
WIKIDATA_ENTITY_URL = "https://www.wikidata.org/wiki/Special:EntityData/{entity_id}.json"


@dataclass
class FetchConfig:
    concurrent: int = 50
    retries: int = 3
    timeout_seconds: int = 20


def load_unique_titles(path: str | Path, limit_titles: int | None = None) -> list[str]:
    file_path = Path(path)
    if file_path.suffix.lower() == ".jsonl":
        titles = set()
        with file_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                title = str(record.get("title", "")).strip()
                if title:
                    titles.add(title)
    elif file_path.suffix.lower() == ".csv":
        frame = pd.read_csv(file_path)
        titles = {str(value).strip() for value in frame["title"].tolist() if str(value).strip()}
    else:
        frame = pd.read_parquet(file_path)
        titles = {str(value).strip() for value in frame["title"].tolist() if str(value).strip()}

    ordered = sorted(titles)
    return ordered[:limit_titles] if limit_titles else ordered


def normalize_title(title: str) -> str:
    return " ".join(str(title or "").split()).strip()


def slugify_title(title: str) -> str:
    return normalize_title(title).replace(" ", "_")


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def read_existing_cache(path: str | Path) -> dict[str, dict]:
    file_path = Path(path)
    if not file_path.exists():
        return {}
    cache: dict[str, dict] = {}
    with file_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            cache[record["entity_name_normalized"]] = record
    return cache


async def fetch_json(
    session: aiohttp.ClientSession, url: str, *, params: dict | None = None, retries: int = 3
) -> dict:
    delay = 1.0
    for attempt in range(retries):
        try:
            async with session.get(url, params=params) as response:
                if response.status in {429, 500, 502, 503, 504}:
                    raise aiohttp.ClientResponseError(
                        request_info=response.request_info,
                        history=response.history,
                        status=response.status,
                        message=f"retryable status {response.status}",
                        headers=response.headers,
                    )
                response.raise_for_status()
                return await response.json()
        except Exception:
            if attempt == retries - 1:
                return {}
            await asyncio.sleep(delay)
            delay *= 2
    return {}


async def fetch_page_metadata(session: aiohttp.ClientSession, title: str, retries: int) -> dict:
    params = {
        "action": "query",
        "format": "json",
        "prop": "pageprops|categories",
        "titles": title,
        "cllimit": "max",
    }
    payload = await fetch_json(session, API_URL, params=params, retries=retries)
    pages = payload.get("query", {}).get("pages", {})
    if not pages:
        return {}
    page = next(iter(pages.values()))
    pageprops = page.get("pageprops", {})
    categories = page.get("categories", [])
    return {
        "pageid": page.get("pageid"),
        "wikidata_item": pageprops.get("wikibase_item"),
        "wiki_count": len(categories) if categories else None,
        "wiki_count_status": "ok" if categories else "unavailable",
    }


async def fetch_pageviews(session: aiohttp.ClientSession, title: str, retries: int) -> tuple[int | None, str]:
    url = PAGEVIEWS_URL.format(title=slugify_title(title), start="20250101", end="20251231")
    payload = await fetch_json(session, url, retries=retries)
    items = payload.get("items", [])
    if not items:
        return None, "unavailable"
    return int(sum(item.get("views", 0) for item in items)), "ok"


async def fetch_wikidata_stats(session: aiohttp.ClientSession, entity_id: str | None, retries: int) -> dict:
    if not entity_id:
        return {
            "site_links": None,
            "site_links_status": "missing_wikidata_item",
            "statements": None,
            "statements_status": "missing_wikidata_item",
            "references": None,
            "references_status": "missing_wikidata_item",
        }
    url = WIKIDATA_ENTITY_URL.format(entity_id=entity_id)
    payload = await fetch_json(session, url, retries=retries)
    entity = payload.get("entities", {}).get(entity_id, {})
    if not entity:
        return {
            "site_links": None,
            "site_links_status": "unavailable",
            "statements": None,
            "statements_status": "unavailable",
            "references": None,
            "references_status": "unavailable",
        }
    sitelinks = entity.get("sitelinks", {})
    claims = entity.get("claims", {})
    statement_count = sum(len(values) for values in claims.values())
    reference_count = 0
    for values in claims.values():
        for value in values:
            reference_count += len(value.get("references", []))
    return {
        "site_links": len(sitelinks),
        "site_links_status": "ok",
        "statements": statement_count,
        "statements_status": "ok",
        "references": reference_count,
        "references_status": "ok",
    }


async def fetch_title_metrics(
    title: str,
    *,
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    retries: int,
) -> dict:
    normalized = normalize_title(title)
    async with semaphore:
        metadata = await fetch_page_metadata(session, normalized, retries)
        page_views, page_views_status = await fetch_pageviews(session, normalized, retries)
        wikidata_stats = await fetch_wikidata_stats(session, metadata.get("wikidata_item"), retries)

    return {
        "entity_name": normalized,
        "entity_name_normalized": normalize_entity_name(normalized),
        "pageid": metadata.get("pageid"),
        "page_views": page_views,
        "page_views_status": page_views_status,
        "site_links": wikidata_stats["site_links"],
        "site_links_status": wikidata_stats["site_links_status"],
        "wiki_count": metadata.get("wiki_count"),
        "wiki_count_status": metadata.get("wiki_count_status", "unavailable"),
        "statements": wikidata_stats["statements"],
        "statements_status": wikidata_stats["statements_status"],
        "references": wikidata_stats["references"],
        "references_status": wikidata_stats["references_status"],
        "api_source": "wikipedia_pageviews+mediawiki+wikidata",
        "fetch_timestamp": now_iso(),
    }


async def run_async(titles: list[str], cache: dict[str, dict], cfg: FetchConfig) -> list[dict]:
    timeout = aiohttp.ClientTimeout(total=cfg.timeout_seconds)
    connector = aiohttp.TCPConnector(limit=cfg.concurrent, ssl=False)
    semaphore = asyncio.Semaphore(cfg.concurrent)
    async with aiohttp.ClientSession(
        timeout=timeout,
        connector=connector,
        headers={"User-Agent": USER_AGENT},
    ) as session:
        tasks = []
        for title in titles:
            normalized = normalize_entity_name(title)
            if normalized in cache:
                continue
            tasks.append(fetch_title_metrics(title, session=session, semaphore=semaphore, retries=cfg.retries))
        if not tasks:
            return []
        return await asyncio.gather(*tasks)


def write_cache(path: str | Path, cache: dict[str, dict]) -> None:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8") as handle:
        for key in sorted(cache):
            handle.write(json.dumps(cache[key], ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch async wiki metrics for feature engineering.")
    parser.add_argument("--input", default=str(QA_THREE_WAY_READY))
    parser.add_argument("--output", default=str(ENTITY_DB))
    parser.add_argument("--report", default=str(FEATURE_MATRIX_FETCH_REPORT))
    parser.add_argument("--concurrent", type=int, default=50)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--limit-titles", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ensure_dirs()
    args = parse_args()
    titles = load_unique_titles(args.input, args.limit_titles)
    cache = read_existing_cache(args.output)
    cfg = FetchConfig(concurrent=args.concurrent, retries=args.retries)
    fetched = asyncio.run(run_async(titles, cache, cfg))
    for record in fetched:
        cache[record["entity_name_normalized"]] = record
    write_cache(args.output, cache)

    status_counter = Counter(record.get("page_views_status", "unknown") for record in cache.values())
    report = {
        "input": str(args.input),
        "output": str(args.output),
        "title_count_requested": len(titles),
        "cache_size": len(cache),
        "new_records": len(fetched),
        "status_counts": dict(status_counter),
        "limit_titles": args.limit_titles,
        "timestamp": now_iso(),
    }
    Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
