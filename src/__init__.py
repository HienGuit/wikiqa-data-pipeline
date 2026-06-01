"""Public project surface for the Wikipedia-to-QA pipeline."""

from __future__ import annotations

from . import config as Config
from .content_cleaner import ContentPipeline
from .crawler import WikipediaCrawler
from .utils import load_taxonomy

__all__ = [
    "ChunkConfig",
    "Config",
    "ContentPipeline",
    "WikipediaCrawler",
    "load_taxonomy",
    "run_chunking",
]


def __getattr__(name: str):
    if name in {"ChunkConfig", "run_chunking"}:
        from .chunker import ChunkConfig, run_chunking

        exports = {
            "ChunkConfig": ChunkConfig,
            "run_chunking": run_chunking,
        }
        return exports[name]
    raise AttributeError(f"module 'src' has no attribute {name!r}")
