"""Compute inter-annotator agreement from the human-verification bundle."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import ensure_dirs  # noqa: E402
from src.qa.iaa import compute_bundle_iaa, write_iaa_outputs  # noqa: E402

DEFAULT_BUNDLE_DIR = ROOT / "data" / "processed" / "datasets" / "human_verification_bundle_20260602"
DEFAULT_REPORT_JSON = DEFAULT_BUNDLE_DIR / "reports" / "iaa_summary.json"
DEFAULT_REPORT_MD = DEFAULT_BUNDLE_DIR / "reports" / "iaa_summary.md"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute Cohen's kappa and agreement summaries from a human-verification bundle."
    )
    parser.add_argument(
        "--bundle-dir", default=str(DEFAULT_BUNDLE_DIR), help="Path to the human verification bundle directory."
    )
    parser.add_argument("--json-output", default=str(DEFAULT_REPORT_JSON), help="Path to write the JSON summary.")
    parser.add_argument(
        "--markdown-output", default=str(DEFAULT_REPORT_MD), help="Path to write the Markdown summary."
    )
    return parser


def main() -> None:
    ensure_dirs()
    args = build_parser().parse_args()
    summary = compute_bundle_iaa(Path(args.bundle_dir))
    write_iaa_outputs(summary, json_output=Path(args.json_output), markdown_output=Path(args.markdown_output))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
