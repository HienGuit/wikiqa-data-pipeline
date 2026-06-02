"""Create publication-ready IAA visualizations from a human-verification bundle."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import ensure_dirs  # noqa: E402
from src.qa.human_verification import load_json, write_json  # noqa: E402
from src.qa.iaa import (  # noqa: E402
    build_visualization_report,
    configure_iaa_matplotlib,
    create_heatmap_figure,
    create_task1_confusion,
    create_task2_confusion,
)

DEFAULT_BUNDLE_DIR = ROOT / "data" / "processed" / "datasets" / "human_verification_bundle_20260602"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create IAA visualizations from a human-verification bundle.")
    parser.add_argument("--bundle-dir", default=str(DEFAULT_BUNDLE_DIR), help="Path to the human-verification bundle.")
    return parser


def main() -> None:
    ensure_dirs()
    configure_iaa_matplotlib()
    args = build_parser().parse_args()
    bundle_dir = Path(args.bundle_dir)
    report_dir = bundle_dir / "reports"
    tasks_dir = bundle_dir / "tasks"

    iaa_summary = load_json(report_dir / "iaa_summary.json")
    task1_rows = load_json(tasks_dir / "task1.json")
    task2_rows = load_json(tasks_dir / "task2.json")

    paths = build_visualization_report(bundle_dir)
    create_heatmap_figure(iaa_summary, Path(paths["heatmap"]))
    create_task1_confusion(task1_rows, Path(paths["task1_confusion"]))
    create_task2_confusion(task2_rows, Path(paths["task2_confusion"]))

    write_json(report_dir / "iaa_visualization_report.json", paths)
    print(json.dumps(paths, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
