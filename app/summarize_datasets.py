"""
summarize_datasets.py — CLI entry point for the dataset summarizer.

Scans a data directory, asks the LLM to describe each dataset (if
``IDS_USE_LLM`` is enabled), and writes a Markdown report to disk.

Usage
-----
    python -m app.summarize_datasets                     # scans ./data
    python -m app.summarize_datasets data/archive        # custom path
    python -m app.summarize_datasets data/ docs/DATA.md  # custom output

Flags
-----
    --no-llm        Skip LLM enrichment even if IDS_USE_LLM is set.
    --json PATH     Also emit the raw catalog as JSON to PATH.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from app.dataset_summary import summarize_datasets
from app.logging_config import setup_logging


def main() -> int:
    """Parse arguments, run the summarizer, return a process exit code."""
    parser = argparse.ArgumentParser(
        prog="python -m app.summarize_datasets",
        description="LLM-assisted dataset inventory for the Trainable IDS.",
    )
    parser.add_argument(
        "root",
        nargs="?",
        default="data",
        help="Directory to scan (default: data).",
    )
    parser.add_argument(
        "output",
        nargs="?",
        default="docs/DATASETS.md",
        help="Markdown output path (default: docs/DATASETS.md).",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Skip the LLM enrichment step and produce a structural report only.",
    )
    parser.add_argument(
        "--json",
        metavar="PATH",
        help="Also write the raw catalog as JSON to this path.",
    )
    args = parser.parse_args()

    setup_logging()

    root = Path(args.root)
    if not root.exists():
        print(f"[error] data directory not found: {root}", file=sys.stderr)
        return 2

    catalog = summarize_datasets(
        root_path=root,
        output_path=args.output,
        use_llm=False if args.no_llm else None,
    )

    if args.json:
        json_path = Path(args.json)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(
            json.dumps(catalog.to_dict(), indent=2, default=str),
            encoding="utf-8",
        )

    print(f"Scanned {len(catalog.datasets)} dataset(s) from {root}")
    print(f"Markdown report: {args.output}")
    if args.json:
        print(f"JSON catalog:    {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
