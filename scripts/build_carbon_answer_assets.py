#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services.carbon_answer_assets_service import (
    DEFAULT_CARBON_CORPUS_MANIFEST,
    DEFAULT_CARBON_CORPUS_PATH,
    build_default_carbon_corpus,
    write_carbon_corpus,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build generated carbon answer assets and corpus passages.")
    parser.add_argument(
        "--corpus-out",
        default=str(DEFAULT_CARBON_CORPUS_PATH),
        help="Output JSONL path for generated carbon answer assets.",
    )
    parser.add_argument(
        "--manifest-out",
        default=str(DEFAULT_CARBON_CORPUS_MANIFEST),
        help="Output manifest path for generated carbon answer assets.",
    )
    parser.add_argument(
        "--products",
        nargs="*",
        default=None,
        help="Optional subset of normalized product ids to build.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    built = build_default_carbon_corpus(product_ids=args.products)
    write_carbon_corpus(
        rows=built["rows"],
        manifest=built["manifest"],
        corpus_path=args.corpus_out,
        manifest_path=args.manifest_out,
    )

    print(json.dumps(
        {
            "corpus_path": str(Path(args.corpus_out).resolve()),
            "manifest_path": str(Path(args.manifest_out).resolve()),
            "doc_count": built["manifest"]["doc_count"],
            "product_count": built["manifest"]["product_count"],
            "products": [
                {
                    "product_id": item["product_id"],
                    "status": item["status"],
                    "doc_count": item["doc_count"],
                    "asset_kinds": item["asset_kinds"],
                }
                for item in built["manifest"]["products"]
            ],
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
