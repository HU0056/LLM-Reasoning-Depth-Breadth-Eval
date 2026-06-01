from __future__ import annotations

import argparse
from pathlib import Path

from .config import PipelineConfig
from .pipeline import build_gsm8k_graphs, download_gsm8k


def add_shared_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="Repository root directory.",
    )
    parser.add_argument(
        "--bound",
        type=float,
        default=None,
        help="Similarity threshold multiplier. Default is 1 - 1/e.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="GSM8K reasoning graph builder")
    add_shared_arguments(parser)

    subparsers = parser.add_subparsers(dest="command", required=True)
    download_parser = subparsers.add_parser(
        "download-gsm8k",
        help="Download and save GSM8K raw data.",
    )
    add_shared_arguments(download_parser)

    build_parser = subparsers.add_parser(
        "build-gsm8k-graphs",
        help="Build graph JSONL files from GSM8K.",
    )
    add_shared_arguments(build_parser)

    run_all_parser = subparsers.add_parser(
        "run-all",
        help="Download GSM8K and build graph JSONL files.",
    )
    add_shared_arguments(run_all_parser)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = PipelineConfig(
        root_dir=args.root,
        **({} if args.bound is None else {"bound": args.bound}),
    )

    if args.command == "download-gsm8k":
        written = download_gsm8k(config)
    elif args.command == "build-gsm8k-graphs":
        written = build_gsm8k_graphs(config)
    else:
        written = download_gsm8k(config) + build_gsm8k_graphs(config)

    for path in written:
        print(path)


if __name__ == "__main__":
    main()
