"""scripts/download.py — CLI entry point for downloading Xeno-canto recordings.

Usage:
    python scripts/download.py
    python scripts/download.py --config config/default.yaml
    python scripts/download.py --species "Turdus merula" "Parus major" --max 50
    python scripts/download.py --countries Italy Austria --quality A
"""

import argparse
import sys
from pathlib import Path

# Allow running from the project root without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml

from src.download import XenoCantoDownloader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download bird recordings from Xeno-canto",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config", default="config/default.yaml", help="Path to YAML config file"
    )
    parser.add_argument(
        "--species", nargs="+", help="Override species list from config"
    )
    parser.add_argument(
        "--max", type=int, help="Max recordings per species (overrides config)"
    )
    parser.add_argument(
        "--quality", help="Min quality rating A–E (overrides config)"
    )
    parser.add_argument(
        "--output-dir", help="Raw audio output directory (overrides config)"
    )
    parser.add_argument(
        "--countries", nargs="+", help="Filter by countries, e.g. Italy Austria"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    species    = args.species    or cfg["species"]
    max_per    = args.max        or cfg["download"]["max_per_species"]
    quality    = args.quality    or cfg["download"]["quality"]
    output_dir = args.output_dir or cfg["data"]["raw_dir"]
    countries  = args.countries  or cfg["download"].get("countries") or None

    downloader = XenoCantoDownloader(output_dir=output_dir)
    results = downloader.download_species(
        species_list=species,
        max_per_species=max_per,
        quality=quality,
        countries=countries if countries else None,
    )

    total = sum(len(v) for v in results.values())
    print(f"\nDownload complete: {total} files across {len(results)} species.")


if __name__ == "__main__":
    main()
