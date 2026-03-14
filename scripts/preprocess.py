"""scripts/preprocess.py — CLI entry point for audio-to-spectrogram conversion.

Converts .mp3 files under data/raw/<species>/ into mel-spectrogram .png images
saved under data/processed/<species>/.

Usage:
    python scripts/preprocess.py
    python scripts/preprocess.py --config config/default.yaml
    python scripts/preprocess.py --input-dir data/raw --output-dir data/processed
    python scripts/preprocess.py --overwrite
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml

from src.preprocessing import AudioConfig, SpectrogramConverter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert .mp3 recordings to mel-spectrogram images",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config", default="config/default.yaml", help="Path to YAML config file"
    )
    parser.add_argument(
        "--input-dir", help="Raw audio directory (overrides config)"
    )
    parser.add_argument(
        "--output-dir", help="Processed images directory (overrides config)"
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Reprocess files that already exist"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    input_dir  = args.input_dir  or cfg["data"]["raw_dir"]
    output_dir = args.output_dir or cfg["data"]["processed_dir"]
    a = cfg.get("audio", {})

    audio_config = AudioConfig(
        sample_rate  = a.get("sample_rate",  22050),
        clip_duration= a.get("clip_duration", 5.0),
        n_mels       = a.get("n_mels",        128),
        n_fft        = a.get("n_fft",         2048),
        hop_length   = a.get("hop_length",    512),
        f_min        = a.get("f_min",         500.0),
        f_max        = a.get("f_max",         15000.0),
        top_db       = a.get("top_db",        80.0),
        img_size     = tuple(a.get("img_size", [224, 224])),
        clip_overlap = a.get("clip_overlap",  0.0),
    )

    converter = SpectrogramConverter(output_dir=output_dir, config=audio_config)
    summary   = converter.process_all(input_dir=input_dir, overwrite=args.overwrite)

    total = sum(summary.values())
    print(f"\nPreprocessing complete: {total} clips across {len(summary)} species.")


if __name__ == "__main__":
    main()
