"""scripts/infer.py — CLI entry point for inference on new audio files.

Usage:
    python scripts/infer.py --input audio.mp3 --checkpoint models/best.pt
    python scripts/infer.py --input audio.mp3 --checkpoint models/best.pt --top-k 5

Note: requires src/inference/ to be implemented (coming in a future milestone).
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run bird species inference on an audio file",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config", default="config/default.yaml", help="Path to YAML config file"
    )
    parser.add_argument(
        "--input", required=True, help="Path to input .mp3 file"
    )
    parser.add_argument(
        "--checkpoint", required=True, help="Path to model checkpoint (.pt)"
    )
    parser.add_argument(
        "--top-k", type=int, default=3, help="Return top-k predictions"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()  # noqa: F841
    # TODO: implement when src/inference/ is ready
    raise NotImplementedError(
        "Infer script not yet implemented — coming in the next milestone.\n"
        "See src/inference/ for the planned implementation."
    )


if __name__ == "__main__":
    main()
