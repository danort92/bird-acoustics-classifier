"""scripts/train.py — CLI entry point for model training.

Usage:
    python scripts/train.py
    python scripts/train.py --config config/default.yaml
    python scripts/train.py --epochs 50 --batch-size 64 --lr 0.0005

Note: requires src/training/ to be implemented (coming in a future milestone).
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the bird species classifier",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config", default="config/default.yaml", help="Path to YAML config file"
    )
    parser.add_argument("--epochs",     type=int,   help="Override number of training epochs")
    parser.add_argument("--batch-size", type=int,   help="Override batch size")
    parser.add_argument("--lr",         type=float, help="Override learning rate")
    parser.add_argument("--output-dir",             help="Override training output directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()  # noqa: F841
    # TODO: implement when src/training/ is ready
    raise NotImplementedError(
        "Training script not yet implemented — coming in the next milestone.\n"
        "See src/training/ for the planned implementation."
    )


if __name__ == "__main__":
    main()
