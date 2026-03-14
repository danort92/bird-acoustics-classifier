"""scripts/train.py — CLI entry point for model training.

Usage:
    python scripts/train.py
    python scripts/train.py --config config/default.yaml
    python scripts/train.py --epochs 50 --batch-size 64 --lr 0.0005
    python scripts/train.py --species "Turdus torquatus" "Cinclus cinclus"
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.model import BirdTrainer, TrainingConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the bird species classifier (EfficientNet-B0)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config",      default="config/default.yaml", help="Path to YAML config file")
    parser.add_argument("--epochs",      type=int,   help="Override number of training epochs")
    parser.add_argument("--batch-size",  type=int,   help="Override batch size")
    parser.add_argument("--lr",          type=float, help="Override learning rate")
    parser.add_argument("--output-dir",              help="Override model output directory")
    parser.add_argument("--species",     nargs="+",  help="Restrict to these species (scientific names)")
    parser.add_argument("--no-scheduler", action="store_true", help="Disable cosine LR scheduler")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cfg = TrainingConfig.from_yaml(args.config)

    if args.epochs:
        cfg.epochs = args.epochs
    if args.batch_size:
        cfg.batch_size = args.batch_size
    if args.lr:
        cfg.learning_rate = args.lr
    if args.output_dir:
        cfg.output_dir = args.output_dir
    if args.no_scheduler:
        cfg.use_scheduler = False

    print(f"Training config:")
    print(f"  model        : {cfg.model_name}")
    print(f"  epochs       : {cfg.epochs}  (patience={cfg.patience})")
    print(f"  batch_size   : {cfg.batch_size}")
    print(f"  learning_rate: {cfg.learning_rate}")
    print(f"  output_dir   : {cfg.output_dir}")
    print(f"  mlflow       : {cfg.tracking_uri}  [{cfg.experiment_name}]")
    if args.species:
        print(f"  species      : {args.species}")

    trainer   = BirdTrainer(cfg)
    best_path, history = trainer.train(species=args.species)

    print(f"\nTraining complete.")
    print(f"  Best model : {best_path}")
    print(f"  Test acc   : {history['test_acc']:.3f}")
    print(f"  Test loss  : {history['test_loss']:.4f}")


if __name__ == "__main__":
    main()
