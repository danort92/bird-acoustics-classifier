"""scripts/evaluate.py — CLI entry point for model evaluation.

Loads a saved checkpoint, runs inference on the test split, and prints
a full classification report. Optionally saves a confusion-matrix PNG.

Usage:
    python scripts/evaluate.py --checkpoint models/best_model.pt
    python scripts/evaluate.py --checkpoint models/best_model.pt --save-cm reports/cm.png
    python scripts/evaluate.py --checkpoint models/best_model.pt --species "Turdus torquatus" "Cinclus cinclus"
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

from src.model import BirdTrainer, TrainingConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained bird species classifier",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config",     default="config/default.yaml", help="Path to YAML config file")
    parser.add_argument("--checkpoint", required=True,                 help="Path to model checkpoint (.pt)")
    parser.add_argument("--data-dir",                                  help="Override processed data directory")
    parser.add_argument("--species",    nargs="+",                     help="Restrict to these species")
    parser.add_argument("--save-cm",                                   help="Save confusion-matrix PNG to this path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cfg = TrainingConfig.from_yaml(args.config)
    if args.data_dir:
        cfg.processed_dir = args.data_dir

    trainer        = BirdTrainer(cfg)
    y_true, y_pred = trainer.evaluate(args.checkpoint, species=args.species)

    print("\nClassification report:")
    print(classification_report(y_true, y_pred, target_names=trainer.classes, digits=3))
    acc = (y_true == y_pred).mean()
    print(f"Overall accuracy: {acc:.3f}")

    if args.save_cm:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        cm   = confusion_matrix(y_true, y_pred)
        cm_n = cm.astype("float") / cm.sum(axis=1, keepdims=True)
        n    = len(trainer.classes)
        short = [c.replace("_", "\n") for c in trainer.classes]

        fig, ax = plt.subplots(figsize=(max(10, n * 0.65), max(8, n * 0.55)))
        im = ax.imshow(cm_n, cmap="Blues", vmin=0, vmax=1)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xticks(range(n)); ax.set_yticks(range(n))
        ax.set_xticklabels(short, fontsize=7, rotation=90)
        ax.set_yticklabels(short, fontsize=7)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ax.set_title(f"Confusion matrix — accuracy={acc:.3f}")
        for i in range(n):
            for j in range(n):
                v = cm_n[i, j]
                if v > 0.01:
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                            fontsize=6, color="white" if v > 0.5 else "black")
        fig.tight_layout()

        out = Path(args.save_cm)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"Confusion matrix saved to: {out}")


if __name__ == "__main__":
    main()
