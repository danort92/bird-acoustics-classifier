"""scripts/infer.py — CLI entry point for inference on new audio files.

Converts an .mp3 file into mel-spectrogram clips (same pipeline as
preprocessing), then runs the trained model on each clip and returns
the aggregated top-k species predictions.

Usage:
    python scripts/infer.py --input audio.mp3 --checkpoint models/best_model.pt
    python scripts/infer.py --input audio.mp3 --checkpoint models/best_model.pt --top-k 5
    python scripts/infer.py --input audio.mp3 --checkpoint models/best_model.pt --output          # saves to outputs/<stem>_predictions.json
    python scripts/infer.py --input audio.mp3 --checkpoint models/best_model.pt --output custom/pred.json
"""

import argparse
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import yaml
from PIL import Image
from torchvision import transforms

from src.model import build_model, load_model, get_transforms
from src.preprocessing import AudioConfig, SpectrogramConverter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict bird species from an audio file",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config",     default="config/default.yaml", help="Path to YAML config file")
    parser.add_argument("--input",      required=True,                 help="Path to input .mp3 file")
    parser.add_argument("--checkpoint", required=True,                 help="Path to model checkpoint (.pt)")
    parser.add_argument("--top-k",      type=int, default=3,           help="Return top-k predictions per clip")
    parser.add_argument("--device",     default="cpu",                 help="Inference device (cpu / cuda)")
    parser.add_argument("--output", nargs="?", const="",
                        help="Save predictions as JSON (default path: outputs/<stem>_predictions.json)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    a = cfg.get("audio", {})
    audio_config = AudioConfig(
        sample_rate   = a.get("sample_rate",  22050),
        clip_duration = a.get("clip_duration", 5.0),
        n_mels        = a.get("n_mels",        128),
        n_fft         = a.get("n_fft",         2048),
        hop_length    = a.get("hop_length",    512),
        f_min         = a.get("f_min",         500.0),
        f_max         = a.get("f_max",         15000.0),
        top_db        = a.get("top_db",        80.0),
        img_size      = tuple(a.get("img_size", [224, 224])),
        clip_overlap  = a.get("clip_overlap",  0.0),
    )

    # Load model
    model, classes, img_size = load_model(args.checkpoint, device=args.device)
    _, val_tf = get_transforms(img_size, augment=False)

    # Convert .mp3 → spectrogram PNGs in a temp directory
    with tempfile.TemporaryDirectory() as tmpdir:
        converter = SpectrogramConverter(output_dir=tmpdir, config=audio_config)

        mp3_path = Path(args.input)
        n_clips  = converter._convert_file(
            mp3_path,
            out_dir=Path(tmpdir),
            overwrite=True,
        )

        png_files = sorted(Path(tmpdir).glob("*.png"))
        if not png_files:
            print(f"No clips extracted from {mp3_path}. File may be too short or corrupted.")
            sys.exit(1)

        print(f"File  : {mp3_path.name}")
        print(f"Clips : {len(png_files)} × {audio_config.clip_duration}s")
        print()

        # Run inference on every clip
        device     = torch.device(args.device)
        clip_preds = []   # list of (clip_idx, [(class, prob), ...])

        with torch.no_grad():
            for i, png in enumerate(png_files):
                img    = Image.open(png).convert("RGB")
                tensor = val_tf(img).unsqueeze(0).to(device)
                probs  = torch.softmax(model(tensor), dim=1)[0].cpu().numpy()
                top_idx = np.argsort(probs)[::-1][: args.top_k]
                clip_preds.append([(classes[j], float(probs[j])) for j in top_idx])
                print(f"  clip {i+1:02d}: " + "  ".join(f"{classes[j]} {probs[j]:.2f}" for j in top_idx))

        # Aggregate: average probabilities across all clips
        n_classes = len(classes)
        avg_probs = np.zeros(n_classes)
        for preds in clip_preds:
            for cls_name, prob in preds:
                avg_probs[classes.index(cls_name)] += prob
        avg_probs /= len(clip_preds)

        top_idx = np.argsort(avg_probs)[::-1][: args.top_k]
        print(f"\nAggregated top-{args.top_k} predictions:")
        for rank, idx in enumerate(top_idx, 1):
            print(f"  {rank}. {classes[idx]:<35s} {avg_probs[idx]:.3f}")

        if args.output is not None:
            stem = Path(args.input).stem
            out_path = Path(args.output) if args.output else Path("outputs") / f"{stem}_predictions.json"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "file": str(mp3_path),
                "n_clips": len(png_files),
                "top_k": args.top_k,
                "aggregated": [
                    {"rank": rank, "species": classes[i], "probability": round(float(avg_probs[i]), 4)}
                    for rank, i in enumerate(top_idx, 1)
                ],
                "per_clip": [
                    {"clip": ci + 1, "predictions": [{"species": s, "probability": round(p, 4)} for s, p in preds]}
                    for ci, preds in enumerate(clip_preds)
                ],
            }
            out_path.write_text(json.dumps(payload, indent=2))
            print(f"\nPredictions saved to: {out_path}")


if __name__ == "__main__":
    main()
