"""app/app.py — Gradio web interface for bird species classification.

Accepts an audio recording (.mp3 or .wav), converts it into mel-spectrogram
clips, runs the trained EfficientNet-B0 model, and returns the top-k species
predictions with confidence scores.

Usage:
    python app/app.py
    python app/app.py --checkpoint models/best_model.pt --share
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import gradio as gr
import numpy as np
import torch
import yaml
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.model import get_transforms, load_model
from src.preprocessing import AudioConfig, SpectrogramConverter

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_CHECKPOINT = "models/best_model.pt"
DEFAULT_CONFIG     = "config/default.yaml"
DEFAULT_TOP_K      = 5

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_audio_config(config_path: str = DEFAULT_CONFIG) -> AudioConfig:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    a = cfg.get("audio", {})
    return AudioConfig(
        sample_rate   = a.get("sample_rate",   22050),
        clip_duration = a.get("clip_duration",  5.0),
        n_mels        = a.get("n_mels",         128),
        n_fft         = a.get("n_fft",          2048),
        hop_length    = a.get("hop_length",     512),
        f_min         = a.get("f_min",          500.0),
        f_max         = a.get("f_max",          15000.0),
        top_db        = a.get("top_db",         80.0),
        img_size      = tuple(a.get("img_size", [224, 224])),
        clip_overlap  = a.get("clip_overlap",   0.0),
    )


def _fmt_class(name: str) -> str:
    """Convert directory name to display name: 'turdus_torquatus' → 'Turdus Torquatus'."""
    return name.replace("_", " ").title()


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def classify_audio(
    audio_path: str | None,
    checkpoint: str,
    top_k: int,
) -> tuple[Image.Image | None, dict, str]:
    """Run species classification on an uploaded audio file.

    Returns
    -------
    spectrogram : PIL Image of the first clip, or None
    predictions : {species_name: confidence} for gr.Label
    status      : informational string
    """
    if not audio_path:
        return None, {}, "Please upload an audio file."

    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.exists():
        return (
            None,
            {},
            f"Checkpoint not found: {checkpoint}\n"
            "Train the model first (Task A) or point to an existing .pt file.",
        )

    try:
        audio_cfg = _load_audio_config()
        model, classes, img_size = load_model(str(checkpoint_path), device="cpu")
        _, val_tf = get_transforms(img_size)
        device = torch.device("cpu")

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            converter   = SpectrogramConverter(output_dir=tmpdir, config=audio_cfg)
            png_paths   = converter.convert_file(
                Path(audio_path), tmpdir_path, overwrite=True
            )

            if not png_paths:
                return (
                    None,
                    {},
                    "No clips extracted. The file may be too short or corrupted.",
                )

            # Display spectrogram of first clip
            first_spec = Image.open(png_paths[0]).convert("RGB")

            # Inference on every clip
            all_probs: list[np.ndarray] = []
            with torch.no_grad():
                for png in png_paths:
                    img    = Image.open(png).convert("RGB")
                    tensor = val_tf(img).unsqueeze(0).to(device)
                    probs  = torch.softmax(model(tensor), dim=1)[0].cpu().numpy()
                    all_probs.append(probs)

            # Aggregate: mean probability across all clips
            avg_probs = np.mean(all_probs, axis=0)
            top_k_capped = min(top_k, len(classes))
            top_idx  = np.argsort(avg_probs)[::-1][:top_k_capped]

            predictions = {_fmt_class(classes[i]): float(avg_probs[i]) for i in top_idx}
            status = (
                f"{len(png_paths)} clip(s) × {audio_cfg.clip_duration}s | "
                f"model: {checkpoint_path.name}"
            )

        return first_spec, predictions, status

    except Exception as exc:  # noqa: BLE001
        return None, {}, f"Error: {exc}"


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

_SPECIES_LIST = (
    "Ring ouzel, Black redstart, Alpine accentor, Yellow-billed chough, "
    "Red-billed chough, Wallcreeper, Water pipit, White-winged snowfinch, "
    "Rock ptarmigan, Black woodpecker, Western capercaillie, Three-toed woodpecker, "
    "Common crossbill, Spotted nutcracker, Firecrest, White-throated dipper, "
    "Collared flycatcher, Whinchat, Rock bunting, Bearded vulture"
)


def build_ui(checkpoint: str = DEFAULT_CHECKPOINT) -> gr.Blocks:
    with gr.Blocks(title="Bird Acoustics Classifier", theme=gr.themes.Soft()) as demo:

        gr.Markdown(
            f"""
# Bird Acoustics Classifier
**Alpine species identification from audio recordings**

Upload an `.mp3` or `.wav` recording. The model slices it into 5-second clips,
computes mel-spectrograms, and returns the most likely species.

> **20 Alpine species**: {_SPECIES_LIST}
"""
        )

        with gr.Row():
            with gr.Column(scale=1):
                audio_input = gr.Audio(
                    label="Audio recording",
                    type="filepath",
                    sources=["upload"],
                )
                checkpoint_input = gr.Textbox(
                    label="Checkpoint path",
                    value=checkpoint,
                    placeholder="models/best_model.pt",
                )
                top_k_slider = gr.Slider(
                    minimum=1,
                    maximum=10,
                    step=1,
                    value=DEFAULT_TOP_K,
                    label="Top-k species",
                )
                run_btn = gr.Button("Classify", variant="primary")

            with gr.Column(scale=2):
                spec_output   = gr.Image(label="Mel spectrogram (clip 1)", type="pil")
                label_output  = gr.Label(label="Species predictions", num_top_classes=10)
                status_output = gr.Textbox(label="Status", interactive=False, lines=2)

        run_btn.click(
            fn=classify_audio,
            inputs=[audio_input, checkpoint_input, top_k_slider],
            outputs=[spec_output, label_output, status_output],
        )

    return demo


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bird Acoustics Classifier — Gradio app",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint", default=DEFAULT_CHECKPOINT,
        help="Path to trained model checkpoint (.pt)",
    )
    parser.add_argument(
        "--share", action="store_true",
        help="Create a public Gradio share link",
    )
    parser.add_argument(
        "--port", type=int, default=7860,
        help="Port to run the server on",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    demo = build_ui(checkpoint=args.checkpoint)
    demo.launch(share=args.share, server_port=args.port)
