"""app/app.py — Gradio web interface for bird species classification.

Accepts one or more audio recordings (.mp3 or .wav), converts them into
mel-spectrogram clips, runs the trained EfficientNet-B0 model, and returns
the top-k species predictions with confidence scores.

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
import pandas as pd
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
# Custom CSS
# ---------------------------------------------------------------------------

CSS = """
/* global */
.gradio-container { max-width: 1200px !important; margin: 0 auto; }

/* header */
.app-header { text-align: center; padding: 20px 0 4px; }
.app-header h1 { font-size: 2rem; font-weight: 700; color: #1b5e20; margin-bottom: 4px; }
.app-header p  { color: #555; font-size: 0.95rem; }

/* species pills */
.species-pills {
    display: flex; flex-wrap: wrap; gap: 5px;
    justify-content: center; margin: 10px 0 16px;
}
.species-pill {
    background: #e8f5e9; border: 1px solid #a5d6a7; border-radius: 12px;
    padding: 2px 9px; font-size: 0.75rem; color: #2e7d32; white-space: nowrap;
}

/* run button */
#run-btn { background: #2e7d32 !important; font-weight: 600; }
#run-btn:hover { background: #1b5e20 !important; }

/* footer */
.app-footer { text-align: center; color: #aaa; font-size: 0.78rem; margin-top: 12px; }
.app-footer a { color: #81c784; }
"""

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


def _resolve_path(f) -> str:
    """Extract a file path string from whatever Gradio hands us."""
    if isinstance(f, str):
        return f
    if isinstance(f, dict):
        return f["name"]
    return f.name  # UploadedFile-like object


def _infer_file(
    audio_path: str,
    model,
    classes: list[str],
    audio_cfg: AudioConfig,
    val_tf,
    device: torch.device,
    top_k: int,
) -> tuple[Image.Image | None, dict[str, float], int]:
    """Run inference on a single audio file.

    Returns (first_spectrogram, {species: confidence}, n_clips).
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        converter   = SpectrogramConverter(output_dir=tmpdir, config=audio_cfg)
        png_paths   = converter.convert_file(Path(audio_path), tmpdir_path, overwrite=True)

        if not png_paths:
            return None, {}, 0

        first_spec = Image.open(png_paths[0]).convert("RGB")

        all_probs: list[np.ndarray] = []
        with torch.no_grad():
            for png in png_paths:
                img    = Image.open(png).convert("RGB")
                tensor = val_tf(img).unsqueeze(0).to(device)
                probs  = torch.softmax(model(tensor), dim=1)[0].cpu().numpy()
                all_probs.append(probs)

    avg_probs    = np.mean(all_probs, axis=0)
    top_k_capped = min(top_k, len(classes))
    top_idx      = np.argsort(avg_probs)[::-1][:top_k_capped]
    predictions  = {_fmt_class(classes[i]): float(avg_probs[i]) for i in top_idx}

    return first_spec, predictions, len(png_paths)


# ---------------------------------------------------------------------------
# Main inference function (multi-file)
# ---------------------------------------------------------------------------

def classify_files(
    files,
    checkpoint: str,
    top_k: int,
) -> tuple[list, pd.DataFrame, str]:
    """Process one or more audio files and return aggregated results.

    Returns
    -------
    gallery : list of (PIL Image, caption) for gr.Gallery
    df      : summary DataFrame for gr.Dataframe
    status  : status string
    """
    empty_df = pd.DataFrame(columns=["File", "Top species", "Confidence", "Clips"])

    if not files:
        return [], empty_df, "Upload one or more .mp3 / .wav files, then click Classify."

    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.exists():
        return (
            [],
            empty_df,
            f"Checkpoint not found: {checkpoint}\n"
            "Train the model first or point to an existing .pt file.",
        )

    try:
        audio_cfg = _load_audio_config()
        model, classes, img_size = load_model(str(checkpoint_path), device="cpu")
        _, val_tf = get_transforms(img_size)
        device    = torch.device("cpu")

        if not isinstance(files, list):
            files = [files]

        gallery: list[tuple[Image.Image, str]] = []
        rows: list[dict] = []

        for f in files:
            path  = _resolve_path(f)
            fname = Path(path).name
            spec, preds, n_clips = _infer_file(
                path, model, classes, audio_cfg, val_tf, device, top_k
            )

            if n_clips == 0:
                rows.append({
                    "File": fname, "Top species": "—",
                    "Confidence": "—", "Clips": 0,
                })
                continue

            top_name, top_conf = next(iter(preds.items()))
            caption = f"{fname}\n{top_name} ({top_conf:.1%})"
            gallery.append((spec, caption))
            rows.append({
                "File":        fname,
                "Top species": top_name,
                "Confidence":  f"{top_conf:.1%}",
                "Clips":       n_clips,
            })

        df = pd.DataFrame(rows)
        status = (
            f"Processed {len(files)} file(s)  ·  "
            f"model: {checkpoint_path.name}  ·  "
            f"clip duration: {audio_cfg.clip_duration}s"
        )
        return gallery, df, status

    except Exception as exc:  # noqa: BLE001
        return [], empty_df, f"Error: {exc}"


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

_SPECIES = [
    "Ring ouzel", "Black redstart", "Alpine accentor", "Yellow-billed chough",
    "Red-billed chough", "Wallcreeper", "Water pipit", "White-winged snowfinch",
    "Rock ptarmigan", "Black woodpecker", "Western capercaillie", "Three-toed woodpecker",
    "Common crossbill", "Spotted nutcracker", "Firecrest", "White-throated dipper",
    "Collared flycatcher", "Whinchat", "Rock bunting", "Bearded vulture",
]

_PILLS_HTML = (
    '<div class="species-pills">'
    + "".join(f'<span class="species-pill">{s}</span>' for s in _SPECIES)
    + "</div>"
)


def build_ui(checkpoint: str = DEFAULT_CHECKPOINT) -> gr.Blocks:
    with gr.Blocks(title="Bird Acoustics Classifier") as demo:

        # ── Header ──────────────────────────────────────────────────────────
        gr.HTML("""
        <div class="app-header">
            <h1>🐦 Bird Acoustics Classifier</h1>
            <p>Alpine species identification from audio recordings &nbsp;·&nbsp;
               EfficientNet-B0 &nbsp;·&nbsp; Xeno-canto</p>
        </div>
        """)
        gr.HTML(
            '<div style="text-align:center">'
            '<span style="font-size:0.82rem;color:#555;font-weight:600">'
            '20 Alpine species (Italy / Austria / Switzerland):</span>'
            f"{_PILLS_HTML}</div>"
        )
        gr.HTML('<hr style="border:none;border-top:1px solid #e0e0e0;margin:4px 0 12px">')

        # ── Main layout ──────────────────────────────────────────────────────
        with gr.Row(equal_height=False):

            # Left panel ─ upload + controls
            with gr.Column(scale=1, min_width=260):
                gr.Markdown("### Upload")
                file_input = gr.File(
                    label="Audio files (.mp3 / .wav)",
                    file_count="multiple",
                    file_types=[".mp3", ".wav"],
                )
                with gr.Accordion("⚙️ Settings", open=False):
                    checkpoint_input = gr.Textbox(
                        label="Checkpoint path",
                        value=checkpoint,
                        placeholder="models/best_model.pt",
                    )
                    top_k_slider = gr.Slider(
                        minimum=1, maximum=10, step=1,
                        value=DEFAULT_TOP_K, label="Top-k species",
                    )
                run_btn = gr.Button("🔍  Classify", variant="primary", elem_id="run-btn")
                status_output = gr.Textbox(
                    label="Status", interactive=False, lines=2, max_lines=3,
                )

            # Right panel ─ results
            with gr.Column(scale=3):
                with gr.Tabs():
                    with gr.Tab("📊 Summary"):
                        table_output = gr.Dataframe(
                            headers=["File", "Top species", "Confidence", "Clips"],
                            label=None,
                            interactive=False,
                            wrap=True,
                        )
                    with gr.Tab("🖼️ Spectrograms"):
                        gallery_output = gr.Gallery(
                            label="Mel spectrograms — first clip per file",
                            columns=3,
                            height="auto",
                            object_fit="contain",
                        )

        # ── Footer ──────────────────────────────────────────────────────────
        gr.HTML(
            '<p class="app-footer">'
            'Audio data from <a href="https://xeno-canto.org" target="_blank">Xeno-canto</a> &nbsp;·&nbsp;'
            'Alpine zone &nbsp;·&nbsp; Italy / Austria / Switzerland'
            '</p>'
        )

        run_btn.click(
            fn=classify_files,
            inputs=[file_input, checkpoint_input, top_k_slider],
            outputs=[gallery_output, table_output, status_output],
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
        "--port", type=int, default=None,
        help="Port to run the server on (default: auto)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    demo = build_ui(checkpoint=args.checkpoint)
    demo.launch(
        share=args.share,
        server_port=args.port,
        max_file_size="200mb",
        theme=gr.themes.Soft(),
        css=CSS,
    )
