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
import zipfile
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


def _expand_files(files: list, tmpdir: Path) -> list[str]:
    """Expand a mixed list of audio files and ZIPs into a flat list of audio paths.

    ZIP contents are extracted under *tmpdir*.
    """
    audio_suffixes = {".mp3", ".wav"}
    result: list[str] = []
    for f in files:
        path = Path(_resolve_path(f))
        if path.suffix.lower() == ".zip":
            with zipfile.ZipFile(path) as zf:
                for member in zf.namelist():
                    if Path(member).suffix.lower() in audio_suffixes:
                        extracted = zf.extract(member, path=tmpdir)
                        result.append(extracted)
        elif path.suffix.lower() in audio_suffixes:
            result.append(str(path))
    return result


def _infer_file(
    audio_path: str,
    model,
    classes: list[str],
    audio_cfg: AudioConfig,
    val_tf,
    device: torch.device,
) -> tuple[Image.Image | None, str, float]:
    """Run inference on a single audio file.

    Returns (first_spectrogram, species_name, confidence).
    Raises RuntimeError if the audio cannot be loaded or produces no clips.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        converter   = SpectrogramConverter(output_dir=tmpdir, config=audio_cfg)

        try:
            png_paths = converter.convert_file(Path(audio_path), tmpdir_path, overwrite=True)
        except Exception as exc:
            raise RuntimeError(f"Failed to convert audio '{Path(audio_path).name}': {exc}") from exc

        if not png_paths:
            raise RuntimeError(
                f"No clips generated for '{Path(audio_path).name}'. "
                "The file may be unreadable, too short, or require ffmpeg for MP3 decoding."
            )

        first_spec = Image.open(png_paths[0]).convert("RGB")

        all_probs: list[np.ndarray] = []
        with torch.no_grad():
            for png in png_paths:
                img    = Image.open(png).convert("RGB")
                tensor = val_tf(img).unsqueeze(0).to(device)
                probs  = torch.softmax(model(tensor), dim=1)[0].cpu().numpy()
                all_probs.append(probs)

        avg_probs = np.mean(all_probs, axis=0)
        top_idx   = int(np.argmax(avg_probs))
        return first_spec, _fmt_class(classes[top_idx]), float(avg_probs[top_idx])


# ---------------------------------------------------------------------------
# Main inference function (multi-file)
# ---------------------------------------------------------------------------

def classify_files(
    files,
    checkpoint: str,
) -> tuple[list, pd.DataFrame, str]:
    """Process one or more audio files and return aggregated results.

    Returns
    -------
    gallery : list of (PIL Image, caption) for gr.Gallery
    df      : summary DataFrame for gr.Dataframe
    status  : status string
    """
    empty_df = pd.DataFrame(columns=["File", "Species", "Confidence"])

    if not files:
        return [], empty_df, "Upload one or more .mp3 / .wav files (or a .zip), then click Classify."

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
        errors: list[str] = []

        # Keep zip_tmpdir alive for the entire inference loop so that files
        # extracted from ZIP archives are not deleted before they are read.
        with tempfile.TemporaryDirectory() as zip_tmpdir:
            expanded = _expand_files(files, Path(zip_tmpdir))

            if not expanded:
                return [], empty_df, "No .mp3 or .wav files found in the uploaded files."

            for path in expanded:
                fname = Path(path).name
                try:
                    spec, top_name, top_conf = _infer_file(
                        str(path), model, classes, audio_cfg, val_tf, device
                    )
                except RuntimeError as exc:
                    errors.append(str(exc))
                    rows.append({"File": fname, "Species": f"Error: {exc}", "Confidence": "—"})
                    continue

                caption = f"{fname}\n{top_name} ({top_conf:.1%})"
                gallery.append((spec, caption))
                rows.append({
                    "File":       fname,
                    "Species":    top_name,
                    "Confidence": f"{top_conf:.1%}",
                })

        df = pd.DataFrame(rows)
        ok_count  = len(expanded) - len(errors)
        status    = (
            f"Processed {ok_count}/{len(expanded)} file(s)  ·  "
            f"model: {checkpoint_path.name}  ·  "
            f"clip duration: {audio_cfg.clip_duration}s"
        )
        if errors:
            status += f"\n⚠️  {len(errors)} file(s) failed — check that ffmpeg is installed for MP3 support."
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
                    label="Audio files (.mp3 / .wav) or .zip archive",
                    file_count="multiple",
                    file_types=[".mp3", ".wav", ".zip"],
                )
                with gr.Accordion("⚙️ Settings", open=False):
                    checkpoint_input = gr.Textbox(
                        label="Checkpoint path",
                        value=checkpoint,
                        placeholder="models/best_model.pt",
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
                            headers=["File", "Species", "Confidence"],
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
            inputs=[file_input, checkpoint_input],
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
        inbrowser=True,
        max_file_size="200mb",
        theme=gr.themes.Soft(),
        css=CSS,
    )
