"""app/app.py — Gradio web interface for bird species classification.

Accepts one or more audio recordings (.mp3 or .wav), converts them into
mel-spectrogram clips, runs the trained EfficientNet-B0 model, and returns
top-k species predictions with confidence scores.

Usage:
    python app/app.py
    python app/app.py --checkpoint models/best_model.pt --share
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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


def _discover_checkpoints(default: str = DEFAULT_CHECKPOINT) -> tuple[list[str], str]:
    """Return (choices, best_default) scanning models/ for .pt files."""
    found = sorted(str(p) for p in Path("models").glob("*.pt")) if Path("models").exists() else []
    if not found:
        found = [default]
    best = default if default in found else found[0]
    return found, best
DEFAULT_CONFIG     = "config/default.yaml"
TOP_K              = 5

# ---------------------------------------------------------------------------
# Species metadata
# ---------------------------------------------------------------------------

SPECIES_INFO: dict[str, dict] = {
    "Turdus Torquatus":        {"common": "Ring ouzel",              "habitat": "Rocky slopes, high-altitude forests",     "xc_query": "Turdus+torquatus"},
    "Phoenicurus Ochruros":    {"common": "Black redstart",          "habitat": "Rocky terrain, mountain villages",        "xc_query": "Phoenicurus+ochruros"},
    "Prunella Collaris":       {"common": "Alpine accentor",         "habitat": "High rocky areas above treeline",         "xc_query": "Prunella+collaris"},
    "Pyrrhocorax Graculus":    {"common": "Yellow-billed chough",    "habitat": "Alpine cliffs and glaciers",              "xc_query": "Pyrrhocorax+graculus"},
    "Pyrrhocorax Pyrrhocorax": {"common": "Red-billed chough",       "habitat": "Alpine meadows and cliffs",               "xc_query": "Pyrrhocorax+pyrrhocorax"},
    "Tichodroma Muraria":      {"common": "Wallcreeper",             "habitat": "Vertical rock faces",                     "xc_query": "Tichodroma+muraria"},
    "Anthus Spinoletta":       {"common": "Water pipit",             "habitat": "Alpine meadows and streams",              "xc_query": "Anthus+spinoletta"},
    "Montifringilla Nivalis":  {"common": "White-winged snowfinch",  "habitat": "Above treeline, snowfields",              "xc_query": "Montifringilla+nivalis"},
    "Lagopus Muta":            {"common": "Rock ptarmigan",          "habitat": "High alpine tundra",                      "xc_query": "Lagopus+muta"},
    "Dryocopus Martius":       {"common": "Black woodpecker",        "habitat": "Subalpine conifer forests",               "xc_query": "Dryocopus+martius"},
    "Tetrao Urogallus":        {"common": "Western capercaillie",    "habitat": "Old-growth conifer forests",              "xc_query": "Tetrao+urogallus"},
    "Picoides Tridactylus":    {"common": "Three-toed woodpecker",   "habitat": "Spruce forests",                          "xc_query": "Picoides+tridactylus"},
    "Loxia Curvirostra":       {"common": "Common crossbill",        "habitat": "Conifer forests",                         "xc_query": "Loxia+curvirostra"},
    "Nucifraga Caryocatactes": {"common": "Spotted nutcracker",      "habitat": "Mountain conifer forests",                "xc_query": "Nucifraga+caryocatactes"},
    "Regulus Ignicapilla":     {"common": "Firecrest",               "habitat": "Mixed mountain forests",                  "xc_query": "Regulus+ignicapilla"},
    "Cinclus Cinclus":         {"common": "White-throated dipper",   "habitat": "Alpine streams and torrents",             "xc_query": "Cinclus+cinclus"},
    "Ficedula Albicollis":     {"common": "Collared flycatcher",     "habitat": "Deciduous mountain forests",              "xc_query": "Ficedula+albicollis"},
    "Saxicola Rubetra":        {"common": "Whinchat",                "habitat": "Subalpine meadows",                       "xc_query": "Saxicola+rubetra"},
    "Emberiza Cia":            {"common": "Rock bunting",            "habitat": "Rocky slopes with sparse vegetation",     "xc_query": "Emberiza+cia"},
    "Gypaetus Barbatus":       {"common": "Bearded vulture",         "habitat": "High alpine cliffs (reintroduced)",       "xc_query": "Gypaetus+barbatus"},
}

# ---------------------------------------------------------------------------
# Custom CSS & theme
# ---------------------------------------------------------------------------

CSS = """
/* ── Layout ──────────────────────────────────────────────────────────── */
.gradio-container { max-width: 1280px !important; margin: 0 auto; }

/* ── Header ──────────────────────────────────────────────────────────── */
.app-header { text-align: center; padding: 24px 0 6px; }
.app-header h1 {
    font-size: 2.2rem; font-weight: 800;
    background: linear-gradient(135deg, #1b5e20 0%, #388e3c 60%, #66bb6a 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin-bottom: 6px;
}
.app-header p { color: #666; font-size: 0.95rem; letter-spacing: 0.02em; }

/* ── Species pills ────────────────────────────────────────────────────── */
.species-pills {
    display: flex; flex-wrap: wrap; gap: 5px;
    justify-content: center; margin: 10px 0 18px;
}
.species-pill {
    background: #e8f5e9; border: 1px solid #a5d6a7; border-radius: 12px;
    padding: 3px 10px; font-size: 0.73rem; color: #2e7d32;
}

/* ── Run button ───────────────────────────────────────────────────────── */
#run-btn { font-weight: 700; letter-spacing: 0.03em; }

/* ── Species info card ────────────────────────────────────────────────── */
.species-card {
    background: linear-gradient(135deg, #f1f8e9, #e8f5e9);
    border: 1px solid #a5d6a7; border-radius: 12px;
    padding: 16px 18px; height: 100%; box-sizing: border-box;
}
.species-card h2  { font-size: 1.2rem; font-weight: 700; color: #1b5e20; margin: 0 0 2px; }
.species-card .sci { font-style: italic; color: #388e3c; font-size: 0.87rem; margin: 0 0 12px; }
.species-card .row { display: flex; gap: 8px; margin-bottom: 6px; align-items: flex-start; }
.species-card .lbl { font-weight: 600; color: #555; min-width: 76px; font-size: 0.83rem; }
.species-card .val { color: #333; font-size: 0.83rem; }
.species-card .conf-bar { height: 8px; border-radius: 4px; margin-top: 10px; background: #c8e6c9; overflow: hidden; }
.species-card .conf-fill { height: 100%; border-radius: 4px; background: linear-gradient(90deg, #43a047, #1b5e20); }
.species-card a { color: #2e7d32; font-size: 0.82rem; text-decoration: none; display: inline-block; margin-top: 12px; }
.species-card a:hover { text-decoration: underline; }

/* ── Footer ───────────────────────────────────────────────────────────── */
.app-footer { text-align: center; color: #aaa; font-size: 0.78rem; margin-top: 16px; padding-bottom: 8px; }
.app-footer a { color: #81c784; }
"""

THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.green,
    neutral_hue=gr.themes.colors.gray,
)

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
    """'turdus_torquatus' → 'Turdus Torquatus'."""
    return name.replace("_", " ").title()


def _resolve_path(f) -> str:
    if isinstance(f, str):
        return f
    if isinstance(f, dict):
        # Gradio 4.x uses "path"; older versions used "name"
        return f.get("path") or f.get("name") or next(iter(f.values()))
    # Object with .path (Gradio 4.x FileData) or .name
    if hasattr(f, "path"):
        return f.path
    return f.name


def _expand_files(files: list, tmpdir: Path) -> list[str]:
    """Expand audio files and ZIPs into a flat list of audio paths."""
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


def _conf_badge(conf: float) -> str:
    if conf >= 0.80:
        return "🟢"
    if conf >= 0.40:
        return "🟡"
    return "🔴"


def _infer_file(
    audio_path: str,
    model,
    classes: list[str],
    audio_cfg: AudioConfig,
    val_tf,
    device: torch.device,
    top_k: int = TOP_K,
) -> tuple[Image.Image | None, list[str], list[float]]:
    """Run inference on a single audio file.

    Returns (first_spectrogram, top_k_species_names, top_k_probs).
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

        avg_probs    = np.mean(all_probs, axis=0)
        top_k_actual = min(top_k, len(classes))
        indices      = np.argsort(avg_probs)[::-1][:top_k_actual]
        top_k_names  = [_fmt_class(classes[i]) for i in indices]
        top_k_probs  = [float(avg_probs[i]) for i in indices]

        return first_spec, top_k_names, top_k_probs


def _build_species_card(species: str, confidence: float) -> str:
    info     = SPECIES_INFO.get(species, {})
    common   = info.get("common", species)
    habitat  = info.get("habitat", "—")
    xc_query = info.get("xc_query", species.replace(" ", "+"))
    xc_url   = f"https://xeno-canto.org/explore?query={xc_query}"
    fill_w   = f"{confidence * 100:.1f}%"

    return f"""
<div class="species-card">
  <h2>🐦 {common}</h2>
  <p class="sci">{species}</p>
  <div class="row">
    <span class="lbl">Habitat</span>
    <span class="val">{habitat}</span>
  </div>
  <div class="row">
    <span class="lbl">Confidence</span>
    <span class="val"><b>{confidence:.1%}</b></span>
  </div>
  <div class="conf-bar">
    <div class="conf-fill" style="width:{fill_w}"></div>
  </div>
  <a href="{xc_url}" target="_blank">🔗 Listen on Xeno-canto →</a>
</div>
"""


def _empty_bar_fig():
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.set_visible(False)
    fig.patch.set_alpha(0)
    plt.tight_layout()
    return fig


def _make_bar_fig(names: list[str], probs: list[float]):
    pct   = [p * 100 for p in probs]
    n     = len(names)
    colors = ["#1b5e20"] + ["#81c784"] * (n - 1)
    fig, ax = plt.subplots(figsize=(6, max(2, n * 0.55)))
    bars = ax.barh(names[::-1], pct[::-1], color=colors[::-1], height=0.6)
    ax.set_xlabel("Confidence (%)", fontsize=9)
    ax.set_xlim(0, 100)
    ax.set_title(f"Top-{n} predictions", fontsize=10, fontweight="bold")
    ax.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="y", labelsize=8)
    ax.tick_params(axis="x", labelsize=8)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Main inference function — generator for progressive updates (D)
# ---------------------------------------------------------------------------

def classify_files(files, checkpoint: str):
    """Process one or more audio files, yielding progressive UI updates."""
    _empty = _empty_bar_fig()
    _empty_df = pd.DataFrame(columns=["", "File", "Predicted Species", "Confidence"])

    def _bail(msg):
        yield [], _empty_df, msg, gr.Dropdown(choices=[], value=None, interactive=True, label="Select a file to inspect"), {}, None, _empty, ""

    if not files:
        yield from _bail("Upload one or more .mp3 / .wav files (or a .zip), then click Classify.")
        return

    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.exists():
        yield from _bail(
            f"Checkpoint not found: {checkpoint}\n"
            "Train the model first or point to an existing .pt file."
        )
        return

    try:
        audio_cfg = _load_audio_config()
        model, classes, img_size = load_model(str(checkpoint_path), device="cpu")
        _, val_tf = get_transforms(img_size)
        device    = torch.device("cpu")
    except Exception as exc:
        yield from _bail(f"Model load error: {exc}")
        return

    if not isinstance(files, list):
        files = [files]

    # Persistent temp dir — survives generator completion so audio playback works
    run_audio_dir = Path(tempfile.mkdtemp(prefix="bac_audio_"))

    gallery: list[tuple[Image.Image, str]] = []
    rows:    list[dict]                    = []
    state:   dict                          = {}
    errors:  list[str]                     = []

    with tempfile.TemporaryDirectory() as zip_tmpdir:
        expanded = _expand_files(files, Path(zip_tmpdir))

        if not expanded:
            yield from _bail("No .mp3 or .wav files found in the uploaded files.")
            return

        for i, path in enumerate(expanded):
            fname = Path(path).name
            yield [], _empty_df, f"⏳ [{i + 1}/{len(expanded)}] Analysing {fname} …", gr.Dropdown(choices=[], value=None, interactive=False, label="Select a file to inspect"), {}, None, _empty, ""

            # Copy to persistent dir so audio player can access it after completion
            dst = run_audio_dir / fname
            shutil.copy2(path, dst)

            try:
                spec, top_names, top_probs = _infer_file(
                    str(path), model, classes, audio_cfg, val_tf, device
                )
            except Exception as exc:
                errors.append(str(exc))
                rows.append({"": "🔴", "File": fname, "Predicted Species": "Error", "Confidence": "—"})
                continue

            badge = _conf_badge(top_probs[0])
            gallery.append((spec, f"{fname}\n{top_names[0]} ({top_probs[0]:.1%})"))
            rows.append({
                "":                  badge,
                "File":              fname,
                "Predicted Species": top_names[0],
                "Confidence":        f"{top_probs[0]:.1%}",
            })
            state[fname] = {
                "audio_path": str(dst),
                "top_names":  top_names,
                "top_probs":  top_probs,
            }

    ok_count = len(expanded) - len(errors)
    status   = (
        f"✅  Done — {ok_count}/{len(expanded)} file(s) classified  ·  "
        f"model: {checkpoint_path.name}  ·  clip: {audio_cfg.clip_duration}s"
    )
    if errors:
        status += f"\n⚠️  {len(errors)} file(s) failed — check that ffmpeg is installed."

    last = list(state.keys())[-1] if state else None
    bar_fig = _make_bar_fig(state[last]["top_names"], state[last]["top_probs"]) if last else _empty
    card    = _build_species_card(state[last]["top_names"][0], state[last]["top_probs"][0]) if last else ""

    yield (
        gallery, pd.DataFrame(rows), status,
        gr.Dropdown(choices=list(state.keys()), value=last, interactive=True, label="Select a file to inspect"),
        state,
        state[last]["audio_path"] if last else None,
        bar_fig, card,
    )


# ---------------------------------------------------------------------------
# Detail-panel callback — triggered when user changes the file dropdown
# ---------------------------------------------------------------------------

def show_detail(selected: str, state: dict):
    if not selected or not state or selected not in state:
        return None, _empty_bar_fig(), ""
    d      = state[selected]
    bar_fig = _make_bar_fig(d["top_names"], d["top_probs"])
    card    = _build_species_card(d["top_names"][0], d["top_probs"][0])
    return d["audio_path"], bar_fig, card


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

_SPECIES_DISPLAY = [
    "Ring ouzel", "Black redstart", "Alpine accentor", "Yellow-billed chough",
    "Red-billed chough", "Wallcreeper", "Water pipit", "White-winged snowfinch",
    "Rock ptarmigan", "Black woodpecker", "Western capercaillie", "Three-toed woodpecker",
    "Common crossbill", "Spotted nutcracker", "Firecrest", "White-throated dipper",
    "Collared flycatcher", "Whinchat", "Rock bunting", "Bearded vulture",
]

_PILLS_HTML = (
    '<div class="species-pills">'
    + "".join(f'<span class="species-pill">{s}</span>' for s in _SPECIES_DISPLAY)
    + "</div>"
)


def build_ui(checkpoint: str = DEFAULT_CHECKPOINT) -> gr.Blocks:
    _ckpt_choices, _ckpt_default = _discover_checkpoints(checkpoint)
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
        gr.HTML('<hr style="border:none;border-top:1px solid #e0e0e0;margin:4px 0 16px">')

        # ── Main row ─────────────────────────────────────────────────────────
        with gr.Row(equal_height=False):

            # Left panel — upload + controls
            with gr.Column(scale=1, min_width=260):
                gr.Markdown("### Upload")
                file_input = gr.File(
                    label="Audio files (.mp3 / .wav) or .zip archive",
                    file_count="multiple",
                    file_types=[".mp3", ".wav", ".zip"],
                )
                with gr.Accordion("⚙️ Settings", open=False):
                    checkpoint_input = gr.Dropdown(
                        label="Model checkpoint",
                        choices=_ckpt_choices,
                        value=_ckpt_default,
                        interactive=True,
                    )
                run_btn = gr.Button("🔍  Classify", variant="primary", elem_id="run-btn")
                status_output = gr.Textbox(
                    label="Status", interactive=False, lines=2, max_lines=4,
                )

            # Right panel — results tabs
            with gr.Column(scale=3):
                with gr.Tabs():
                    with gr.Tab("📊 Summary"):
                        gr.HTML(
                            '<p style="font-size:0.78rem;color:#666;margin:0 0 6px">'
                            '🟢 ≥ 80% &nbsp;·&nbsp; 🟡 40–79% &nbsp;·&nbsp; 🔴 &lt; 40%'
                            '</p>'
                        )
                        table_output = gr.Dataframe(
                            headers=["", "File", "Predicted Species", "Confidence"],
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

        # ── Detail section ───────────────────────────────────────────────────
        gr.HTML('<hr style="border:none;border-top:1px solid #e0e0e0;margin:18px 0 12px">')
        gr.Markdown("### Detail view")

        file_dropdown = gr.Dropdown(
            choices=[],
            label="Select a file to inspect",
            interactive=True,
        )

        with gr.Row(equal_height=False):

            # B — Audio player
            with gr.Column(scale=1, min_width=220):
                audio_player = gr.Audio(label="Audio playback", interactive=False)

            # A — Top-K bar chart (matplotlib, works across all Gradio versions)
            with gr.Column(scale=2):
                bar_plot = gr.Plot(value=_empty_bar_fig(), label=f"Top-{TOP_K} predictions")

            # C — Species info card
            with gr.Column(scale=1, min_width=220):
                species_card = gr.HTML("")

        # ── Footer ──────────────────────────────────────────────────────────
        gr.HTML(
            '<p class="app-footer">'
            'Audio data from <a href="https://xeno-canto.org" target="_blank">Xeno-canto</a>'
            ' &nbsp;·&nbsp; Alpine zone &nbsp;·&nbsp; Italy / Austria / Switzerland'
            '</p>'
        )

        # ── State (must be inside Blocks context) ───────────────────────────
        results_state = gr.State({})

        # ── Events ──────────────────────────────────────────────────────────
        _classify_outputs = [
            gallery_output, table_output, status_output,
            file_dropdown, results_state,
            audio_player, bar_plot, species_card,
        ]

        run_btn.click(
            fn=classify_files,
            inputs=[file_input, checkpoint_input],
            outputs=_classify_outputs,
        )

        file_dropdown.change(
            fn=show_detail,
            inputs=[file_dropdown, results_state],
            outputs=[audio_player, bar_plot, species_card],
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
        theme=THEME,
        css=CSS,
    )
