# Bird Acoustics Classifier

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![EfficientNet](https://img.shields.io/badge/Model-EfficientNet--B0-blueviolet)
![librosa](https://img.shields.io/badge/Audio-librosa-orange)
![Gradio](https://img.shields.io/badge/UI-Gradio-F97316?logo=gradio&logoColor=white)
![MLflow](https://img.shields.io/badge/Tracking-MLflow-0194E2?logo=mlflow&logoColor=white)
![Xeno--canto](https://img.shields.io/badge/Data-Xeno--canto%20API-4CAF50)
![Species](https://img.shields.io/badge/Species-20%20Alpine-8BC34A)
![License](https://img.shields.io/badge/License-MIT-yellow)

Automatic bird species recognition from audio recordings using **EfficientNet-B0 fine-tuning on mel spectrograms**, with data sourced from the **Xeno-canto API**.

Target habitat: **Alpine zone** (Italy / Austria / Switzerland) — 20 characteristic species:

| Species | Common name | Habitat |
|---------|-------------|---------|
| *Turdus torquatus* | Ring ouzel | Rocky slopes, high-altitude forests |
| *Phoenicurus ochruros* | Black redstart | Rocky terrain, mountain villages |
| *Prunella collaris* | Alpine accentor | High rocky areas above treeline |
| *Pyrrhocorax graculus* | Yellow-billed chough | Alpine cliffs and glaciers |
| *Pyrrhocorax pyrrhocorax* | Red-billed chough | Alpine meadows and cliffs |
| *Tichodroma muraria* | Wallcreeper | Vertical rock faces |
| *Anthus spinoletta* | Water pipit | Alpine meadows and streams |
| *Montifringilla nivalis* | White-winged snowfinch | Above treeline, snowfields |
| *Lagopus muta* | Rock ptarmigan | High alpine tundra |
| *Dryocopus martius* | Black woodpecker | Subalpine conifer forests |
| *Tetrao urogallus* | Western capercaillie | Old-growth conifer forests |
| *Picoides tridactylus* | Three-toed woodpecker | Spruce forests |
| *Loxia curvirostra* | Common crossbill | Conifer forests |
| *Nucifraga caryocatactes* | Spotted nutcracker | Mountain conifer forests |
| *Regulus ignicapilla* | Firecrest | Mixed mountain forests |
| *Cinclus cinclus* | White-throated dipper | Alpine streams and torrents |
| *Ficedula albicollis* | Collared flycatcher | Deciduous mountain forests |
| *Saxicola rubetra* | Whinchat | Subalpine meadows |
| *Emberiza cia* | Rock bunting | Rocky slopes with sparse vegetation |
| *Gypaetus barbatus* | Bearded vulture | High alpine cliffs (reintroduced) |

---

## Why 20 species — and how to add more

### Rationale for the current selection

The 20 species were chosen based on two criteria:

1. **Ecological coherence** — all are characteristic of the Alpine zone (Italy / Austria / Switzerland), making the classifier useful for a single, well-defined habitat.
2. **Compute budget** — with `max_per_species: 100` recordings and 30 training epochs on a single consumer GPU (or Google Colab free tier), the full pipeline completes in roughly 2–3 hours. Scaling to more species increases download size, preprocessing time, and training time roughly linearly.

> On a machine without a GPU, training 20 species for 30 epochs already takes several hours. Adding more species without access to a dedicated GPU or cloud accelerator is feasible but slow.

### Adding more species

The entire pipeline is driven by the species list in `config/default.yaml` — no code changes are needed.

**Step 1 — Find valid species names**

Use the scientific name exactly as it appears on [Xeno-canto](https://xeno-canto.org). Search the site to verify that enough recordings exist (aim for at least 30–50 per species).

**Step 2 — Edit the config**

```yaml
# config/default.yaml
species:
  - Turdus torquatus
  - Cinclus cinclus
  # ... existing 18 species ...
  - Aquila chrysaetos       # Golden eagle  ← add new species here
  - Tetrao tetrix           # Black grouse
```

**Step 3 — Re-run the pipeline**

```bash
# Download recordings only for the new species (faster)
python scripts/download.py --species "Aquila chrysaetos" "Tetrao tetrix" --max 100

# Or re-download everything from scratch
python scripts/download.py

# Regenerate spectrograms (skip existing ones automatically)
python scripts/preprocess.py

# Retrain — the model head is rebuilt to match the new number of classes
python scripts/train.py

# Launch the updated demo
python app/app.py
```

> `train.py` rebuilds the EfficientNet-B0 classification head automatically to match the number of species found in `data/processed/`. You do **not** need to edit any code — only the YAML.

**Practical limits (rough estimates)**

| Species | ~Audio files | ~Preprocessing | ~Training (GPU) | ~Training (CPU only) |
|--------:|-------------:|---------------:|----------------:|---------------------:|
| 20 | 2 000 | 20 min | 1–2 h | 4–8 h |
| 50 | 5 000 | 45 min | 3–5 h | 12–20 h |
| 100 | 10 000 | 1.5 h | 6–10 h | 30–50 h |

For large expansions, consider reducing `max_per_species` (e.g. 50) or increasing `batch_size` and using a cloud GPU (Colab, Kaggle, Vast.ai).

---

## Pipeline

| Step | Module | Notebook | CLI script |
|------|--------|----------|------------|
| 1. Download audio | `src/download.py` | `pipeline.ipynb` | `scripts/download.py` |
| 2. Audio → mel spectrograms | `src/preprocessing.py` | `pipeline.ipynb` | `scripts/preprocess.py` |
| 3. Train EfficientNet-B0 | `src/model.py` | `pipeline.ipynb` | `scripts/train.py` |
| 4. Evaluate & track metrics | `src/model.py` | `pipeline.ipynb` | `scripts/evaluate.py` |
| 5. Interactive demo | `app/app.py` | — | `python app/app.py` |

---

## Project structure

```
bird-acoustics-classifier/
├── config/
│   └── default.yaml        # centralised config (species, audio, training, mlflow)
├── data/
│   ├── raw/                # .mp3 recordings from Xeno-canto (per species)
│   └── processed/          # mel spectrogram .png tiles (per species)
├── models/                 # saved model checkpoints (best_model.pt)
├── notebooks/
│   └── pipeline.ipynb      # full pipeline: download → preprocessing → training → evaluation
├── outputs/                # training artefacts (loss curves, confusion matrices)
├── reports/                # evaluation reports and plots
├── scripts/                # CLI entry points (terminal-friendly alternatives to notebook)
│   ├── download.py
│   ├── preprocess.py
│   ├── train.py
│   ├── evaluate.py
│   └── infer.py
├── src/                    # reusable Python modules
│   ├── download.py         # Xeno-canto API downloader
│   ├── preprocessing.py    # audio → mel spectrogram converter
│   └── model.py            # EfficientNet-B0, BirdTrainer, inference helpers
├── app/
│   └── app.py              # Gradio web interface
├── tests/
│   ├── test_download.py
│   └── test_preprocessing.py
└── requirements.txt
```

---

## Installation

```bash
git clone https://github.com/danort92/bird-acoustics-classifier.git
cd bird-acoustics-classifier
pip install -r requirements.txt
```

Set your Xeno-canto API key (required since October 2025):

```bash
export XENO_CANTO_API_KEY="your_api_key_here"
```

> Get a free key at <https://xeno-canto.org/article/854> after registering.
> If not set, the downloader will prompt interactively.

---

## Quick start

### 1 — Download recordings

```python
from src.download import XenoCantoDownloader

dl = XenoCantoDownloader(output_dir="data/raw")

# Grade-A only (cleanest recordings)
dl.download_species(["Turdus torquatus", "Cinclus cinclus"], max_per_species=50)

# Mixed quality — improves robustness on real-world recordings
dl.download_species(
    ["Turdus torquatus", "Cinclus cinclus"],
    max_per_species=100,
    quality_mix={"A": 60, "B": 30, "C": 10},
)
```

Or via CLI:

```bash
# All species in config/default.yaml
python scripts/download.py

# Custom species list
python scripts/download.py --species "Turdus torquatus" "Cinclus cinclus" --max 50
```

### 2 — Generate mel spectrograms

```python
from src.preprocessing import SpectrogramConverter, AudioConfig

conv = SpectrogramConverter(output_dir="data/processed")
conv.process_all(input_dir="data/raw")
```

Or via CLI:

```bash
python scripts/preprocess.py          # uses config/default.yaml
python scripts/preprocess.py --overwrite   # overwrite existing PNGs
```

### 3 — Train the model

```python
from src.model import BirdTrainer, TrainingConfig

cfg     = TrainingConfig.from_yaml()   # reads config/default.yaml
trainer = BirdTrainer(cfg)
best_path, history = trainer.train()   # saves models/best_model.pt
```

Or via CLI:

```bash
python scripts/train.py
python scripts/train.py --epochs 50 --batch-size 64 --lr 5e-4
```

Training logs per-epoch loss/accuracy to the console and to MLflow. The best checkpoint (lowest val loss) is saved to `models/best_model.pt`.

### 4 — Evaluate

```python
from src.model import BirdTrainer, TrainingConfig

cfg     = TrainingConfig.from_yaml()
trainer = BirdTrainer(cfg)
y_true, y_pred = trainer.evaluate("models/best_model.pt")
```

Or via CLI:

```bash
python scripts/evaluate.py --checkpoint models/best_model.pt
```

### 5 — Interactive demo (Gradio)

```bash
python app/app.py
```

Then open **http://localhost:7860** in your browser.

Options:

```bash
python app/app.py --checkpoint models/best_model.pt   # custom checkpoint
python app/app.py --port 8080                         # custom port
python app/app.py --share                             # public Gradio link
```

The app accepts `.mp3` or `.wav` files (or a `.zip` archive), slices them into 5-second clips, runs the model on each clip, and returns the best species prediction with confidence score, plus the mel spectrogram of the first clip.

The **Settings** panel in the UI includes a **Model checkpoint** dropdown that automatically discovers all `.pt` files in the `models/` directory — no restart needed to switch between checkpoints.

---

## Training from scratch

No pre-trained checkpoint is required. The entire pipeline — from raw audio to a ready-to-use model — runs automatically with the commands above. There is **nothing to upload manually**.

The sequence is:

```
API key → download .mp3 → generate spectrograms → train → models/best_model.pt → Gradio app
```

`scripts/train.py` calls `BirdTrainer.train()`, which automatically saves the best checkpoint (lowest validation loss) to `models/best_model.pt` at the end of training. The Gradio app reads that file by default, and it will appear automatically in the UI checkpoint dropdown.

To retrain from scratch:

```bash
python scripts/train.py          # overwrites models/best_model.pt
python app/app.py                # now uses your freshly trained model
```

---

## Notebook

| Notebook | Description | Colab |
|----------|-------------|-------|
| `pipeline.ipynb` | Full pipeline: download → preprocessing → training → evaluation | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/danort92/bird-acoustics-classifier/blob/claude/setup-project-structure-lVRTH/notebooks/pipeline.ipynb) |

### Running locally

```bash
pip install -r requirements.txt
jupyter notebook notebooks/pipeline.ipynb
```

### Google Colab

Open the badge above, then **Runtime → Run all**. The setup cell clones the repo, installs dependencies, and symlinks `data/raw` and `data/processed` to Google Drive so files survive session restarts.

---

## Configuration

All parameters live in `config/default.yaml`. Edit it to change species, audio settings, or training hyperparameters without touching the code:

```yaml
species:
  - Turdus torquatus
  - Cinclus cinclus
  # ... 18 more

download:
  max_per_species: 100
  quality: "A"            # grade filter when quality_mix is not set (A–E)
  # quality_mix:          # blend of grades — weights are relative, not absolute counts
  #   A: 60               # ~60 % grade-A
  #   B: 30               # ~30 % grade-B
  #   C: 10               # ~10 % grade-C
  countries: []           # e.g. ["Italy", "Austria"] — empty = worldwide

audio:
  sample_rate: 22050
  clip_duration: 5.0      # seconds per spectrogram tile
  n_mels: 128
  n_fft: 2048
  hop_length: 512
  f_min: 500.0            # Hz — filters wind/traffic noise
  f_max: 15000.0          # Hz
  top_db: 80.0            # log-amplitude dynamic range
  img_size: [224, 224]    # matches EfficientNet input

training:
  model: efficientnet_b0
  batch_size: 32
  epochs: 30
  learning_rate: 0.001
  val_split: 0.15
  test_split: 0.15
  seed: 42
  patience: 7             # early stopping
```

---

## Experiment tracking (MLflow)

By default MLflow logs to a local `mlruns/` folder. To use **DagsHub** (free remote tracking):

1. Create a free account at <https://dagshub.com> and connect this repository.
2. Export the following variables before training:

```bash
export MLFLOW_TRACKING_URI=https://dagshub.com/<username>/bird-acoustics-classifier.mlflow
export MLFLOW_TRACKING_USERNAME=<your-dagshub-username>
export MLFLOW_TRACKING_PASSWORD=<your-dagshub-token>
```

No code change needed — the env var overrides the `tracking_uri` in the config.

To browse the local UI:

```bash
mlflow ui
# open http://localhost:5000
```

---

## Technologies

| Library | Role |
|---------|------|
| **PyTorch / TorchVision** | EfficientNet-B0 training and fine-tuning |
| **Librosa** | Audio loading and mel spectrogram computation |
| **Gradio** | Interactive web demo |
| **MLflow** | Experiment tracking and checkpoint logging |
| **Xeno-canto API v3** | Bird song audio dataset |
| **scikit-learn** | Stratified splits, evaluation metrics |
| **soundfile** | Audio file I/O backend for Librosa (WAV/FLAC/OGG) |
| **Pillow / NumPy** | Image handling and array operations |
