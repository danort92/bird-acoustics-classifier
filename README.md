# Bird Acoustics Classifier

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
dl.download_species(["Turdus torquatus", "Cinclus cinclus"], max_per_species=30)
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

The app accepts `.mp3` or `.wav` files, slices them into 5-second clips, runs the model on each clip, and returns the top-k species with confidence scores, plus the mel spectrogram of the first clip.

---

## Training from scratch

No pre-trained checkpoint is required. The entire pipeline — from raw audio to a ready-to-use model — runs automatically with the commands above. There is **nothing to upload manually**.

The sequence is:

```
API key → download .mp3 → generate spectrograms → train → models/best_model.pt → Gradio app
```

`scripts/train.py` calls `BirdTrainer.train()`, which automatically saves the best checkpoint (lowest validation loss) to `models/best_model.pt` at the end of training. The Gradio app reads that file by default.

If a pre-trained `best_model.pt` is already present in the repo (tracked via Git LFS), it is used immediately by `python app/app.py` without retraining. To retrain from scratch anyway:

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
  quality: "all"          # "all" = any grade; "A" = best quality only
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
| **Pillow / NumPy** | Image handling and array operations |
