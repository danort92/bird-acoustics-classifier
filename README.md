# Bird Acoustics Classifier

Automatic bird species recognition from audio recordings using **EfficientNet fine-tuning on mel spectrograms**, with data sourced from the **Xeno-canto API**.

Target habitat: **Alpine zone** (Italy / Austria / Switzerland) — 20 characteristic species.

## Pipeline

| Step | Module | Notebook |
|------|--------|----------|
| 1. Download audio | `src/download.py` | `01_download.ipynb` |
| 2. Audio → mel spectrograms | `src/preprocessing.py` | `02_preprocessing.ipynb` |
| 3. Train EfficientNet | *(coming)* | `03_training.ipynb` |
| 4. Evaluate & track metrics | *(coming)* | `04_evaluation.ipynb` |
| 5. Interactive demo | *(coming)* | Gradio app |

## Project structure

```
bird-acoustics-classifier/
├── data/
│   ├── raw/            # .mp3 recordings from Xeno-canto (per species)
│   └── processed/      # mel spectrogram .png tiles (per species)
├── notebooks/          # Step-by-step Jupyter / Colab notebooks
├── src/                # Reusable Python modules
├── app/                # Gradio application (coming)
├── models/             # Saved checkpoints
├── reports/            # Plots and metrics
└── requirements.txt
```

## Notebooks

| Notebook | Description | Colab |
|----------|-------------|-------|
| `01_download.ipynb` | Download `.mp3` recordings from Xeno-canto API | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/danort92/bird-acoustics-classifier/blob/claude/setup-project-structure-lVRTH/notebooks/01_download.ipynb) |
| `02_preprocessing.ipynb` | Convert audio → mel spectrogram PNG tiles | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/danort92/bird-acoustics-classifier/blob/claude/setup-project-structure-lVRTH/notebooks/02_preprocessing.ipynb) |

### Google Colab — data persistence across sessions

Both notebooks store `data/` on **Google Drive** so files survive when the Colab VM is recycled:

```
MyDrive/
└── bird-acoustics-classifier/
    └── data/
        ├── raw/          ← filled by notebook 01
        └── processed/    ← filled by notebook 02
```

When you open **notebook 02 in a new session**, the first cell mounts Drive and sets:

```python
RAW_DIR       = '/content/drive/MyDrive/bird-acoustics-classifier/data/raw'
PROCESSED_DIR = '/content/drive/MyDrive/bird-acoustics-classifier/data/processed'
```

so it automatically finds the recordings downloaded by notebook 01 — **no need to re-download**.

## Installation (local)

```bash
pip install -r requirements.txt
```

Set your Xeno-canto API key (required since October 2025):

```bash
export XENO_CANTO_API_KEY="your_api_key_here"
```

> Obtain a free key at <https://xeno-canto.org/article/854> after registering.
> If not set, the downloader will prompt interactively.

## Quick start (local)

```python
# 1 — Download
from src.download import XenoCantoDownloader
dl = XenoCantoDownloader(output_dir="data/raw")
dl.download_species(["Turdus torquatus", "Cinclus cinclus"], max_per_species=30)

# 2 — Preprocess
from src.preprocessing import SpectrogramConverter, AudioConfig
conv = SpectrogramConverter(output_dir="data/processed")
conv.process_all(input_dir="data/raw")
```

## Technologies

- **PyTorch / TorchVision** — EfficientNet training and fine-tuning
- **Librosa** — audio loading and mel spectrogram computation
- **Gradio** — interactive demo interface
- **MLflow** — experiment tracking
- **Xeno-canto API v3** — bird song audio dataset
