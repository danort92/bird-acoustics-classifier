# Bird Acoustics Classifier

Automatic bird species recognition from audio recordings using **EfficientNet fine-tuning on mel spectrograms**, with data sourced from the **Xeno-canto API**.

## Description

The project implements a complete pipeline for automatic bird species identification from audio recordings:

1. **Data download** — Fetch `.mp3` audio recordings from [Xeno-canto](https://xeno-canto.org/) via API
2. **Pre-processing** — Convert audio → mel spectrograms (2D images)
3. **Training** — Fine-tune EfficientNet on mel spectrograms
4. **Evaluation** — Metrics and reports tracked with MLflow
5. **Demo** — Interactive web interface with Gradio

## Project structure

```
bird-acoustics-classifier/
├── data/
│   ├── raw/            # .mp3 audio files downloaded from Xeno-canto (per species)
│   └── processed/      # Pre-processed mel spectrograms
├── notebooks/          # Step-by-step Jupyter notebooks
├── src/                # Reusable Python modules
├── app/                # Gradio application
├── models/             # Saved checkpoints and models
├── reports/            # Plots, metrics, MLflow reports
├── requirements.txt
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

## Configuration

Set your Xeno-canto API key:

```bash
export XENO_CANTO_API_KEY="your_api_key_here"
```

> If not set, the module will prompt for the key interactively.

## Quick start

```python
from src.download import XenoCantoDownloader

downloader = XenoCantoDownloader(output_dir="data/raw")
species = ["Turdus merula", "Erithacus rubecula", "Parus major"]
downloader.download_species(species, max_per_species=50)
```

## Notebooks

| Notebook | Description | Colab |
|---|---|---|
| `01_download.ipynb` | Download audio recordings from the Xeno-canto API | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/danort92/bird-acoustics-classifier/blob/claude/setup-project-structure-lVRTH/notebooks/01_download.ipynb) |

## Technologies

- **PyTorch / TorchVision** — EfficientNet training and fine-tuning
- **Torchaudio / Librosa** — Audio processing and mel spectrograms
- **Gradio** — Interactive demo interface
- **MLflow** — Experiment tracking and metrics
- **Xeno-canto API** — Bird song audio dataset
