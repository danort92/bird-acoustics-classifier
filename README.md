# Bird Acoustics Classifier

Classificatore di specie di uccelli da audio usando **EfficientNet fine-tuning su spettrogrammi mel**, con dati scaricati dall'**API Xeno-canto**.

## Descrizione

Il progetto implementa una pipeline completa per il riconoscimento automatico di specie di uccelli a partire da registrazioni audio:

1. **Download dati** — Recupero di tracce audio `.mp3` da [Xeno-canto](https://xeno-canto.org/) tramite API
2. **Pre-processing** — Conversione audio → spettrogrammi mel (immagini 2D)
3. **Training** — Fine-tuning di EfficientNet su spettrogrammi mel
4. **Valutazione** — Metriche e report con MLflow
5. **Demo** — Interfaccia web interattiva con Gradio

## Struttura del progetto

```
bird-acoustics-classifier/
├── data/
│   ├── raw/            # Audio .mp3 scaricati da Xeno-canto (per specie)
│   └── processed/      # Spettrogrammi mel pre-processati
├── notebooks/          # Jupyter notebook step-by-step
├── src/                # Moduli Python riutilizzabili
├── app/                # Applicazione Gradio
├── models/             # Checkpoint e modelli salvati
├── reports/            # Grafici, metriche, report MLflow
├── requirements.txt
└── README.md
```

## Installazione

```bash
pip install -r requirements.txt
```

## Configurazione

Imposta la tua API key di Xeno-canto:

```bash
export XENO_CANTO_API_KEY="your_api_key_here"
```

> Se non impostata, il modulo chiederà la chiave in modo interattivo.

## Utilizzo rapido

```python
from src.download import XenoCantoDownloader

downloader = XenoCantoDownloader(output_dir="data/raw")
species = ["Turdus merula", "Erithacus rubecula", "Parus major"]
downloader.download_species(species, max_per_species=50)
```

## Notebook

| Notebook | Descrizione |
|---|---|
| `01_download.ipynb` | Download audio da Xeno-canto API |

## Tecnologie

- **PyTorch / TorchVision** — Training e fine-tuning EfficientNet
- **Torchaudio / Librosa** — Elaborazione audio e spettrogrammi mel
- **Gradio** — Interfaccia demo interattiva
- **MLflow** — Tracking esperimenti e metriche
- **Xeno-canto API** — Dataset audio di canti di uccelli
