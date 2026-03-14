"""
src/preprocessing.py — Audio-to-mel-spectrogram preprocessing pipeline.

Converts .mp3 recordings (organised as data/raw/<species>/*.mp3) into
mel-spectrogram images (.png) saved under data/processed/<species>/.

Each long recording is sliced into fixed-duration clips so every image
has the same shape — a requirement for CNN training.

Usage:
    from src.preprocessing import SpectrogramConverter

    conv = SpectrogramConverter(output_dir="data/processed")
    conv.process_all(input_dir="data/raw")

Compatible with local environments and Google Colab.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import librosa
import librosa.display
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class AudioConfig:
    """All tunable parameters for the mel-spectrogram conversion."""

    sample_rate: int = 22050        # Hz — standard for bird sound models
    clip_duration: float = 5.0     # seconds per spectrogram tile
    n_mels: int = 128              # mel filter banks (height of image)
    n_fft: int = 2048              # FFT window size
    hop_length: int = 512          # frames between successive FFTs
    f_min: float = 500.0           # low-frequency cut-off (Hz) — reduces wind noise
    f_max: float = 15000.0         # high-frequency cut-off (Hz)
    top_db: float = 80.0           # dynamic range for log-amplitude clipping
    img_size: Tuple[int, int] = field(default_factory=lambda: (224, 224))  # (W, H) px
    # Overlap between consecutive clips (0.0 = no overlap, 0.5 = 50 %)
    clip_overlap: float = 0.0


# ---------------------------------------------------------------------------
# Core converter
# ---------------------------------------------------------------------------

class SpectrogramConverter:
    """Convert a directory tree of .mp3 files into mel-spectrogram .png images.

    Directory layout expected:
        input_dir/
            <species_a>/
                recording1.mp3
                recording2.mp3
            <species_b>/
                ...

    Output layout produced:
        output_dir/
            <species_a>/
                recording1_clip000.png
                recording1_clip001.png
                ...
            <species_b>/
                ...
    """

    def __init__(
        self,
        output_dir: str = "data/processed",
        config: Optional[AudioConfig] = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.config = config or AudioConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_all(
        self,
        input_dir: str = "data/raw",
        overwrite: bool = False,
        species: list[str] | None = None,
    ) -> dict[str, int]:
        """Process .mp3 files found under *input_dir*.

        Args:
            species: Optional list of species names (e.g. ``["Turdus merula"]``).
                     When provided, only matching sub-directories are processed.
                     Spaces are normalised to underscores for directory matching.
                     When *None*, all sub-directories are processed.

        Returns a dict mapping species directory name → number of clips produced.
        """
        input_path = Path(input_dir)
        allowed = (
            {s.replace(" ", "_").lower() for s in species} if species is not None else None
        )
        species_dirs = sorted(
            d for d in input_path.iterdir()
            if d.is_dir() and (allowed is None or d.name in allowed)
        )

        if not species_dirs:
            logger.warning("No species sub-directories found in %s", input_path)
            return {}

        summary: dict[str, int] = {}
        for species_dir in species_dirs:
            clips = self.process_species(species_dir, overwrite=overwrite)
            summary[species_dir.name] = clips

        total = sum(summary.values())
        logger.info("Done. %d species | %d total clips", len(summary), total)
        return summary

    def process_species(
        self,
        species_dir: Path,
        overwrite: bool = False,
    ) -> int:
        """Process all .mp3 files for a single species directory."""
        mp3_files = sorted(species_dir.glob("*.mp3"))
        if not mp3_files:
            logger.warning("No .mp3 files in %s — skipping", species_dir)
            return 0

        out_species = self.output_dir / species_dir.name
        out_species.mkdir(parents=True, exist_ok=True)

        total_clips = 0
        for mp3 in tqdm(mp3_files, desc=species_dir.name, leave=False):
            clips = self._convert_file(mp3, out_species, overwrite=overwrite)
            total_clips += clips

        logger.info("%-35s → %d clips", species_dir.name, total_clips)
        return total_clips

    def convert_file(
        self,
        mp3_path: str | Path,
        output_dir: str | Path,
        overwrite: bool = False,
    ) -> List[Path]:
        """Public wrapper: convert a single .mp3 and return list of output paths."""
        saved = self._convert_file(
            Path(mp3_path), Path(output_dir), overwrite=overwrite, return_paths=True
        )
        return saved  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _load_audio(self, path: Path) -> Optional[np.ndarray]:
        """Load audio, resample, and return mono waveform.  Returns None on error."""
        try:
            y, _ = librosa.load(path, sr=self.config.sample_rate, mono=True)
            return y
        except Exception as exc:
            logger.error("Cannot load %s: %s", path.name, exc)
            return None

    def _audio_to_mel(self, y: np.ndarray) -> np.ndarray:
        """Compute log-amplitude mel spectrogram for a waveform array."""
        cfg = self.config
        S = librosa.feature.melspectrogram(
            y=y,
            sr=cfg.sample_rate,
            n_fft=cfg.n_fft,
            hop_length=cfg.hop_length,
            n_mels=cfg.n_mels,
            fmin=cfg.f_min,
            fmax=cfg.f_max,
        )
        S_db = librosa.power_to_db(S, ref=np.max, top_db=cfg.top_db)
        return S_db  # shape: (n_mels, time_frames)

    def _save_spectrogram(
        self,
        S_db: np.ndarray,
        save_path: Path,
        sr: int,
        hop_length: int,
        f_min: float,
        f_max: float,
    ) -> None:
        """Render *S_db* as a borderless PNG image."""
        W, H = self.config.img_size
        dpi = 100
        fig, ax = plt.subplots(figsize=(W / dpi, H / dpi), dpi=dpi)
        librosa.display.specshow(
            S_db,
            sr=sr,
            hop_length=hop_length,
            fmin=f_min,
            fmax=f_max,
            x_axis=None,
            y_axis=None,
            ax=ax,
            cmap="magma",
        )
        ax.set_axis_off()
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight", pad_inches=0)
        plt.close(fig)

    def _convert_file(
        self,
        mp3_path: Path,
        out_dir: Path,
        overwrite: bool = False,
        return_paths: bool = False,
    ):
        """Slice one audio file into fixed-length clips and save each as PNG.

        Returns the number of clips saved (or a list of paths if return_paths).
        """
        cfg = self.config
        clip_samples = int(cfg.clip_duration * cfg.sample_rate)
        step_samples = int(clip_samples * (1.0 - cfg.clip_overlap))

        y = self._load_audio(mp3_path)
        if y is None:
            return [] if return_paths else 0

        # Pad short recordings to at least one full clip
        if len(y) < clip_samples:
            y = np.pad(y, (0, clip_samples - len(y)))

        saved_paths: List[Path] = []
        clip_idx = 0
        start = 0
        while start + clip_samples <= len(y):
            clip = y[start : start + clip_samples]
            tag = f"_clip{clip_idx:03d}.png"
            save_path = out_dir / (mp3_path.stem + tag)

            if overwrite or not save_path.exists():
                S_db = self._audio_to_mel(clip)
                self._save_spectrogram(
                    S_db,
                    save_path,
                    sr=cfg.sample_rate,
                    hop_length=cfg.hop_length,
                    f_min=cfg.f_min,
                    f_max=cfg.f_max,
                )
            saved_paths.append(save_path)
            clip_idx += 1
            start += step_samples

        return saved_paths if return_paths else len(saved_paths)

    # ------------------------------------------------------------------
    # Visualisation helpers (used in the notebook)
    # ------------------------------------------------------------------

    def plot_spectrogram(
        self,
        mp3_path: str | Path,
        clip_index: int = 0,
        ax: Optional[plt.Axes] = None,
        title: Optional[str] = None,
    ) -> plt.Figure:
        """Return a matplotlib Figure showing the mel spectrogram of one clip.

        Parameters
        ----------
        mp3_path : path to the audio file
        clip_index : which clip window to plot (0-based)
        ax : existing Axes to draw into (optional)
        title : axis title (defaults to filename + clip index)
        """
        cfg = self.config
        mp3_path = Path(mp3_path)

        y = self._load_audio(mp3_path)
        if y is None:
            raise ValueError(f"Could not load {mp3_path}")

        clip_samples = int(cfg.clip_duration * cfg.sample_rate)
        step_samples = int(clip_samples * (1.0 - cfg.clip_overlap))

        if len(y) < clip_samples:
            y = np.pad(y, (0, clip_samples - len(y)))

        start = clip_index * step_samples
        if start + clip_samples > len(y):
            raise IndexError(
                f"clip_index={clip_index} out of range for {mp3_path.name} "
                f"(audio length {len(y)/cfg.sample_rate:.1f}s, "
                f"clip duration {cfg.clip_duration}s)"
            )
        clip = y[start : start + clip_samples]
        S_db = self._audio_to_mel(clip)

        standalone = ax is None
        if standalone:
            fig, ax = plt.subplots(figsize=(6, 3))
        else:
            fig = ax.get_figure()

        img = librosa.display.specshow(
            S_db,
            sr=cfg.sample_rate,
            hop_length=cfg.hop_length,
            fmin=cfg.f_min,
            fmax=cfg.f_max,
            x_axis="time",
            y_axis="mel",
            ax=ax,
            cmap="magma",
        )
        fig.colorbar(img, ax=ax, format="%+2.0f dB")
        ax.set_title(title or f"{mp3_path.name} — clip {clip_index}")

        return fig

    def plot_waveform_and_spectrogram(
        self,
        mp3_path: str | Path,
        clip_index: int = 0,
    ) -> plt.Figure:
        """Side-by-side waveform + mel spectrogram for one audio clip."""
        cfg = self.config
        mp3_path = Path(mp3_path)

        y = self._load_audio(mp3_path)
        if y is None:
            raise ValueError(f"Could not load {mp3_path}")

        clip_samples = int(cfg.clip_duration * cfg.sample_rate)
        step_samples = int(clip_samples * (1.0 - cfg.clip_overlap))

        if len(y) < clip_samples:
            y = np.pad(y, (0, clip_samples - len(y)))

        start = clip_index * step_samples
        clip = y[start : start + clip_samples]
        S_db = self._audio_to_mel(clip)

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        times = np.linspace(0, cfg.clip_duration, len(clip))
        axes[0].plot(times, clip, color="steelblue", linewidth=0.5)
        axes[0].set_xlabel("Time (s)")
        axes[0].set_ylabel("Amplitude")
        axes[0].set_title("Waveform")
        axes[0].set_xlim(0, cfg.clip_duration)

        img = librosa.display.specshow(
            S_db,
            sr=cfg.sample_rate,
            hop_length=cfg.hop_length,
            fmin=cfg.f_min,
            fmax=cfg.f_max,
            x_axis="time",
            y_axis="mel",
            ax=axes[1],
            cmap="magma",
        )
        fig.colorbar(img, ax=axes[1], format="%+2.0f dB")
        axes[1].set_title("Mel Spectrogram")

        fig.suptitle(f"{mp3_path.name} — clip {clip_index}", fontsize=11)
        fig.tight_layout()
        return fig
