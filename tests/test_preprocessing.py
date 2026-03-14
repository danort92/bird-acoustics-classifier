"""Unit tests for src/preprocessing.py."""

import numpy as np
import pytest
from pathlib import Path

from src.preprocessing import AudioConfig, SpectrogramConverter


def test_audio_config_defaults():
    cfg = AudioConfig()
    assert cfg.sample_rate == 22050
    assert cfg.clip_duration == 5.0
    assert cfg.n_mels == 128


def test_audio_to_mel_shape():
    cfg = AudioConfig(sample_rate=22050, clip_duration=5.0)
    converter = SpectrogramConverter(config=cfg)

    clip_samples = int(cfg.clip_duration * cfg.sample_rate)
    y = np.random.randn(clip_samples).astype(np.float32)
    S_db = converter._audio_to_mel(y)

    assert S_db.shape[0] == cfg.n_mels


def test_process_all_empty_dir(tmp_path):
    converter = SpectrogramConverter(output_dir=str(tmp_path / "processed"))
    summary = converter.process_all(input_dir=str(tmp_path / "raw"))
    assert summary == {}
