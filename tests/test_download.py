"""Unit tests for src/download.py."""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.download import XenoCantoDownloader, _sanitise_name


def test_sanitise_name():
    assert _sanitise_name("Turdus merula") == "turdus_merula"
    assert _sanitise_name("  Parus major  ") == "parus_major"


def test_download_recording_skips_missing_url(tmp_path):
    with patch("src.download._get_api_key", return_value="fake-key"):
        dl = XenoCantoDownloader(output_dir=str(tmp_path), api_key="fake-key")

    result = dl.download_recording({"id": "1", "file": ""}, tmp_path)
    assert result is None


def test_list_downloaded_empty(tmp_path):
    with patch("src.download._get_api_key", return_value="fake-key"):
        dl = XenoCantoDownloader(output_dir=str(tmp_path), api_key="fake-key")

    assert dl.list_downloaded() == {}
