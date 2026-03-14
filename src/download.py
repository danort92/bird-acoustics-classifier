"""
src/download.py — Xeno-canto audio downloader for bird species classification.

Downloads .mp3 recordings from the Xeno-canto API and organises them under
data/raw/<species_name>/.  Compatible with local environments and Google Colab.

Usage:
    from src.download import XenoCantoDownloader

    downloader = XenoCantoDownloader(output_dir="data/raw")
    species = ["Turdus merula", "Erithacus rubecula"]
    downloader.download_species(species, max_per_species=50)

Environment variable:
    XENO_CANTO_API_KEY — optional; the module works without an API key for the
                         public Xeno-canto v2 endpoint.  Set it if you have a
                         registered key to get higher rate limits.
"""

import os
import time
import getpass
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

XENO_CANTO_API_V2 = "https://xeno-canto.org/api/2/recordings"


def _sanitise_name(name: str) -> str:
    """Convert a species name to a safe directory name (lowercase, underscores)."""
    return name.strip().lower().replace(" ", "_")


def _get_api_key() -> Optional[str]:
    """
    Return the Xeno-canto API key.

    Resolution order:
    1. Environment variable ``XENO_CANTO_API_KEY``
    2. Interactive prompt (hidden input so the key is not echoed)
    3. None — proceed without a key (public endpoints still work)
    """
    key = os.environ.get("XENO_CANTO_API_KEY")
    if key:
        logger.info("API key loaded from environment variable.")
        return key

    # Interactive prompt — works in terminal and Colab
    try:
        key = getpass.getpass(
            "Xeno-canto API key (press Enter to skip): "
        ).strip()
    except (EOFError, OSError):
        # Non-interactive environment (e.g. CI): skip silently
        key = ""

    if key:
        logger.info("API key provided interactively.")
        return key

    logger.info("No API key provided — using public access.")
    return None


class XenoCantoDownloader:
    """
    Download bird audio recordings from the Xeno-canto API.

    Parameters
    ----------
    output_dir : str | Path
        Root directory where files are saved.  Sub-directories are created
        automatically per species, e.g. ``output_dir/turdus_merula/``.
    api_key : str | None
        Xeno-canto API key.  When *None* the module tries the environment
        variable ``XENO_CANTO_API_KEY`` and then an interactive prompt.
    request_delay : float
        Seconds to wait between HTTP requests (be polite to the API).
    """

    def __init__(
        self,
        output_dir: str = "data/raw",
        api_key: Optional[str] = None,
        request_delay: float = 0.5,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.api_key = api_key or _get_api_key()
        self.request_delay = request_delay
        self._session = requests.Session()
        if self.api_key:
            self._session.headers.update({"X-API-Key": self.api_key})

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search_species(
        self,
        species: str,
        quality: str = "A",
        max_results: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Search Xeno-canto for recordings of *species*.

        Parameters
        ----------
        species : str
            Scientific name, e.g. ``"Turdus merula"``.
        quality : str
            Minimum recording quality rating (A–E).  Default ``"A"``.
        max_results : int
            Maximum number of recordings to return (across all pages).

        Returns
        -------
        list[dict]
            Raw recording metadata dicts from the API.
        """
        recordings: List[Dict[str, Any]] = []
        page = 1

        while len(recordings) < max_results:
            params: Dict[str, Any] = {
                "query": f"{species} q:{quality}",
                "page": page,
            }
            logger.info(
                "Searching '%s' (quality>=%s) — page %d …", species, quality, page
            )
            try:
                resp = self._session.get(
                    XENO_CANTO_API_V2, params=params, timeout=30
                )
                resp.raise_for_status()
            except requests.RequestException as exc:
                logger.error("API request failed: %s", exc)
                break

            data = resp.json()
            page_recordings = data.get("recordings", [])
            if not page_recordings:
                break

            recordings.extend(page_recordings)
            num_pages = int(data.get("numPages", 1))
            if page >= num_pages:
                break
            page += 1
            time.sleep(self.request_delay)

        return recordings[:max_results]

    def download_recording(
        self,
        recording: Dict[str, Any],
        species_dir: Path,
    ) -> Optional[Path]:
        """
        Download a single recording and save it to *species_dir*.

        Parameters
        ----------
        recording : dict
            Metadata dict from the Xeno-canto API.
        species_dir : Path
            Target directory (created if absent).

        Returns
        -------
        Path | None
            Path of the saved file, or *None* on failure.
        """
        species_dir.mkdir(parents=True, exist_ok=True)

        rec_id = recording.get("id", "unknown")
        file_url = recording.get("file", "")
        if not file_url:
            logger.warning("Recording %s has no file URL — skipping.", rec_id)
            return None

        # Ensure the URL has a scheme
        if file_url.startswith("//"):
            file_url = "https:" + file_url

        filename = f"xc{rec_id}.mp3"
        dest = species_dir / filename

        if dest.exists():
            logger.debug("Already downloaded: %s", dest)
            return dest

        try:
            resp = self._session.get(file_url, timeout=60, stream=True)
            resp.raise_for_status()
            with dest.open("wb") as fh:
                for chunk in resp.iter_content(chunk_size=8192):
                    fh.write(chunk)
            logger.info("Saved %s", dest)
            return dest
        except requests.RequestException as exc:
            logger.error("Failed to download recording %s: %s", rec_id, exc)
            return None

    def download_species(
        self,
        species_list: List[str],
        max_per_species: int = 50,
        quality: str = "A",
    ) -> Dict[str, List[Path]]:
        """
        Download recordings for a list of species.

        Parameters
        ----------
        species_list : list[str]
            Scientific names, e.g. ``["Turdus merula", "Parus major"]``.
        max_per_species : int
            Maximum recordings to download per species.
        quality : str
            Minimum Xeno-canto quality rating (A–E).

        Returns
        -------
        dict[str, list[Path]]
            Mapping from species name to list of downloaded file paths.
        """
        results: Dict[str, List[Path]] = {}

        for species in species_list:
            logger.info("=== Processing: %s ===", species)
            safe_name = _sanitise_name(species)
            species_dir = self.output_dir / safe_name

            recordings = self.search_species(
                species, quality=quality, max_results=max_per_species
            )
            logger.info(
                "Found %d recording(s) for '%s'.", len(recordings), species
            )

            downloaded: List[Path] = []
            for rec in recordings:
                path = self.download_recording(rec, species_dir)
                if path:
                    downloaded.append(path)
                time.sleep(self.request_delay)

            results[species] = downloaded
            logger.info(
                "Downloaded %d/%d for '%s'.",
                len(downloaded),
                len(recordings),
                species,
            )

        return results

    def list_downloaded(self) -> Dict[str, List[Path]]:
        """
        Return a dict of already-downloaded files grouped by species directory.
        """
        result: Dict[str, List[Path]] = {}
        for species_dir in sorted(self.output_dir.iterdir()):
            if species_dir.is_dir():
                mp3_files = sorted(species_dir.glob("*.mp3"))
                if mp3_files:
                    result[species_dir.name] = mp3_files
        return result
