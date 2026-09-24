"""Reproducibly download and cache the original UCI data file."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen

FALLBACK_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"
UCI_PAGE = "https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data"
EXPECTED_SHA256 = "b21f3d81db8071257d5ff1deaeba1fd4303b62712e6fcc9715c7a86202cb5871"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_dataset(destination: Path, force: bool = False) -> Path:
    """Download german.data atomically; keep an existing cache unless forced."""
    destination = Path(destination)
    if destination.exists() and not force:
        actual_hash = sha256(destination)
        if actual_hash != EXPECTED_SHA256:
            raise ValueError(
                f"Cached dataset checksum mismatch at {destination}: "
                f"expected {EXPECTED_SHA256}, found {actual_hash}. "
                "Delete the file or rerun with --force-download."
            )
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    errors = []
    for url in (FALLBACK_URL,):
        try:
            with urlopen(url, timeout=30) as response:
                payload = response.read()
            if not payload.strip():
                raise ValueError("download was empty")
            temporary = destination.with_suffix(destination.suffix + ".tmp")
            temporary.write_bytes(payload)
            downloaded_hash = sha256(temporary)
            if downloaded_hash != EXPECTED_SHA256:
                temporary.unlink(missing_ok=True)
                raise ValueError(
                    f"download checksum mismatch: expected {EXPECTED_SHA256}, "
                    f"found {downloaded_hash}"
                )
            temporary.replace(destination)
            metadata = {
                "source_url": url,
                "dataset_page": UCI_PAGE,
                "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
                "sha256": sha256(destination),
                "bytes": destination.stat().st_size,
            }
            destination.with_suffix(".metadata.json").write_text(
                json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
            )
            return destination
        except Exception as exc:  # each official endpoint is attempted
            errors.append(f"{url}: {exc}")
    raise RuntimeError("Unable to download the UCI dataset:\n" + "\n".join(errors))
