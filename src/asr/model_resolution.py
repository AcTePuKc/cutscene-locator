"""Deterministic model resolution and optional download plumbing."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .config import ASRConfig


class ModelResolutionError(ValueError):
    """Raised when model path resolution or download fails."""


@dataclass(frozen=True)
class DownloadMetadata:
    """Metadata recorded for downloaded model assets."""

    backend: str
    model_size: str
    downloaded_at: str
    version: str | None


def resolve_model_cache_dir() -> Path:
    """Return deterministic model cache directory and ensure it exists."""

    if os.name == "nt":
        base = Path(os.environ.get("USERPROFILE", Path.home()))
    else:
        base = Path.home()

    cache_dir = base / ".cutscene-locator" / "models"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _is_model_present(path: Path) -> bool:
    if path.is_file():
        return True
    if path.is_dir():
        return any(path.iterdir())
    return False


def _default_download_url(*, backend_name: str, model_size: str) -> str:
    if backend_name != "mock" or model_size != "tiny":
        raise ModelResolutionError(
            "Auto-download URL is not configured for this backend/model. "
            "Provide --model-path explicitly or set CUTSCENE_LOCATOR_MODEL_DOWNLOAD_URL."
        )
    return "https://example.com/cutscene-locator/mock-tiny-model.bin"


def _resolve_download_url(*, backend_name: str, model_size: str) -> str:
    env_url = os.environ.get("CUTSCENE_LOCATOR_MODEL_DOWNLOAD_URL")
    if env_url:
        return env_url
    return _default_download_url(backend_name=backend_name, model_size=model_size)


def _write_metadata_file(*, model_dir: Path, metadata: DownloadMetadata) -> None:
    metadata_path = model_dir / "model_metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "backend": metadata.backend,
                "model_size": metadata.model_size,
                "downloaded_at": metadata.downloaded_at,
                "version": metadata.version,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def download_model_to_cache(
    *,
    backend_name: str,
    model_size: str,
    cache_dir: Path,
    progress_callback: Callable[[float], None] | None,
    cancel_check: Callable[[], bool] | None,
) -> Path:
    """Download model into cache directory with resume support and metadata."""

    model_dir = cache_dir / backend_name / model_size
    model_dir.mkdir(parents=True, exist_ok=True)
    model_file = model_dir / "model.bin"
    partial_file = model_dir / "model.bin.part"

    if cancel_check is not None and cancel_check():
        raise ModelResolutionError("Model download cancelled before start.")

    url = _resolve_download_url(backend_name=backend_name, model_size=model_size)
    existing_size = partial_file.stat().st_size if partial_file.exists() else 0

    headers: dict[str, str] = {}
    if existing_size > 0:
        headers["Range"] = f"bytes={existing_size}-"

    request = Request(url=url, headers=headers)

    try:
        with urlopen(request, timeout=60) as response:
            total_length_header = response.headers.get("Content-Length")
            total_length = int(total_length_header) if total_length_header is not None else None
            total_with_existing = (
                (total_length + existing_size) if total_length is not None else None
            )

            downloaded = existing_size
            if progress_callback is not None and total_with_existing:
                progress_callback((downloaded / total_with_existing) * 100.0)

            with partial_file.open("ab") as handle:
                while True:
                    if cancel_check is not None and cancel_check():
                        raise ModelResolutionError("Model download cancelled by caller.")

                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break

                    handle.write(chunk)
                    downloaded += len(chunk)

                    if progress_callback is not None and total_with_existing:
                        progress_callback(min(100.0, (downloaded / total_with_existing) * 100.0))

    except HTTPError as exc:
        raise ModelResolutionError(
            f"Model download failed with HTTP {exc.code} for URL '{url}'. "
            "Use --model-path to provide a local model or set CUTSCENE_LOCATOR_MODEL_DOWNLOAD_URL."
        ) from exc
    except URLError as exc:
        raise ModelResolutionError(
            f"Model download failed for URL '{url}': {exc.reason}. "
            "Check network access or use --model-path."
        ) from exc

    partial_size = partial_file.stat().st_size if partial_file.exists() else 0
    if partial_size == 0:
        raise ModelResolutionError(
            "Model download produced an empty file. "
            "Verify download URL contents or use --model-path."
        )

    partial_file.replace(model_file)

    _write_metadata_file(
        model_dir=model_dir,
        metadata=DownloadMetadata(
            backend=backend_name,
            model_size=model_size,
            downloaded_at=datetime.now(timezone.utc).isoformat(),
            version=None,
        ),
    )

    if progress_callback is not None:
        progress_callback(100.0)

    return model_dir


def resolve_model_path(config: ASRConfig) -> Path:
    """Resolve model path with deterministic precedence and optional auto-download."""

    cache_dir = resolve_model_cache_dir()

    if config.model_path is not None:
        explicit_path = config.model_path
        if _is_model_present(explicit_path):
            return explicit_path
        raise ModelResolutionError(
            f"Model not found at explicit --model-path '{explicit_path}'."
        )

    local_models_dir = Path("models") / config.backend_name
    if _is_model_present(local_models_dir):
        return local_models_dir

    cached_model_dir = cache_dir / config.backend_name / "tiny"
    if _is_model_present(cached_model_dir):
        return cached_model_dir

    if config.auto_download == "tiny":
        return download_model_to_cache(
            backend_name=config.backend_name,
            model_size="tiny",
            cache_dir=cache_dir,
            progress_callback=config.progress_callback,
            cancel_check=config.cancel_check,
        )

    raise ModelResolutionError(
        "Model could not be resolved. Checked --model-path, local models/, and cache. "
        "To allow download, pass --auto-download tiny."
    )
