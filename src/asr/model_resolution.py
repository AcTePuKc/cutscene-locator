"""Deterministic model resolution and optional download plumbing."""

from __future__ import annotations

import json
import os
import platform
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import import_module
from pathlib import Path
from typing import Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .config import ASRConfig

_FASTER_WHISPER_MODEL_REPOS: dict[str, str] = {
    "tiny": "Systran/faster-whisper-tiny",
    "base": "Systran/faster-whisper-base",
    "small": "Systran/faster-whisper-small",
}
_DEFAULT_MODEL_REVISION = "default"


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


def _sanitize_repo_id(repo_id: str) -> str:
    """Sanitize repo id for deterministic cache directory names."""

    return repo_id.replace("/", "--")


def _is_model_present(path: Path) -> bool:
    if path.is_file():
        return True
    if path.is_dir():
        return any(path.iterdir())
    return False


def _validate_model_repo_snapshot(*, backend_name: str, model_dir: Path) -> None:
    """Validate backend-specific required files for model directories."""

    if backend_name == "faster-whisper":
        required_paths = [
            model_dir / "config.json",
            model_dir / "model.bin",
        ]
        missing = [path.name for path in required_paths if not path.exists()]

        tokenizer_assets = [
            "tokenizer.json",
            "vocabulary.json",
            "vocabulary.txt",
            "vocab.json",
            "vocab.txt",
        ]
        has_tokenizer_asset = any((model_dir / filename).exists() for filename in tokenizer_assets)
        if not has_tokenizer_asset:
            missing.append(
                "one of tokenizer.json, vocabulary.json, vocabulary.txt, vocab.json, vocab.txt"
            )

        if missing:
            missing_display = ", ".join(sorted(missing))
            found_files = ", ".join(sorted(path.name for path in model_dir.iterdir())) if model_dir.is_dir() else "<none>"
            raise ModelResolutionError(
                "Resolved faster-whisper model is missing required files: "
                f"{missing_display}. Expected a CTranslate2-converted Whisper model directory. "
                f"Found files: {found_files}"
            )




def _is_windows_platform() -> bool:
    return platform.system().lower().startswith("win")


def _resolve_progress_enabled(download_progress: bool | None) -> bool:
    if download_progress is not None:
        return download_progress
    return not _is_windows_platform()


def _apply_progress_env(*, progress_enabled: bool) -> None:
    if progress_enabled:
        os.environ.pop("HF_HUB_DISABLE_PROGRESS_BARS", None)
    else:
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"


def _snapshot_download_with_progress(
    snapshot_download: Callable[..., object],
    *,
    repo_id: str,
    local_dir: str,
    revision: str | None,
    progress_enabled: bool,
) -> None:
    kwargs = {
        "repo_id": repo_id,
        "local_dir": local_dir,
        "revision": revision,
    }
    if not progress_enabled:
        kwargs["local_dir_use_symlinks"] = False
        kwargs["tqdm_class"] = None

    try:
        snapshot_download(**kwargs)
    except TypeError:
        kwargs.pop("local_dir_use_symlinks", None)
        kwargs.pop("tqdm_class", None)
        snapshot_download(**kwargs)

def _default_download_url(*, backend_name: str, model_size: str) -> str:
    if backend_name != "mock" or model_size != "tiny":
        raise ModelResolutionError(
            "Auto-download URL is not configured for this backend/model. "
            "Provide --model-path explicitly or set CUTSCENE_LOCATOR_MODEL_DOWNLOAD_URL."
        )
    return "https://example.com/cutscene-locator/mock-tiny-model.bin"


def _resolve_download_url(*, backend_name: str, model_size: str) -> str:
    if backend_name == "faster-whisper":
        raise ModelResolutionError(
            "faster-whisper auto-download uses Hugging Face snapshots and does not accept "
            "CUTSCENE_LOCATOR_MODEL_DOWNLOAD_URL."
        )

    env_url = os.environ.get("CUTSCENE_LOCATOR_MODEL_DOWNLOAD_URL")
    if env_url:
        return env_url
    return _default_download_url(backend_name=backend_name, model_size=model_size)


def _download_faster_whisper_snapshot(
    *,
    model_size: str,
    model_dir: Path,
    progress_callback: Callable[[float], None] | None,
    cancel_check: Callable[[], bool] | None,
    download_progress: bool | None,
) -> None:
    repo_id = _FASTER_WHISPER_MODEL_REPOS.get(model_size)
    if repo_id is None:
        supported = ", ".join(sorted(_FASTER_WHISPER_MODEL_REPOS))
        raise ModelResolutionError(
            f"Unsupported faster-whisper auto-download size '{model_size}'. "
            f"Expected one of: {supported}."
        )

    if cancel_check is not None and cancel_check():
        raise ModelResolutionError("Model download cancelled before start.")

    try:
        huggingface_hub_module = import_module("huggingface_hub")
    except ModuleNotFoundError as exc:
        raise ModelResolutionError(
            "faster-whisper auto-download requires huggingface_hub. "
            "Install dependencies or provide --model-path."
        ) from exc

    snapshot_download = getattr(huggingface_hub_module, "snapshot_download", None)
    if snapshot_download is None:
        raise ModelResolutionError(
            "huggingface_hub is installed but snapshot_download is unavailable. "
            "Upgrade huggingface_hub or provide --model-path."
        )

    progress_enabled = _resolve_progress_enabled(download_progress)
    _apply_progress_env(progress_enabled=progress_enabled)

    if progress_callback is not None:
        progress_callback(0.0)

    try:
        _snapshot_download_with_progress(
            snapshot_download,
            repo_id=repo_id,
            local_dir=str(model_dir),
            revision=None,
            progress_enabled=progress_enabled,
        )
    except Exception as exc:  # pragma: no cover - message validated through tests
        raise ModelResolutionError(
            "Failed to auto-download faster-whisper model from Hugging Face. "
            f"Repository: '{repo_id}'. Check connectivity/permissions or provide --model-path."
        ) from exc

    if cancel_check is not None and cancel_check():
        raise ModelResolutionError("Model download cancelled by caller.")

    if progress_callback is not None:
        progress_callback(100.0)


def _download_model_id_snapshot(
    *,
    backend_name: str,
    model_id: str,
    revision: str | None,
    cache_dir: Path,
    progress_callback: Callable[[float], None] | None,
    cancel_check: Callable[[], bool] | None,
    download_progress: bool | None,
) -> Path:
    if cancel_check is not None and cancel_check():
        raise ModelResolutionError("Model download cancelled before start.")

    try:
        huggingface_hub_module = import_module("huggingface_hub")
    except ModuleNotFoundError as exc:
        raise ModelResolutionError(
            "Model-id download requires huggingface_hub. "
            "Install dependencies or provide --model-path."
        ) from exc

    snapshot_download = getattr(huggingface_hub_module, "snapshot_download", None)
    if snapshot_download is None:
        raise ModelResolutionError(
            "huggingface_hub is installed but snapshot_download is unavailable. "
            "Upgrade huggingface_hub or provide --model-path."
        )

    resolved_revision = revision if revision else _DEFAULT_MODEL_REVISION
    model_dir = cache_dir / backend_name / _sanitize_repo_id(model_id) / resolved_revision
    model_dir.mkdir(parents=True, exist_ok=True)

    progress_enabled = _resolve_progress_enabled(download_progress)
    _apply_progress_env(progress_enabled=progress_enabled)

    if progress_callback is not None:
        progress_callback(0.0)

    try:
        _snapshot_download_with_progress(
            snapshot_download,
            repo_id=model_id,
            revision=revision,
            local_dir=str(model_dir),
            progress_enabled=progress_enabled,
        )
    except Exception as exc:  # pragma: no cover - validated in tests
        raise ModelResolutionError(
            "Failed to download model-id snapshot from Hugging Face. "
            f"Repository: '{model_id}' revision='{resolved_revision}'. "
            "Check connectivity/permissions or provide --model-path."
        ) from exc

    if cancel_check is not None and cancel_check():
        raise ModelResolutionError("Model download cancelled by caller.")

    _validate_model_repo_snapshot(backend_name=backend_name, model_dir=model_dir)

    if progress_callback is not None:
        progress_callback(100.0)

    return model_dir


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
    download_progress: bool | None,
) -> Path:
    """Download model into cache directory with resume support and metadata."""

    model_dir = cache_dir / backend_name / model_size
    model_dir.mkdir(parents=True, exist_ok=True)
    model_file = model_dir / "model.bin"
    partial_file = model_dir / "model.bin.part"

    if backend_name == "faster-whisper":
        _download_faster_whisper_snapshot(
            model_size=model_size,
            model_dir=model_dir,
            progress_callback=progress_callback,
            cancel_check=cancel_check,
            download_progress=download_progress,
        )
        _write_metadata_file(
            model_dir=model_dir,
            metadata=DownloadMetadata(
                backend=backend_name,
                model_size=model_size,
                downloaded_at=datetime.now(timezone.utc).isoformat(),
                version=None,
            ),
        )
        return model_dir

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
            _validate_model_repo_snapshot(backend_name=config.backend_name, model_dir=explicit_path)
            return explicit_path
        raise ModelResolutionError(
            f"Model not found at explicit --model-path '{explicit_path}'."
        )

    if config.model_id is not None:
        resolved_revision = config.revision if config.revision else _DEFAULT_MODEL_REVISION
        cached_model_id_dir = cache_dir / config.backend_name / _sanitize_repo_id(config.model_id) / resolved_revision
        if _is_model_present(cached_model_id_dir):
            _validate_model_repo_snapshot(backend_name=config.backend_name, model_dir=cached_model_id_dir)
            if config.log_callback is not None:
                config.log_callback("model resolution: cached hit")
                config.log_callback(f"model resolution: resolved cache directory: {cached_model_id_dir}")
            return cached_model_id_dir

        if config.log_callback is not None:
            config.log_callback("model resolution: downloading")
            config.log_callback(f"model resolution: resolved cache directory: {cached_model_id_dir}")

        return _download_model_id_snapshot(
            backend_name=config.backend_name,
            model_id=config.model_id,
            revision=config.revision,
            cache_dir=cache_dir,
            progress_callback=config.progress_callback,
            cancel_check=config.cancel_check,
            download_progress=config.download_progress,
        )

    requested_size = config.auto_download

    local_models_root = Path("models") / config.backend_name
    if requested_size is not None:
        local_models_for_size = local_models_root / requested_size
        if _is_model_present(local_models_for_size):
            _validate_model_repo_snapshot(backend_name=config.backend_name, model_dir=local_models_for_size)
            return local_models_for_size

    if _is_model_present(local_models_root):
        _validate_model_repo_snapshot(backend_name=config.backend_name, model_dir=local_models_root)
        return local_models_root

    cache_backend_dir = cache_dir / config.backend_name
    if requested_size is not None:
        cached_model_for_size = cache_backend_dir / requested_size
        if _is_model_present(cached_model_for_size):
            _validate_model_repo_snapshot(backend_name=config.backend_name, model_dir=cached_model_for_size)
            return cached_model_for_size

    cached_model_dir = cache_backend_dir / "tiny"
    if _is_model_present(cached_model_dir):
        _validate_model_repo_snapshot(backend_name=config.backend_name, model_dir=cached_model_dir)
        return cached_model_dir

    if requested_size in {"tiny", "base", "small"}:
        downloaded_model = download_model_to_cache(
            backend_name=config.backend_name,
            model_size=requested_size,
            cache_dir=cache_dir,
            progress_callback=config.progress_callback,
            cancel_check=config.cancel_check,
            download_progress=config.download_progress,
        )
        _validate_model_repo_snapshot(backend_name=config.backend_name, model_dir=downloaded_model)
        return downloaded_model

    raise ModelResolutionError(
        "Model could not be resolved. Checked --model-path, local models/, and cache. "
        "To allow download, pass --auto-download tiny or use --model-id."
    )
