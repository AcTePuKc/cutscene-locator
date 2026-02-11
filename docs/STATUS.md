# Project Status – cutscene-locator

This file is the **single source of truth** for project progress.

If a task is implemented but NOT recorded here, it is treated as **not done**.

All agents must update this file when completing or modifying tasks.

---

## Legend

- [ ] not started
- [~] in progress
- [x] completed
- (stub) placeholder only
- (test) covered by tests

---

## Milestone 0 – Repository scaffolding (DONE)

- [x] Repository structure created
- [x] Documentation skeleton added
- [x] CLI-first scope defined
- [x] ffmpeg dependency defined
- [x] Data contracts fixed

---

## Milestone 1 – Core pipeline (CLI, mock ASR)

### Ingest / Preprocessing

- [x] ffmpeg availability check
- [x] Audio extraction from video via ffmpeg
- [x] Audio normalization to canonical WAV
- [x] Chunking implementation (time-based)
- [x] Temporary file management (`out/_tmp/`)

### Script handling

- [x] TSV/CSV parser
- [x] Required column validation (`id`, `original`)
- [x] Optional column passthrough (`translation`, `file`, etc.)
- [x] Script normalization pipeline

### ASR (mock backend)

- [x] ASR backend interface
- [x] Mock ASR backend (JSON input)
- [x] ASR segment validation (timestamps, text)

### Matching

- [x] Text normalization (shared ASR/script)
- [x] Fuzzy matching implementation
- [x] Ranked candidate selection
- [x] Confidence scoring
- [x] Low-confidence flagging

### Scene reconstruction

- [x] Scene gap logic
- [x] Chronological stitching
- [x] Scene ID generation
- [x] Overlap tolerance

### Exports

- [x] `matches.csv` writer
- [x] `scenes.json` writer
- [x] `subs_qa.srt` writer
- [x] `subs_target.srt` writer (optional)

### CLI

- [x] Argument parsing
- [x] Help and version output
- [x] Exit codes
- [x] Verbose logging
- [x] Error handling
- [x] Windows progress-thread guard + verbose stage markers (`cli.py`, `src/match/engine.py`, `tests/test_cli.py`, `tests/test_matching.py`)

---

## Milestone 1.5 – ASR Architecture Lock

### Backend architecture

- [x] Final ASR backend interface definition
- [x] Backend registry system
- [x] Backend discovery via CLI flag
- [x] Backend capability metadata (supports_word_timestamps, supports_alignment, etc.)

### Model resolution strategy

- [x] models/ directory convention
- [x] --model-path explicit override
- [x] --auto-download <model-size> optional flag
- [x] Optional auto-download tiny implemented
- [x] Deterministic model cache directory
- [x] No silent fallback policy
- [x] Generic model-id download/cache support (`src/asr/model_resolution.py`, `src/asr/config.py`, `cli.py`)
- [x] Windows-safe model-id cache fast-path + progress control (`src/asr/model_resolution.py`, `src/asr/config.py`, `cli.py`, `tests/test_model_resolution.py`, `tests/test_cli.py`)
- [x] faster-whisper progress-off NullProgressBar shim + HF progress separation (`src/asr/faster_whisper_backend.py`, `src/asr/model_resolution.py`, `tests/test_faster_whisper_backend.py`, `tests/test_model_resolution.py`)

### Device handling

- [x] --device cpu/cuda/auto
- [x] Deterministic device selection
- [x] Clear error if requested device unavailable

### UI-readiness hooks

- [x] Progress callback interface in backend config
- [x] Cancellation-ready interface (future-safe)
- [x] Backend metadata reporting (model name, version, device)

---

## Milestone 2 – Real ASR backends (post-MVP)

### Whisper / Qwen / others

- [ ] ASR backend adapter (generic)
- [ ] WhisperX backend
- [x] Qwen ASR backend
- [x]  faster-whisper backend (pilot)
- [x]  faster-whisper auto-download mapping (tiny/base/small → HF repo)
- [x]  CUDA enablement notes + detection (ctranslate2/whisper backend)
- [ ] Timestamp normalization across backends

### Advanced alignment

- [ ] Forced alignment support
- [ ] Word-level timestamps (optional)
- [ ] Cross-chunk continuity handling

---

## Milestone 3 – Quality and tooling

- [ ] Unit test coverage for core modules
- [ ] Test fixtures (audio + script)
- [ ] Integration test (optional, gated)
- [ ] Performance profiling (large scripts)
- [ ] Config file support (optional)

---

## Milestone 4 – UI (out of scope for now)

- [ ] PySide6 UI scaffold
- [ ] CLI → UI adapter
- [ ] Scene review interface
- [ ] Subtitle preview player

---

## Known risks / open questions

- GPU requirements for long-form ASR
- Diarization accuracy in overlapping speech
- Handling repeated short lines at scale
- Performance on very large scripts (10k+ lines)

---

## Change log (manual)

> Keep this short. One line per meaningful change.

- YYYY-MM-DD – Initial STATUS.md created
- 2026-02-11 – Milestone 1 Phase 1 CLI skeleton + ffmpeg preflight completed (`cli.py`, `tests/test_cli.py`).
- 2026-02-11 – Milestone 1 Phase 2 script ingestion + normalization completed (`src/ingest/script_parser.py`, `src/match/normalization.py`, `cli.py`, `tests/test_script_parser.py`, `tests/test_cli.py`).

- 2026-02-11 – Milestone 1 Phase 3 ASR abstraction + mock backend + validation completed (`src/asr/backends.py`, `src/asr/__init__.py`, `cli.py`, `tests/test_asr.py`, `tests/test_cli.py`, `tests/fixtures/mock_asr_valid.json`, `tests/fixtures/mock_asr_invalid_start_end.json`).
- 2026-02-11 – Milestone 1 Phase 4 matching engine completed (`src/match/engine.py`, `src/match/__init__.py`, `cli.py`, `tests/test_matching.py`, `tests/test_cli.py`).
- 2026-02-11 – Milestone 1 Phase 5 scene reconstruction completed (`src/scene/reconstruct.py`, `src/scene/__init__.py`, `cli.py`, `tests/test_scene.py`, `tests/test_cli.py`).

- 2026-02-11 – Milestone 1 Phase 6 export writers + CLI full pipeline wiring completed (`src/export/writers.py`, `src/export/__init__.py`, `cli.py`, `tests/test_exports.py`, `tests/test_cli.py`).

- 2026-02-11 – Milestone 1 Phase 7 ffmpeg preprocessing + deterministic `_tmp` workspace + chunking completed (`src/ingest/preprocess.py`, `src/ingest/__init__.py`, `cli.py`, `tests/test_preprocess.py`, `tests/test_cli.py`).

- 2026-02-11 – Milestone 1.5 Phase 1 ASR registry + CLI model/device/config plumbing completed (`src/asr/registry.py`, `src/asr/config.py`, `src/asr/__init__.py`, `cli.py`, `tests/test_asr_registry.py`, `tests/test_cli.py`).

- 2026-02-11 – Milestone 1.5 Phase 2 model cache + auto-download plumbing + UI-ready hooks completed (`src/asr/model_resolution.py`, `src/asr/config.py`, `src/asr/__init__.py`, `cli.py`, `tests/test_model_resolution.py`, `tests/test_cli.py`, `tests/test_asr_registry.py`).

- 2026-02-11 – Milestone 1.5 Phase 3 ASR interface finalization + deterministic device resolution + metadata reporting completed (`src/asr/base.py`, `src/asr/backends.py`, `src/asr/device.py`, `src/asr/__init__.py`, `src/match/engine.py`, `cli.py`, `tests/test_asr.py`, `tests/test_cli.py`, `tests/test_matching.py`, `tests/test_exports.py`, `tests/fixtures/mock_asr_valid.json`, `tests/fixtures/mock_asr_invalid_start_end.json`, `docs/Data-contracts.md`).

- 2026-02-11 – Milestone 2 faster-whisper auto-download mapping completed (`src/asr/model_resolution.py`, `tests/test_model_resolution.py`, `docs/STATUS.md`).

- 2026-02-11 – Milestone 2 CUDA detection + guidance and verbose timings completed (`src/asr/device.py`, `src/asr/faster_whisper_backend.py`, `src/asr/__init__.py`, `cli.py`, `docs/CUDA.md`, `tests/test_asr.py`, `tests/test_cli.py`, `tests/test_faster_whisper_backend.py`, `docs/STATUS.md`).

- 2026-02-11 – Milestone 2 Qwen ASR backend completed (`src/asr/qwen3_asr_backend.py`, `src/asr/registry.py`, `src/asr/__init__.py`, `cli.py`, `pyproject.toml`, `tests/test_asr_registry.py`, `docs/STATUS.md`).

- 2026-02-11 – Generic model-id download/cache support completed (`src/asr/model_resolution.py`, `src/asr/config.py`, `src/asr/faster_whisper_backend.py`, `cli.py`, `tests/test_model_resolution.py`, `tests/test_cli.py`, `tests/test_faster_whisper_backend.py`, `docs/CLI.md`, `docs/ASR_ARCHITECTURE.md`, `docs/STATUS.md`).

- 2026-02-11 – Faster-whisper model validation unified in model resolution (removed backend duplicate checks, expanded tokenizer asset acceptance, improved error file listing) (`src/asr/model_resolution.py`, `src/asr/faster_whisper_backend.py`, `tests/test_model_resolution.py`, `tests/test_faster_whisper_backend.py`, `docs/STATUS.md`).

- 2026-02-11 – Fixed Windows model-id cached hit behavior by bypassing snapshot_download, added progress control and verbose model-resolution logs (`src/asr/model_resolution.py`, `src/asr/config.py`, `cli.py`, `tests/test_model_resolution.py`, `tests/test_cli.py`, `docs/CLI.md`, `docs/STATUS.md`).

- 2026-02-11 – Disabled tqdm/HF progress env by default on Windows at CLI startup, added per-stage verbose markers, and switched matching to periodic verbose progress logging (`cli.py`, `src/match/engine.py`, `tests/test_cli.py`, `tests/test_matching.py`, `docs/STATUS.md`).

- 2026-02-11 – Faster-whisper progress-off fix: replaced None pbar with backend-local NullProgressBar patch and decoupled HF snapshot progress disabling from faster-whisper transcribe progress (`src/asr/faster_whisper_backend.py`, `src/asr/model_resolution.py`, `tests/test_faster_whisper_backend.py`, `tests/test_model_resolution.py`, `docs/STATUS.md`).
