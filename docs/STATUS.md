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
- [x] Deterministic matching candidate reduction (token/first-word/length-bucket indexes + optional quick-filter + monotonic window) (`src/match/engine.py`, `cli.py`, `tests/test_matching.py`, `tests/test_cli.py`, `docs/CLI.md`)
- [x] ASR segmentation consistency knobs (`--asr-vad-filter`, `--asr-merge-short-segments`) with deterministic short-segment merge in faster-whisper backend (`src/asr/config.py`, `src/asr/faster_whisper_backend.py`, `cli.py`, `tests/test_faster_whisper_backend.py`, `docs/CLI.md`)
- [x] Exposed safe faster-whisper decode knobs in CLI/config (`--asr-language`, `--asr-beam-size`, `--asr-temperature`, `--asr-best-of`, `--asr-no-speech-threshold`, `--asr-logprob-threshold`) with deterministic validation and backend kwarg gating (`cli.py`, `src/asr/config.py`, `src/asr/faster_whisper_backend.py`, `src/asr/asr_worker.py`, `tests/test_cli.py`, `tests/test_faster_whisper_backend.py`, `tests/test_asr_worker.py`)
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
- [x] faster-whisper progress/runtime separation from HF snapshot download controls (`src/asr/faster_whisper_backend.py`, `src/asr/model_resolution.py`, `tests/test_faster_whisper_backend.py`, `tests/test_model_resolution.py`)
- [x] faster-whisper CUDA-safe progress path + Windows subprocess isolation + `--compute-type` (`cli.py`, `src/asr/faster_whisper_backend.py`, `src/asr/asr_worker.py`, `src/asr/config.py`, `tests/test_faster_whisper_backend.py`, `tests/test_cli.py`, `docs/CLI.md`, `docs/Integration-issues.md`)
- [x] faster-whisper 1.2.1 transcribe compatibility fix (removed unsupported `progress` kwarg while keeping CUDA subprocess isolation) (`src/asr/faster_whisper_backend.py`, `tests/test_faster_whisper_backend.py`, `docs/STATUS.md`)
- [x] faster-whisper transcribe kwarg signature guard for version-safe execution (`src/asr/faster_whisper_backend.py`, `tests/test_faster_whisper_backend.py`, `docs/STATUS.md`)
- [x] Worker JSON → validated `ASRResult` parsing in parent CLI + typed metadata access (`cli.py`, `src/asr/backends.py`, `src/asr/__init__.py`, `tests/test_asr.py`)
- [x] Windows CUDA worker hardening: removed worker `--progress`, set CUDA-only tqdm/HF env in worker pre-import path, and improved native abort diagnostics (`src/asr/asr_worker.py`, `src/asr/faster_whisper_backend.py`, `cli.py`, `tests/test_asr_worker.py`, `tests/test_faster_whisper_backend.py`, `tests/test_cli.py`, `docs/Integration-issues.md`)
- [x] faster-whisper CUDA preflight diagnostics in CLI (ctranslate2 version/device count + compute-type mitigation guidance) (`cli.py`, `tests/test_cli.py`, `docs/CLI.md`, `docs/Integration-issues.md`)
- [x] ASR worker verbose environment dump + in-worker minimal WhisperModel preflight transcribe (`src/asr/asr_worker.py`, `tests/test_asr_worker.py`)
- [x] ASR worker preflight made non-fatal and CUDA-skipped; CPU/auto preflight uses `vad_filter=False` and samples only first segment (`src/asr/asr_worker.py`, `tests/test_asr_worker.py`)
- [x] CUDA worker crash tracing + tqdm monitor-thread disable + unbuffered faulthandler worker spawn (`src/asr/asr_worker.py`, `src/asr/faster_whisper_backend.py`, `cli.py`, `tests/test_asr_worker.py`, `tests/test_cli.py`)
- [x] CUDA worker import isolation from package root + CUDA segment-consumption markers/forced materialization + conservative CUDA transcribe kwargs (`src/asr/asr_worker.py`, `src/asr/faster_whisper_backend.py`, `tests/test_asr_worker.py`, `tests/test_faster_whisper_backend.py`)
- [x] qwen3-asr snapshot artifact validation for explicit path, model-id cache/download resolution with deterministic errors (`src/asr/model_resolution.py`, `tests/test_model_resolution.py`)

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
- [x] Declared-vs-enabled backend status metadata + disabled-backend CLI guidance (`src/asr/registry.py`, `cli.py`, `tests/test_asr_registry.py`, `tests/test_cli.py`, `docs/CLI.md`)
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
- 2026-02-12 – Docs sync: updated CLI backend names/dependency extras and `--model-id` compatibility caveats; clarified explicit unchecked Milestone 2 next-work items (`docs/CLI.md`, `docs/STATUS.md`).
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

- 2026-02-11 – Windows CUDA crash isolation for faster-whisper via subprocess worker, CUDA-safe progress handling, ASR init/transcribe stage markers, and CLI `--compute-type` support (`cli.py`, `src/asr/faster_whisper_backend.py`, `src/asr/asr_worker.py`, `src/asr/config.py`, `tests/test_faster_whisper_backend.py`, `tests/test_cli.py`, `docs/CLI.md`, `docs/Integration-issues.md`, `docs/STATUS.md`).

- 2026-02-11 – Fixed ASR subprocess typing/contract consistency by parsing worker JSON into validated ASRResult in parent CLI and tightening JSON helper typing (`cli.py`, `src/asr/backends.py`, `src/asr/__init__.py`, `tests/test_asr.py`, `docs/STATUS.md`).

- 2026-02-11 – Fixed faster-whisper 1.2.1 worker crash by removing unsupported `progress` transcribe kwarg while preserving CUDA subprocess isolation (`src/asr/faster_whisper_backend.py`, `tests/test_faster_whisper_backend.py`, `docs/STATUS.md`).

- 2026-02-11 – Added transcribe kwarg signature filtering guard in faster-whisper backend to prevent runtime kwarg incompatibilities across versions (`src/asr/faster_whisper_backend.py`, `tests/test_faster_whisper_backend.py`, `docs/STATUS.md`).
- 2026-02-11 – Added ASR worker verbose runtime dump (`sys.executable`, `sys.path`, `ctranslate2.__file__`, env subset) plus minimal in-worker WhisperModel transcribe preflight before backend execution (`src/asr/asr_worker.py`, `tests/test_asr_worker.py`, `docs/STATUS.md`).

- 2026-02-11 – Hardened Windows CUDA faster-whisper worker by removing worker progress arg, setting CUDA tqdm/HF env pre-import, tightening worker/parent ASRResult typing, and improving native-abort diagnostics (`src/asr/asr_worker.py`, `src/asr/faster_whisper_backend.py`, `cli.py`, `tests/test_asr_worker.py`, `tests/test_faster_whisper_backend.py`, `tests/test_cli.py`, `docs/Integration-issues.md`, `docs/STATUS.md`).
- 2026-02-11 – Added faster-whisper CUDA preflight diagnostics (ctranslate2 version/device count/device/compute-type) and strengthened CUDA abort guidance with float32-first + compatibility matrix note (`cli.py`, `tests/test_cli.py`, `docs/CLI.md`, `docs/Integration-issues.md`, `docs/STATUS.md`).

- 2026-02-12 – Stabilized Windows CUDA faster-whisper worker path by forcing transcribe `vad_filter=False`, explicitly passing `language=None` when unset, logging exact transcribe kwargs via callback, and setting worker tqdm/HF progress env guards for all devices (`src/asr/faster_whisper_backend.py`, `src/asr/asr_worker.py`, `tests/test_faster_whisper_backend.py`, `tests/test_asr_worker.py`, `docs/STATUS.md`).

- 2026-02-12 – Fixed Windows CUDA preflight instability by making worker minimal preflight non-fatal, skipping it on CUDA, and constraining CPU/auto preflight to `vad_filter=False` with first-segment sampling (`src/asr/asr_worker.py`, `tests/test_asr_worker.py`, `docs/STATUS.md`).
- 2026-02-12 – Added CUDA worker step flush markers with per-step failure logs, disabled tqdm monitor thread in CUDA worker runtime setup, and spawned ASR worker with `-u -X faulthandler` for improved native-abort diagnostics (`src/asr/asr_worker.py`, `cli.py`, `tests/test_asr_worker.py`, `tests/test_cli.py`, `docs/STATUS.md`).


- 2026-02-12 – Isolated ASR worker imports from `src` package root (direct `src.asr.*` module imports), added CUDA segment-consumption start/end markers with forced list materialization, and set conservative CUDA transcribe decode knobs (`beam_size=1`, `best_of=1`, `temperature=0`) for crash localization/mitigation (`src/asr/asr_worker.py`, `src/asr/faster_whisper_backend.py`, `tests/test_asr_worker.py`, `tests/test_faster_whisper_backend.py`, `docs/STATUS.md`).

- 2026-02-12 – Added deterministic matching optimizations (indexed candidate reduction, two-pass quick filter, optional monotonic window) and ASR segmentation consistency controls (`--asr-vad-filter`, `--asr-merge-short-segments`) with short-segment merge support (`src/match/engine.py`, `src/asr/config.py`, `src/asr/faster_whisper_backend.py`, `cli.py`, `tests/test_matching.py`, `tests/test_faster_whisper_backend.py`, `tests/test_cli.py`, `docs/CLI.md`, `docs/STATUS.md`).

- 2026-02-12 – Exposed safe faster-whisper transcribe knobs in CLI/ASR config and worker path, with validation and transcribe-kwarg gating (including no `best_of` when `temperature=0`) plus CLI/backend tests (`cli.py`, `src/asr/config.py`, `src/asr/faster_whisper_backend.py`, `src/asr/asr_worker.py`, `tests/test_cli.py`, `tests/test_faster_whisper_backend.py`, `tests/test_asr_worker.py`, `docs/STATUS.md`).

- 2026-02-12 – Tightened ASR typing contract by making validated ASR segment keys required, adding explicit protocol stub return, and fixing mock-meta narrowing to remove Optional member access without changing runtime validation behavior (`src/asr/base.py`, `src/asr/backends.py`, `docs/STATUS.md`).

- 2026-02-12 – Replaced custom match similarity scoring with RapidFuzz (`fuzz.WRatio` primary, `fuzz.partial_ratio` fallback) while preserving deterministic candidate indexing/filtering and added a unit test that monkeypatches `fuzz.WRatio` to prove RapidFuzz invocation (`src/match/engine.py`, `tests/test_matching.py`, `docs/STATUS.md`).
- 2026-02-12 – Added declared-vs-enabled ASR backend status metadata with missing-dependency reporting, actionable disabled-backend CLI errors, and backend availability docs/tests updates (`src/asr/registry.py`, `src/asr/__init__.py`, `cli.py`, `tests/test_asr_registry.py`, `tests/test_cli.py`, `docs/CLI.md`, `docs/STATUS.md`).
- 2026-02-12 – Added explicit forced-alignment path (separate `src/align/` contract/validator), marked alignment-capable backend metadata, and blocked alignment backends from ASR-only CLI mode with capability/contract tests (`src/align/base.py`, `src/align/validation.py`, `src/align/__init__.py`, `src/asr/registry.py`, `cli.py`, `tests/test_alignment.py`, `tests/test_cli.py`, `tests/test_asr_registry.py`, `docs/Data-contracts.md`, `docs/Integration.md`, `docs/STATUS.md`).

- 2026-02-12 – Added qwen3-asr Hugging Face snapshot validation in model resolution (config/tokenizer/weights checks) covering explicit --model-path, model-id cache hits, and model-id downloads, with deterministic error messaging tests (`src/asr/model_resolution.py`, `tests/test_model_resolution.py`, `docs/STATUS.md`).
