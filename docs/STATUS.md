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
- Completed milestone entries with file markers must include at least one implementation path (`src/` or `cli.py`) and one test path (`tests/`)

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
- [x] Worker failure diagnostics preserve clean non-verbose failures and emit deterministic labeled stdout/stderr blocks in verbose mode (`cli.py`, `tests/test_cli.py`, `docs/CLI.md`)
- [x] ASR preflight-only CLI mode (`--asr-preflight-only`) for backend availability/model resolution/device probe sanity checks with deterministic JSON output (`cli.py`, `tests/test_cli.py`, `docs/CLI.md`)
- [x] Qwen3 readiness QA coverage: deterministic `qwen3-asr` preflight JSON smoke assertion + optional env-gated local init-only loader smoke (no inference, offline-by-default CI), including from_pretrained-without-`device` and explicit post-load device transfer assertions (`tests/test_cli.py`, `tests/test_qwen3_asr_backend.py`, `docs/CLI.md`)
- [x] Windows progress-thread guard + verbose stage markers (`cli.py`, `src/match/engine.py`, `tests/test_cli.py`, `tests/test_matching.py`)

---

## Milestone 1.5 – ASR Architecture Lock

### Backend architecture

- [x] Final ASR backend interface definition
- [x] Backend registry system
- [x] Backend discovery via CLI flag
- [x] Backend capability metadata (supports_word_timestamps, supports_alignment, etc.)
- [x] Backend status API distinguishes unknown vs declared-disabled backends (`name`, `enabled`, `missing_dependencies`, `reason`) and preserves enabled-only discovery via `list_backends()` (`src/asr/registry.py`, `cli.py`, `tests/test_asr_registry.py`, `tests/test_cli.py`, `docs/CLI.md`)
- [x] Docs/backend-name consistency guard: declared registry backend names must be present in `docs/CLI.md` and `docs/STATUS.md` (`tests/test_docs_consistency.py`, `docs/CLI.md`, `docs/STATUS.md`)

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
- [x] ASR worker backend made explicit via required `--asr-backend` arg with deterministic runtime dispatch (faster-whisper/qwen3-asr/whisperx/vibevoice) and backend-specific preflight behavior (`src/asr/asr_worker.py`, `cli.py`, `tests/test_asr_worker.py`, `tests/test_cli.py`)
- [x] CUDA worker crash tracing + tqdm monitor-thread disable + unbuffered faulthandler worker spawn (`src/asr/asr_worker.py`, `src/asr/faster_whisper_backend.py`, `cli.py`, `tests/test_asr_worker.py`, `tests/test_cli.py`)
- [x] CUDA worker import isolation from package root + CUDA segment-consumption markers/forced materialization + conservative CUDA transcribe kwargs (`src/asr/asr_worker.py`, `src/asr/faster_whisper_backend.py`, `tests/test_asr_worker.py`, `tests/test_faster_whisper_backend.py`)
- [x] qwen3-asr snapshot artifact validation for explicit path/model-id cache/download resolution with deterministic errors, requiring core artifacts while treating processor/preprocessor config as optional (`src/asr/model_resolution.py`, `tests/test_model_resolution.py`, `docs/CLI.md`, `docs/Data-contracts.md`)

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

Implementation sequence note (dependency order; execute serially to avoid architecture drift and keep `docs/Integration.md` + `docs/Data-contracts.md` contracts authoritative):
1. Generic adapter
2. Timestamp normalization
3. WhisperX
4. Forced alignment path
5. VibeVoice

Contract notes:
- Qwen3-ASR model variants (`0.6B` / `1.7B`) are handled through the same `qwen3-asr` backend contract/configuration path, not separate backend classes.
- Forced-aligner models are alignment-mode backends and are not equivalent to ASR-only transcript-generation models.

- [x] ASR backend adapter (generic) (`src/asr/adapters.py`, `src/asr/base.py`, `src/asr/registry.py`, `src/asr/__init__.py`, `cli.py`, `tests/test_asr_registry.py`, `tests/test_cli.py`, `docs/ASR_ARCHITECTURE.md`)
- [x] ASR adapter static return-path typing fix (Protocol stubs use explicit ellipsis bodies) + callable adapter contract regression test (`src/asr/adapters.py`, `tests/test_asr_registry.py`, `docs/STATUS.md`)
- [x] WhisperX backend (`src/asr/whisperx_backend.py`, `src/asr/registry.py`, `src/asr/adapters.py`, `src/asr/model_resolution.py`, `pyproject.toml`, `tests/test_whisperx_backend.py`, `tests/test_asr_registry.py`)
- [x] VibeVoice backend via shared adapter path (`src/asr/vibevoice_backend.py`, `src/asr/registry.py`, `src/asr/adapters.py`, `src/asr/__init__.py`, `pyproject.toml`, `tests/test_vibevoice_backend.py`, `tests/test_asr_registry.py`, `tests/test_cli.py`, `docs/CLI.md`, `docs/Integration-issues.md`, `docs/ASR_ARCHITECTURE.md`)
- [x] Qwen ASR backend
- [x]  faster-whisper backend (pilot)
- [x] Declared-vs-enabled backend status metadata + disabled-backend CLI guidance (`src/asr/registry.py`, `cli.py`, `tests/test_asr_registry.py`, `tests/test_cli.py`, `docs/CLI.md`)
- [x]  faster-whisper auto-download mapping (tiny/base/small → HF repo)
- [x]  CUDA enablement notes + detection (ctranslate2/whisper backend)
- [x] Timestamp normalization across backends (`src/asr/timestamp_normalization.py`, `src/asr/faster_whisper_backend.py`, `src/asr/qwen3_asr_backend.py`, `tests/test_asr.py`, `tests/test_faster_whisper_backend.py`, `tests/test_qwen3_asr_backend.py`, `tests/fixtures/asr_timestamp_edges_faster_whisper.json`, `tests/fixtures/asr_timestamp_edges_qwen3.json`)
- [x] Deterministic backend readiness audit matrix + verification checklist (qwen3-asr/whisperx/vibevoice) including registry/install-state diagnostics, model artifact layout validation, and backend-specific CUDA preflight reason reporting (`src/asr/readiness.py`, `src/asr/model_resolution.py`, `scripts/verify_backend_readiness.py`, `tests/test_backend_readiness.py`, `tests/test_asr_registry.py`, `docs/CLI.md`, `docs/Integration-issues.md`, `docs/Integration.md`)
- [x] Qwen3-ASR deterministic compatibility smoke checks (pipeline call shape + `return_timestamps=True` + strict chunk timestamp tuple normalization assumptions) and docs matrix/troubleshooting separation for artifact-vs-runtime init failures (`src/asr/qwen3_asr_backend.py`, `tests/test_qwen3_asr_backend.py`, `docs/CLI.md`, `docs/Integration-issues.md`)
- [x] Qwen extras/importability alignment: pinned project-managed `asr_qwen3` extra to explicit `qwen-asr` compatible range, kept registry/readiness checks on `qwen_asr` import target, and tightened dependency error/install guidance with import-target tests (`pyproject.toml`, `src/asr/qwen3_asr_backend.py`, `src/asr/registry.py`, `src/asr/readiness.py`, `tests/test_asr_registry.py`, `tests/test_qwen3_asr_backend.py`, `docs/CLI.md`)
- [x] Qwen3-ASR transcribe kwarg filtering excludes `temperature` to prevent qwen_asr/Transformers invalid-generation-flag warnings while keeping deterministic unsupported-option handling for other decode knobs (`src/asr/qwen3_asr_backend.py`, `tests/test_qwen3_asr_backend.py`, `docs/STATUS.md`)

### Advanced alignment

- [x] Forced alignment support (`src/align/base.py`, `src/align/validation.py`, `src/align/qwen3_forced_aligner.py`, `src/asr/registry.py`, `cli.py`, `tests/test_alignment.py`, `tests/test_qwen3_forced_aligner.py`, `tests/test_asr_registry.py`)
- [x] Word-level timestamps (optional) (`src/align/qwen3_forced_aligner.py`, `tests/test_qwen3_forced_aligner.py`, `docs/Data-contracts.md`)
- [x] Cross-chunk continuity handling (`src/asr/adapters.py`, `cli.py`, `tests/test_adapters.py`, `docs/Data-contracts.md`)
- [x] Cross-chunk continuity typed return narrowing (`ASRSegment`/`ASRMeta` explicit reconstruction) to keep static `ASRResult` proof after boundary merges (`src/asr/adapters.py`, `tests/test_adapters.py`)

---

## Milestone 3 – Quality and tooling

- [x] CLI docs/parser flag parity consistency test + declared-disabled backend definition consistency assertion (`tests/test_docs_consistency.py`, `docs/CLI.md`, `cli.py`)
- [ ] Unit test coverage for core modules
- [x] Test fixtures (audio + script) (`src/asr/timestamp_normalization.py`, `cli.py`, `tests/fixtures/script_integration_sample.tsv`, `tests/fixtures/asr_normalized_faster_whisper.json`, `tests/fixtures/asr_normalized_qwen3_asr.json`, `tests/fixtures/asr_normalized_whisperx_vibevoice.json`)
- [x] Integration test (optional, gated) (`cli.py`, `src/export/writers.py`, `tests/test_integration_pipeline.py`)
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

- 2026-02-13 – Fixed qwen3-asr backend test path expectation to be OS-deterministic by asserting resolved `Path` equivalence for `from_pretrained` model path (instead of raw separator-sensitive string equality), preserving intent without changing runtime formatting (`tests/test_qwen3_asr_backend.py`, `docs/STATUS.md`).
- 2026-02-13 – Added deterministic worker-failure diagnostics gating in CLI ASR subprocess handling: non-verbose mode keeps concise errors without dumping worker streams, while `--verbose` now prints labeled worker stdout/stderr blocks in deterministic order before raising; added regression tests for both paths and documented troubleshooting behavior (`cli.py`, `tests/test_cli.py`, `docs/CLI.md`, `docs/STATUS.md`).
- 2026-02-13 – Removed `temperature` from qwen3-asr transcribe/generation kwargs path to avoid qwen_asr/Transformers invalid-generation-flag warnings while preserving deterministic unsupported-option filtering for other non-qwen decode knobs; added regression coverage asserting qwen transcribe is called without `temperature` (`src/asr/qwen3_asr_backend.py`, `tests/test_qwen3_asr_backend.py`, `docs/STATUS.md`).

- 2026-02-12 – Ensured project-managed qwen install flow matches runtime importability by constraining `asr_qwen3` to explicit `qwen-asr` compatibility range, keeping registry/readiness dependency checks on `qwen_asr`, improving qwen backend missing-dependency error guidance (`cutscene-locator[asr_qwen3]` + `import qwen_asr`), and adding dependency-focused registry/backend tests (`pyproject.toml`, `src/asr/qwen3_asr_backend.py`, `tests/test_asr_registry.py`, `tests/test_qwen3_asr_backend.py`, `docs/CLI.md`, `docs/STATUS.md`).
- 2026-02-12 – Added lightweight qwen QA checks that stay inference-free by default: reinforced `--asr-preflight-only --asr-backend qwen3-asr` single-line JSON contract coverage, added explicit no-`device` `from_pretrained` kwarg-shape assertions in qwen backend/init smoke, introduced fully env-gated local runtime smoke (`CUTSCENE_QWEN3_RUNTIME_SMOKE`, `CUTSCENE_QWEN3_MODEL_PATH`, `CUTSCENE_QWEN3_RUNTIME_AUDIO`), and documented env vars + expected smoke output contracts in CLI preflight troubleshooting notes (`tests/test_cli.py`, `tests/test_qwen3_asr_backend.py`, `docs/CLI.md`, `docs/STATUS.md`).
- 2026-02-12 – Standardized no-autoswitch CUDA wording to use explicit manual rerun with `--device cpu` in CLI readiness matrix language, clarified declared-but-disabled backend docs wording (including experimental-backend-disabled-by-default), kept CLI disabled-backend runtime errors deterministic/actionable with install-extra hints only for missing optional dependencies, and extended docs/CLI regression coverage for no-autoswitch phrasing and disabled-backend definitions (`docs/CLI.md`, `cli.py`, `tests/test_docs_consistency.py`, `tests/test_cli.py`, `docs/STATUS.md`).
- 2026-02-12 – Unified qwen3-asr artifact contract wording across model-resolution/readiness/docs, keeping only core artifacts required (`config.json`, tokenizer assets, `tokenizer_config.json`, model weights) while treating `processor_config.json`/`preprocessor_config.json` as optional; retained regression coverage for pass-without-processor-configs and missing-core failures (`src/asr/model_resolution.py`, `src/asr/readiness.py`, `docs/CLI.md`, `docs/Data-contracts.md`, `tests/test_model_resolution.py`, `docs/STATUS.md`).

- 2026-02-12 – Simplified qwen3-asr `dtype` resolver/loader kwargs to remove redundant `None`-guarding and redundant branching: `_resolve_dtype` now returns `str` passthrough deterministically and `from_pretrained` init kwargs are constructed directly with `dtype`, preserving no-`device` init and post-load device-move behavior (`src/asr/qwen3_asr_backend.py`, `docs/STATUS.md`).

- 2026-02-12 – Updated qwen3-asr runtime init to match real API usage by removing `device=` from `Qwen3ASRModel.from_pretrained(...)`, preserving deterministic `dtype` mapping, enforcing explicit post-load device transfer (`.model.to(...)`/`.to(...)`) with deterministic loader/API-mismatch errors when unsupported, and expanded backend + readiness smoke tests for no-device-kwarg and transfer behavior (`src/asr/qwen3_asr_backend.py`, `tests/test_qwen3_asr_backend.py`, `tests/test_cli.py`, `docs/STATUS.md`).

- 2026-02-12 – Tightened no-silent-fallback wording to require manual rerun with `--device cpu`, clarified the authoritative definition/causes for a "declared but disabled backend", kept dependency-gated CLI backend errors deterministic/actionable with install-extras only for missing-optional-dependency cases, and expanded docs consistency checks to prevent fallback-wording regressions (`docs/CLI.md`, `cli.py`, `tests/test_docs_consistency.py`, `docs/STATUS.md`).

- 2026-02-12 – Made qwen3-asr snapshot contract explicitly treat `processor_config.json`/`preprocessor_config.json` as optional (never required for pass), aligned readiness preconditions wording, and updated model-resolution tests to validate pass-without-processor-configs plus core-artifact-only failure expectations (`src/asr/readiness.py`, `docs/CLI.md`, `tests/test_model_resolution.py`, `docs/STATUS.md`).

- 2026-02-12 – Added deterministic qwen3-asr compatibility smoke checks (mocked `transformers.pipeline` call-shape assertions, explicit `return_timestamps=True`, and strict `chunks` timestamp tuple contract checks) and documented a qwen3 variant compatibility matrix plus troubleshooting guidance for "model resolves but pipeline init fails" to separate artifact issues from runtime/API issues (`tests/test_qwen3_asr_backend.py`, `docs/CLI.md`, `docs/Integration-issues.md`, `docs/STATUS.md`).
- 2026-02-12 – Clarified `--asr-preflight-only` invocation parity in CLI docs with explicit installed/source command examples (including Windows `py .\cli.py`), documented identical single-line JSON stdout contract across invocation modes, added PowerShell copy/line-continuation artifact note, and extended docs consistency coverage for this requirement (`docs/CLI.md`, `tests/test_docs_consistency.py`, `docs/STATUS.md`).
- 2026-02-12 – Added authoritative CLI docs definition for "Declared but disabled" backend diagnostics (missing optional dependencies, feature-flag-disabled backend, experimental backend disabled by default), clarified that dependency-gated errors may include install-extra hints, and added docs consistency coverage for the new definition (`docs/CLI.md`, `tests/test_docs_consistency.py`, `docs/STATUS.md`).
- 2026-02-12 – Replaced misleading CUDA "fallback" terminology with explicit CPU rerun wording in backend readiness/CUDA guidance docs, and added a docs consistency test that blocks reintroduction of CPU fallback phrasing in that context (`docs/CLI.md`, `docs/Integration-issues.md`, `tests/test_docs_consistency.py`, `docs/STATUS.md`).
- 2026-02-12 – Verified backend CUDA probe source by actual runtime API usage (`faster-whisper`→`ctranslate2`; `qwen3-asr`/`whisperx`/`vibevoice`→`torch`), exposed probe label in `--asr-preflight-only` JSON (`device.cuda_probe_label`), aligned CLI backend matrix/docs, and added regression tests tying backend selection to declared probe label (`cli.py`, `src/asr/device.py`, `tests/test_cli.py`, `tests/test_asr.py`, `docs/CLI.md`, `docs/STATUS.md`).

- 2026-02-12 – Relaxed qwen3-asr snapshot contract to require only core deterministic artifacts (`config.json`, tokenizer assets, `tokenizer_config.json`, model weights) while treating `processor_config.json`/`preprocessor_config.json` as optional; added regression coverage for config-absent valid layouts and kept deterministic missing-core failures (`src/asr/model_resolution.py`, `tests/test_model_resolution.py`, `docs/CLI.md`, `docs/Data-contracts.md`, `docs/STATUS.md`).

- 2026-02-12 – Added deterministic backend readiness matrix/checklist for `qwen3-asr`, `whisperx`, and `vibevoice` (dependency importability, registry enabled-state validation, model artifact layout validation, and backend-appropriate CUDA preflight reason reporting) with a no-inference verification script and registry/readiness regression tests (`src/asr/readiness.py`, `src/asr/model_resolution.py`, `scripts/verify_backend_readiness.py`, `tests/test_backend_readiness.py`, `tests/test_asr_registry.py`, `docs/CLI.md`, `docs/Integration.md`, `docs/Integration-issues.md`, `docs/STATUS.md`).

- 2026-02-12 – Added offline, dependency-light gated integration-style deterministic pipeline test with representative normalized backend fixtures (faster-whisper, qwen3-asr, whisperx/vibevoice-style timestamps), validating stable `matches.csv` ordering, stable `scenes.json` boundaries, and no fabricated timestamps (`tests/test_integration_pipeline.py`, `tests/fixtures/script_integration_sample.tsv`, `tests/fixtures/asr_normalized_faster_whisper.json`, `tests/fixtures/asr_normalized_qwen3_asr.json`, `tests/fixtures/asr_normalized_whisperx_vibevoice.json`, `docs/STATUS.md`).

- 2026-02-12 – Added deterministic STATUS governance checks to enforce milestone checkbox/file-marker consistency and changelog-vs-milestone non-contradiction, including explicit contributor guidance to update milestone checkbox lines (not changelog-only notes) (`tests/test_docs_consistency.py`, `docs/STATUS.md`).

- 2026-02-12 – Added deterministic cross-chunk continuity handling in ASR adapter bridge (absolute timestamp conversion via chunk offsets + boundary duplicate/split/tiny-fragment merges), covered by unit tests and documented tie-break semantics (`src/asr/adapters.py`, `cli.py`, `tests/test_adapters.py`, `docs/Data-contracts.md`, `docs/STATUS.md`).

- 2026-02-12 – Brought `docs/CLI.md` into parser parity by documenting every `build_parser()` flag with defaults/valid ranges/backend applicability, and added deterministic docs consistency test to fail on CLI/docs drift (`docs/CLI.md`, `tests/test_docs_consistency.py`, `cli.py`, `docs/STATUS.md`).

> Keep this short. One line per meaningful change.

- 2026-02-12 – Added qwen3-asr readiness QA coverage with deterministic single-line preflight JSON payload assertions and an optional env-gated local-model init-only smoke test (`CUTSCENE_QWEN3_INIT_SMOKE=1`, `CUTSCENE_QWEN3_MODEL_PATH=...`) that does not run inference; documented usage in CLI preflight troubleshooting notes (`tests/test_cli.py`, `docs/CLI.md`, `docs/STATUS.md`).

- 2026-02-12 – Updated qwen3-asr pipeline initialization failure messaging to match the core model artifact contract (config/tokenizer/tokenizer_config/weights; processor/preprocessor optional) and include concise runtime debugging hints; added backend error-message regression coverage (`src/asr/qwen3_asr_backend.py`, `tests/test_qwen3_asr_backend.py`, `docs/STATUS.md`).

- YYYY-MM-DD – Initial STATUS.md created
- 2026-02-12 – Fixed static ASR adapter return-path typing by replacing docstring-only `ASRAdapter` protocol methods with explicit ellipsis stubs (signature-preserving), and added adapter-registry coverage to assert each registered adapter exposes callable `transcribe` (`src/asr/adapters.py`, `tests/test_asr_registry.py`, `docs/STATUS.md`).
- 2026-02-12 – Fixed adapter callback typing contract for injected faster-whisper helper callbacks by introducing explicit keyword-only callback protocols, aligning adapter invocation semantics, and adding CLI/adapter dispatch tests for callback injection paths (`src/asr/adapters.py`, `cli.py`, `tests/test_cli.py`, `tests/test_asr_registry.py`, `docs/STATUS.md`).
- 2026-02-12 – Clarified Milestone 2 stable implementation sequence (generic adapter → timestamp normalization → WhisperX → forced alignment path → VibeVoice), documented Qwen3-ASR shared-backend variant handling (`0.6B`/`1.7B`), and reaffirmed that forced-aligner models are alignment-mode only per Integration/Data-contracts authority (`docs/STATUS.md`).
- 2026-02-12 – STATUS consistency audit: synchronized Milestone 2/Advanced alignment checkboxes with implemented backend features (including word-level alignment timestamps), and added docs consistency coverage to fail when changelog-complete backend milestones are left unchecked (`docs/STATUS.md`, `tests/test_docs_consistency.py`).
- 2026-02-12 – Added VibeVoice ASR backend behind existing adapter/registry contracts with deterministic timestamp normalization and dependency-gated CLI guidance; documented backend mode limits and install extras; added backend/registry/CLI tests and implementation-order gate notes (`src/asr/vibevoice_backend.py`, `src/asr/registry.py`, `src/asr/adapters.py`, `src/asr/__init__.py`, `pyproject.toml`, `tests/test_vibevoice_backend.py`, `tests/test_asr_registry.py`, `tests/test_cli.py`, `docs/CLI.md`, `docs/Integration-issues.md`, `docs/ASR_ARCHITECTURE.md`, `docs/STATUS.md`).
- 2026-02-12 – Added WhisperX ASR backend behind existing backend registry/adapter contracts with optional extras + dependency-gated status reporting, wired deterministic model resolution compatibility (`--model-path`/`--model-id`) and canonical timestamp normalization into ASRResult mapping, and added registry/normalization tests (`src/asr/whisperx_backend.py`, `src/asr/registry.py`, `src/asr/adapters.py`, `src/asr/model_resolution.py`, `src/asr/__init__.py`, `pyproject.toml`, `tests/test_whisperx_backend.py`, `tests/test_asr_registry.py`, `docs/STATUS.md`).
- 2026-02-12 – Extended backend-contract qwen3-asr snapshot validation to require transformers pipeline artifacts (`tokenizer_config.json` and processor/preprocessor config) for deterministic model-id/model-path resolution across valid 0.6B and 1.7B layouts; improved qwen pipeline init guidance and added valid/invalid layout tests (`src/asr/model_resolution.py`, `src/asr/qwen3_asr_backend.py`, `tests/test_model_resolution.py`, `docs/CLI.md`, `docs/STATUS.md`).
- 2026-02-12 – Added dedicated `qwen3-forced-aligner` backend with explicit alignment I/O contract and validation, kept `qwen3-asr` on transcript semantics, and enforced capability-based CLI gating/tests for aligner-only backends (`src/align/base.py`, `src/align/qwen3_forced_aligner.py`, `src/align/validation.py`, `src/asr/registry.py`, `cli.py`, `tests/test_qwen3_forced_aligner.py`, `tests/test_asr_registry.py`, `tests/test_cli.py`, `docs/Data-contracts.md`, `docs/Integration.md`, `docs/STATUS.md`).
- 2026-02-12 – Hardened qwen3-asr model snapshot validation with deterministic required-artifact checks across explicit --model-path, model-id cache hits, and post-download model-id paths; added valid/invalid snapshot layout tests (`src/asr/model_resolution.py`, `tests/test_model_resolution.py`, `docs/STATUS.md`).
- 2026-02-12 – Refined backend discovery UX by adding structured backend `reason` status, exposing declared backend listing, preserving enabled-only lookup semantics, and improving disabled-backend CLI install guidance/messages with tests/docs updates (`src/asr/registry.py`, `src/asr/__init__.py`, `cli.py`, `tests/test_asr_registry.py`, `tests/test_cli.py`, `docs/CLI.md`, `docs/STATUS.md`).
- 2026-02-12 – Docs sync: updated CLI backend names/dependency extras and `--model-id` compatibility caveats; clarified explicit unchecked Milestone 2 next-work items (`docs/CLI.md`, `docs/STATUS.md`).
- 2026-02-12 – Refined backend timestamp normalization to preserve ASR text payloads, enforce deterministic timing normalization before validation, document overlap/equal-boundary semantics, and expand ASR edge tests for overlaps/equal boundaries/invalid values (`src/asr/timestamp_normalization.py`, `src/asr/faster_whisper_backend.py`, `src/asr/qwen3_asr_backend.py`, `tests/test_asr.py`, `tests/test_faster_whisper_backend.py`, `tests/test_qwen3_asr_backend.py`, `tests/fixtures/asr_timestamp_edges_faster_whisper.json`, `tests/fixtures/asr_timestamp_edges_qwen3.json`, `docs/Data-contracts.md`, `docs/STATUS.md`).
- 2026-02-12 – Added deterministic timestamp normalization contract for real ASR backends (numeric finite seconds, non-negative times, half-up rounding to 6 decimals, stable start/end ordering, zero-length drop, and strict rejection of pathological timings) and covered edge fixtures for empty/overlap/float precision/pathological cases (`src/asr/timestamp_normalization.py`, `src/asr/faster_whisper_backend.py`, `src/asr/qwen3_asr_backend.py`, `tests/test_asr.py`, `tests/test_faster_whisper_backend.py`, `tests/test_qwen3_asr_backend.py`, `tests/fixtures/asr_timestamp_edges_faster_whisper.json`, `tests/fixtures/asr_timestamp_edges_qwen3.json`, `docs/STATUS.md`).

- 2026-02-12 – Tightened ASR adapter protocol typing by making `_BaseASRAdapter` an explicit abstract contract, aligning `get_asr_adapter` return typing with concrete adapter instances, and adding focused adapter registry/contract regression tests plus dispatch context type usage updates (`src/asr/adapters.py`, `tests/test_adapters.py`, `tests/test_asr_registry.py`, `docs/STATUS.md`).

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

- 2026-02-12 – Added ASR adapter abstraction + adapter registry dispatch in CLI (standardized model-loading/transcribe plumbing, backend-kwargs filtering hook, canonical output normalization path), plus explicit capability validation (`supports_segment_timestamps` required, alignment disallowed for ASR mode) and deterministic adapter-dispatch tests/docs updates (`src/asr/adapters.py`, `src/asr/registry.py`, `src/asr/__init__.py`, `cli.py`, `tests/test_asr_registry.py`, `tests/test_cli.py`, `docs/ASR_ARCHITECTURE.md`, `docs/STATUS.md`).

- 2026-02-12 – Clarified ASR-vs-alignment model family support with a backend/mode matrix in CLI docs, documented deterministic loaded-when-requested model resolution/no-fallback policy, and added a docs consistency test to prevent drift between registry backend names and docs (`docs/CLI.md`, `tests/test_docs_consistency.py`, `docs/STATUS.md`).

- 2026-02-12 – Expanded shared ASR adapter dispatch with registry capability preflight in adapter execution (single CLI dispatch path, no-fallback failure propagation, and deterministic adapter invocation tests) (`src/asr/adapters.py`, `src/asr/__init__.py`, `cli.py`, `tests/test_asr_registry.py`, `tests/test_cli.py`, `docs/STATUS.md`).
- 2026-02-12 – Added additional forced-alignment contract validation coverage for malformed spans and missing timestamps while keeping capability-gating behavior explicit (`tests/test_alignment.py`, `docs/STATUS.md`).

- 2026-02-12 – Refined ctranslate2 CUDA probe typing and deterministic non-numeric fallback handling in device resolution, with focused tests for numeric/string/object/exception getter outcomes (`src/asr/device.py`, `tests/test_asr.py`, `docs/STATUS.md`).

- [x] Typed snapshot artifact schema definitions for model validation (faster-whisper/whisperx/qwen3-asr) to keep deterministic error messages while improving static typing (`src/asr/model_resolution.py`, `tests/test_model_resolution.py`)

- 2026-02-12 – Made model snapshot artifact schemas strongly typed and extended schema validation tests across faster-whisper/whisperx/qwen3-asr while preserving runtime error text format (`src/asr/model_resolution.py`, `tests/test_model_resolution.py`, `docs/STATUS.md`).

- [x] Backend-specific CUDA probe routing in ASR adapter dispatch with centralized torch/ctranslate2 selection and deterministic reason labels (`src/asr/device.py`, `src/asr/adapters.py`, `src/asr/qwen3_asr_backend.py`, `src/asr/whisperx_backend.py`, `src/asr/vibevoice_backend.py`, `tests/test_asr.py`, `tests/test_asr_registry.py`)
- [x] Authoritative backend→CUDA-probe mapping codified from runtime API usage, exposed in ASR preflight output (`device.cuda_probe_label`), and synchronized in CLI docs/backend tests (`src/asr/device.py`, `cli.py`, `tests/test_asr.py`, `tests/test_cli.py`, `docs/CLI.md`)
- [x] Deterministic ASR-vs-alignment mode/contract gating guidance: qwen3 model-family separation documented and forced-aligner backend identifiers rejected from ASR-only path with explicit alignment-contract messaging (`src/asr/registry.py`, `cli.py`, `tests/test_cli.py`, `tests/test_asr_registry.py`, `docs/Integration.md`, `docs/Data-contracts.md`, `docs/CLI.md`)

- 2026-02-12 – Removed deprecated Hugging Face `local_dir_use_symlinks` usage from snapshot download wrapper using signature-based kwargs gating, preserved deterministic HF progress env toggling, and expanded model-resolution progress on/off tests (`src/asr/model_resolution.py`, `tests/test_model_resolution.py`, `docs/STATUS.md`).
- 2026-02-13 – Clarified qwen3 ASR-vs-forced-alignment flow separation across integration/data-contract/CLI docs and tightened runtime ASR-path gating to reject alignment backends with explicit `reference_spans[]` guidance; added CLI/registry validation tests for mode-contract gating (`docs/Integration.md`, `docs/Data-contracts.md`, `docs/CLI.md`, `src/asr/registry.py`, `cli.py`, `tests/test_cli.py`, `tests/test_asr_registry.py`, `docs/STATUS.md`).

- 2026-02-12 – Added lightweight `--asr-preflight-only` CLI path that reuses backend registry/capability checks, model resolution, and backend-specific device probe logic, prints deterministic structured JSON for QA logs, and exits before ingest/transcribe/match/export; added success/failure CLI tests and docs usage notes (`cli.py`, `tests/test_cli.py`, `docs/CLI.md`, `docs/STATUS.md`).

- 2026-02-12 – Tightened `apply_cross_chunk_continuity` return typing by rebuilding merged segments as explicit `ASRSegment` payloads and metadata as explicit `ASRMeta`, then returning a typed `ASRResult`; added boundary-merge regression coverage asserting contract-shaped keys/meta survive merge normalization (`src/asr/adapters.py`, `tests/test_adapters.py`, `docs/STATUS.md`).

- 2026-02-12 – Hardened `--asr-preflight-only` stdout determinism: emits exactly one machine-parseable JSON line (including with `--verbose`), with no stage/success log leakage, while preserving preflight-only scope to backend availability validation + model resolution + device resolution; updated CLI tests and preflight docs contract (`cli.py`, `tests/test_cli.py`, `docs/CLI.md`, `docs/STATUS.md`).


- 2026-02-12 – Removed faster-whisper hardcoding from ASR worker subprocess path by requiring explicit `--asr-backend`, dispatching runtime backend instances deterministically, keeping faster-whisper preflight semantics unchanged, and ensuring non-ctranslate2 backends skip WhisperModel preflight; CLI now forwards selected backend to worker and tests cover command args + deterministic backend error text (`src/asr/asr_worker.py`, `cli.py`, `tests/test_asr_worker.py`, `tests/test_cli.py`, `docs/STATUS.md`).
- 2026-02-12 – Switched qwen3-asr runtime from Transformers pipeline to official qwen_asr Qwen3ASRModel.from_pretrained(...), mapped supported CLI options (language, deterministic timestamps, device, compute_type→dtype), raised explicit deterministic errors for unsupported decode knobs, updated qwen3 optional dependency metadata, and added backend success/failure/unsupported-option unit tests (`pyproject.toml`, `src/asr/qwen3_asr_backend.py`, `src/asr/registry.py`, `src/asr/readiness.py`, `tests/test_qwen3_asr_backend.py`, `tests/test_asr_registry.py`, `tests/test_cli.py`, `docs/STATUS.md`).

- 2026-02-12 – Resolved WhisperX CUDA probe source-of-truth ambiguity by mapping WhisperX preflight/device routing to ctranslate2 (matching runtime API path), synchronized readiness/CLI matrix wording, and updated probe-label coverage in ASR/CLI tests (`src/asr/device.py`, `src/asr/readiness.py`, `docs/CLI.md`, `tests/test_asr.py`, `tests/test_cli.py`, `docs/STATUS.md`).

- 2026-02-12 – Tightened ASR static typing around segment contracts without changing runtime payload format: segment normalizers now return `list[ASRSegment]`, worker backend dispatch/parse paths use explicit ASR contract typing, and parse/worker/subprocess tests now cover missing required segment keys (`src/asr/timestamp_normalization.py`, `src/asr/faster_whisper_backend.py`, `src/asr/whisperx_backend.py`, `src/asr/vibevoice_backend.py`, `src/asr/qwen3_asr_backend.py`, `src/asr/backends.py`, `src/asr/asr_worker.py`, `tests/test_asr.py`, `tests/test_asr_worker.py`, `tests/test_cli.py`, `docs/STATUS.md`).

- 2026-02-12 – Enforced backend-explicit ASR worker CLI parsing with required/validated `--asr-backend` choices, moved worker runtime config creation to backend-specific `ASRConfig` construction, and added worker/CLI tests that verify backend arg requirement and subprocess forwarding of the selected backend value (`src/asr/asr_worker.py`, `tests/test_asr_worker.py`, `tests/test_cli.py`, `docs/STATUS.md`).


- 2026-02-12 – Verified WhisperX runtime path uses WhisperX/CTranslate2 model loading (`whisperx.load_model`) and enforced a single CUDA-probe mapping (`ctranslate2`) across readiness dependencies/docs/tests; added readiness coverage asserting WhisperX probe/dependency labels (`src/asr/whisperx_backend.py`, `src/asr/readiness.py`, `docs/CLI.md`, `tests/test_backend_readiness.py`, `docs/STATUS.md`).
- 2026-02-12 – Aligned ASR parse/contract static typing by tightening validated meta/result construction (`ASRMeta`/`ASRResult`), widening parser input typing to accept both typed results and mapping-style JSON payloads, and adding worker/CLI/ASR tests that exercise contract re-parse paths and required segment key enforcement (`src/asr/backends.py`, `tests/test_asr.py`, `tests/test_asr_worker.py`, `tests/test_cli.py`, `docs/STATUS.md`).
- 2026-02-12 – Follow-up typing fix: made ASR result validator accept any `Mapping` root (not only `dict`) to match widened parse contract, and updated parse test coverage to use a non-`dict` Mapping implementation for contract-path enforcement (`src/asr/backends.py`, `tests/test_asr.py`, `docs/STATUS.md`).

- 2026-02-13 – Improved qwen3-asr failure diagnostics by preserving concise deterministic ValueError messages while emitting full traceback only when verbose logging callback is enabled; exceptions now keep original root cause chaining and tests verify verbose-only traceback output plus __cause__ preservation (`src/asr/qwen3_asr_backend.py`, `tests/test_qwen3_asr_backend.py`, `docs/STATUS.md`).

- 2026-02-13 – Refactored qwen3-asr verbose traceback coverage into a single subTest-parameterized unit test to remove duplicated init/transcribe failure assertions while preserving deterministic diagnostics checks (`tests/test_qwen3_asr_backend.py`, `docs/STATUS.md`).

- 2026-02-13 – Codified regression tests for qwen3-asr and CLI worker failure diagnostics: added a warning-path guard proving qwen transcribe kwargs never forward `temperature`, added failure-path coverage for verbose traceback emission plus preserved chained causes, and hardened CLI worker failure assertions for verbose/non-verbose stdout/stderr behavior with Windows-safe newline and path assertions (`tests/test_qwen3_asr_backend.py`, `tests/test_cli.py`, `docs/STATUS.md`).
