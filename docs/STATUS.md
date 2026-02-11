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
- [ ] Audio extraction from video via ffmpeg
- [ ] Audio normalization to canonical WAV
- [ ] Chunking implementation (time-based)
- [ ] Temporary file management (`out/_tmp/`)

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

- [ ] Text normalization (shared ASR/script)
- [ ] Fuzzy matching implementation
- [ ] Ranked candidate selection
- [ ] Confidence scoring
- [ ] Low-confidence flagging

### Scene reconstruction

- [ ] Scene gap logic
- [ ] Chronological stitching
- [ ] Scene ID generation
- [ ] Overlap tolerance

### Exports

- [ ] `matches.csv` writer
- [ ] `scenes.json` writer
- [ ] `subs_qa.srt` writer
- [ ] `subs_target.srt` writer (optional)

### CLI

- [x] Argument parsing
- [x] Help and version output
- [x] Exit codes
- [x] Verbose logging
- [x] Error handling

---

## Milestone 2 – Real ASR backends (post-MVP)

### Whisper / Qwen / others

- [ ] ASR backend adapter (generic)
- [ ] WhisperX backend
- [ ] Qwen ASR backend
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
