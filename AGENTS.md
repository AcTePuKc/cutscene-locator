# AGENTS

This repository contains a CLI-first tool that reconstructs dialogue scenes from audio/video using ASR + matching against a known script (TSV/CSV). It is intended for localization/QA workflows.

## Core principles

- CLI-first. No UI in initial milestones.
- Deterministic outputs whenever possible:
  - ffmpeg preprocessing is deterministic.
  - matching returns ranked candidates with confidence scores; never "invent" a single answer without a score.
- The tool does NOT translate. It only locates and aligns existing script lines to audio.

Coordination rule: Any implemented task must be recorded in docs/STATUS.md (checkbox + file path). If it is not recorded, it is treated as not done.

No duplicate stubs: Before creating a new stub/module, search the repo for an existing implementation and update it instead.

Tests are the announcement: New functionality must include at least one unit test or fixture update.

## Scope (initial milestones)

1. Input ingestion and preprocessing
   - Accept audio or video input.
   - Use `ffmpeg` to extract/normalize audio to a canonical WAV format.
   - Optional chunking (e.g., 5-minute chunks).

2. Matching and scene reconstruction
   - Run ASR (initially via a mock backend for tests).
   - Normalize ASR text and script text consistently.
   - Fuzzy-match ASR segments to script lines.
   - Stitch matches into scenes based on time gaps.

3. Exports
   - `matches.csv` (ranked matches + scores + timestamps)
   - `scenes.json` (scene groups + timeline)
   - `subs_qa.srt` (timestamps with original + translation if available)
   - `subs_target.srt` (timestamps with translated text only, if provided)

## Non-goals

- No automatic translation or LLM rewriting.
- No downloading content from external platforms.
- No OCR.
- No audio decoding/encoding implemented in Python beyond calling ffmpeg.

## Tooling rules

### ffmpeg usage (mandatory)

- Do not implement custom audio/video decoding.
- Use `ffmpeg` via subprocess for:
  - extracting audio from video
  - resampling
  - converting to mono
  - writing canonical WAV
  - chunking (time-based splitting)
- On startup, verify ffmpeg availability (`ffmpeg -version`).
- If missing, fail fast with a clear error message.

### Dependency discipline

- Prefer minimal dependencies.
- Keep ASR backends optional/extras.
- Unit tests must not require GPU, network access, or large model downloads.

### Test strategy

- Use fixtures for:
  - a small TSV/CSV script sample
  - a small "mock ASR output" JSON with timestamps
- For unit tests, ASR must be mocked.
- Integration tests may be optional and must be explicitly gated (e.g., `--run-integration`).

### Coding standards

- Python 3.10+.
- Use type hints.
- Prefer pure functions for normalization/matching where possible.
- Keep modules small and single-purpose.

## Architecture expectations

- `src/ingest/`: input handling and ffmpeg wrapper
- `src/vad/`: optional voice activity detection (can be deferred)
- `src/asr/`: ASR backend interface + implementations
- `src/match/`: normalization + fuzzy matching
- `src/scene/`: scene stitching and grouping
- `src/export/`: writers (CSV/JSON/SRT)
- `cli.py`: CLI entrypoint only (thin wrapper)

ASR must be implemented behind a backend interface, e.g.:

- `MockBackend` (tests)
- `WhisperXBackend` (later)
- `QwenBackend` (later)
- `VibeVoiceBackend` (later, if feasible)

## Definition of done (Milestone 1)

- `cutscene-locator` CLI runs with a mock ASR input and produces:
  - `out/matches.csv`
  - `out/scenes.json`
  - `out/subs_qa.srt`
- Includes unit tests for:
  - script parsing
  - normalization
  - fuzzy matching
  - scene grouping
  - export writers
- Clear errors and help text.
