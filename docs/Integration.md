# Integration

This document defines the exact input/output contracts and the processing pipeline for `cutscene-locator`.

The goal is to deterministically reconstruct dialogue scenes from audio/video and align them to an existing script (TSV/CSV).

---

## Overview

High-level pipeline:

1. Input ingestion (audio or video)
2. Audio preprocessing via ffmpeg
3. (Optional) chunking
4. Branch A: ASR (speech-to-text) with timestamps
5. ASR backend preflight (install/readiness + model artifact validation + device probe)
6. Branch B: Forced alignment (known transcript/script text + timestamp spans)
7. Text normalization
7. Fuzzy matching against script
8. Scene stitching
9. Export (CSV / JSON / SRT)

The tool never translates text. It only aligns existing text.

---

## Inputs

### 1. Media input

Accepted formats:

- Audio: `.wav`, `.flac`, `.mp3`, `.ogg`
- Video: `.mp4`, `.mkv`, `.avi`, `.mov`

The tool **does not process video directly**.
If a video is provided, audio is extracted using ffmpeg.

Only one media input is accepted per run.

Example:

```bash

cutscene-locator --input mission_03.mp4 --script script.tsv --out out/

```

---

### 2. Script input (required)

Supported formats:

- `.tsv`
- `.csv`

Minimum required columns:

- `id` – unique line identifier
- `original` – original-language line text

Optional columns (if present, passed through):

- `translation`
- `file`
- `mission`
- `speaker`

The tool must **not modify the original script file**.

Example TSV:

```bash

id original translation file
M03_012 You don't get it. Ти не го разбираш. m03_dialogue

```

---

## Audio preprocessing (ffmpeg)

ffmpeg is a **mandatory external dependency**.

### Canonical audio format

All inputs are converted to:

- WAV
- mono
- PCM s16le
- 16 kHz sample rate

Example ffmpeg intent (illustrative):

```bash

ffmpeg -i input.mp4 -ac 1 -ar 16000 -c:a pcm_s16le output.wav

````

### Rules

- No audio decoding or resampling is implemented in Python.
- ffmpeg is always invoked via subprocess.
- Temporary files are written under:
  - `out/_tmp/`
- If ffmpeg is missing:
  - fail fast
  - print a clear error message

---

## Chunking

Chunking is optional but recommended.

- Default chunk size: 300 seconds (5 minutes)
- Chunks must overlap **0 seconds** (no sliding windows)
- Each chunk produces independent ASR output

Chunk metadata must include:

- original media filename
- chunk index
- absolute time offset

---

## ASR stage

### Requirements

ASR output **must** provide:

- recognized text
- start timestamp (seconds)
- end timestamp (seconds)

Speaker labels are optional.


Readiness preflight for runtime ASR backends (`qwen3-asr`, `whisperx`, `vibevoice`) is deterministic and must run before inference. `qwen3-asr` belongs only to the ASR transcript path (`ASRResult`) and is not an alignment backend:


- verify optional dependency importability for selected backend extras,
- verify backend status is enabled in registry,
- verify model artifact layout matches backend contract,
- report backend-appropriate device probe reason; if CUDA is unavailable, operator retries with `--device cpu`.

### Backend abstraction

ASR must be implemented behind a backend interface.

Initial milestone:

- `MockASRBackend` reading pre-generated JSON (for tests)

Later backends (out of scope for Milestone 1):

- WhisperX
- Qwen ASR
- VibeVoice ASR

---


## Forced alignment stage (explicit path)

Forced alignment is a distinct pipeline path and must not be routed through the ASR transcript interface. Forced-aligner model families (for example `qwen3-forced-aligner`) require the alignment contract/path and are not drop-in replacements for ASR transcript backends.

### Contract

Forced alignment input:

- canonical audio path
- known transcript/script spans (`reference_spans[]` with `ref_id` + `text`) provided by caller

Forced alignment output must conform to the alignment contract in `docs/Data-contracts.md`:

- `transcript_text` (echo of known transcript text)
- `spans[]` with `span_id`, `start`, `end`, `text`, `confidence`
- `meta` with `backend`, `version`, `device`

### Capability gating

Backends declaring `supports_alignment=True` are alignment backends and must be invoked via the alignment pipeline.

CLI ASR mode (`--asr-backend`) is transcript-generation mode only and must reject alignment-only backends with a clear error.

Registry capability metadata also includes `timestamp_guarantee` so downstream logic can reason deterministically about timestamp trust level:

- `text-only`: transcript-first backend; timestamps are not deterministic guarantees and should be treated as provisional unless explicit forced alignment is run.
- `segment-level`: deterministic segment timestamps are available directly from ASR output.
- `alignment-required`: backend is not an ASR transcript generator and must run through forced alignment contracts.

`qwen3-asr` is explicitly `text-only`; `qwen3-forced-aligner` remains a separate `alignment-required` backend.

---

## Text normalization

Normalization is applied consistently to:

- ASR text
- script `original` column

Normalization rules:

- lowercase
- trim whitespace
- collapse repeated spaces
- strip punctuation
- reduce filler tokens (e.g. "uh", "um", "ah") to low weight or remove

Normalization is **not** destructive to exported text.

---

## Matching

For each ASR segment:

- Compute fuzzy similarity against all candidate script lines
- Return ranked candidates with scores
- Do not auto-select without confidence

Recommended output per segment:

- `segment_id`
- `start_time`
- `end_time`
- `asr_text`
- `best_match_id`
- `best_match_text`
- `score`

Low-confidence matches must be flagged.

---

## Scene stitching

Scenes are reconstructed from time-aligned matches.

Rules:

- A scene is a sequence of matched segments
- A new scene starts if:
  - time gap > configurable threshold (default: 10 seconds)
- Scene ordering is strictly time-based

Scene output includes:

- scene_id
- start_time
- end_time
- ordered list of matched line IDs

---

## Outputs

### 1. matches.csv

One row per ASR segment.

Required columns:

- segment_id
- start_time
- end_time
- asr_text
- matched_id
- score

---

### 2. scenes.json

Structured scene reconstruction.

Example structure:

```json
{
  "scene_id": "scene_001",
  "start_time": 12.34,
  "end_time": 89.10,
  "lines": [
    "M03_012",
    "M03_013",
    "M03_014"
  ]
}
````

---

### 3. subs_qa.srt

QA subtitle file.

- Includes timestamps
- Shows original text
- May include translation on a second line (if available)

Purpose: review and correction.

---

### 4. subs_target.srt

Target-language subtitles.

- Uses translation column only
- Same timestamps as QA file
- Generated only if translation column exists

---

## Error handling

- Missing required columns → error
- ffmpeg failure → error
- ASR backend failure → error
- Low-confidence matches → warning, not error

---

## Determinism guarantee

Given:

- same input media
- same script
- same ASR backend
- same configuration

The outputs must be identical.

No randomness is permitted outside ASR internals.
