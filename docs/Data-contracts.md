# Data Contracts

This document defines the exact data formats used by `cutscene-locator`.
All producers and consumers must follow these contracts strictly.

---

## 1. Script file (TSV / CSV)

### Required columns

- `id` (string, unique)
- `original` (string)

### Optional columns

- `translation` (string)
- `file` (string)
- `mission` (string)
- `speaker` (string)

### TSV example

```cs

id original translation file mission
M01_001 You don't get it. Ти не го разбираш. m01_dialogue mission_01
M01_002 Yeah. Да. m01_dialogue mission_01
M01_003 I'm serious. Говоря сериозно. m01_dialogue mission_01

```

### Rules

- The script file is read-only.
- Row order must not be trusted.
- IDs must be preserved exactly.

---

## 2. Canonical audio (internal)

This format is produced by ffmpeg and consumed by ASR backends.

- WAV
- mono
- PCM s16le
- 16 kHz
- Stored under: `<out>/_tmp/audio/`

Example filename:

```bash

input_chunk_000.wav

```

---

## 3. ASR output (internal, normalized JSON)

All ASR backends must produce output that conforms to this structure.

### ASR JSON schema

```json
{
  "segments": [
    {
      "segment_id": "seg_0001",
      "start": 12.34,
      "end": 15.56,
      "text": "you don't get it",
      "speaker": "SPEAKER_1"
    }
  ],
  "meta": {
    "backend": "mock",
    "model": "unknown",
    "version": "1.0",
    "device": "cpu"
  }
}
```

### Rules

- `start` and `end` are absolute times (seconds) relative to original media.
- Timestamp normalization is applied before `validate_asr_result(...)` in real ASR backends.
- `start`/`end` must be numeric, finite, and non-negative.
- Deterministic float policy: timestamps are rounded with decimal half-up to 6 fractional digits.
- Boundary policy: `end >= start` is accepted during normalization; zero-length segments are dropped before contract validation.
- Stable ordering policy: segments are deterministically sorted by `(start, end, original_index)`.
- Overlap semantics: inter-segment overlaps are allowed and preserved (no implicit de-overlap).
- Cross-chunk continuity semantics (ASR→matching bridge):
  - Chunk-local timestamps are converted to absolute timeline values via `ChunkMetadata.absolute_offset_seconds` before matching/scene stitching.
  - Deterministic boundary merge applies only to adjacent chunk indexes (`n` and `n+1`).
  - Repeated/overlapping boundary duplicates with equivalent normalized text are merged into one segment.
  - Split utterances across adjacent chunks with a tiny boundary gap are merged and text is concatenated with token-overlap de-duplication.
  - Tiny head/tail boundary fragments (short duration) are merged into the neighboring segment when the same boundary rule applies.
  - Tie-break/preservation rules: keep the earlier segment `segment_id` and `speaker` when present, preserve earliest `start` + latest `end`, preserve stable deterministic ordering by `(start, end, original_index)`.
- `speaker` is optional.
- ASR text is not rewritten by timestamp normalization.
- `qwen3-asr` is text-first ASR at runtime: timestamped `chunks[]` are consumed when present, but timestamp emission is not guaranteed by a universal runtime flag and must be treated as backend-output dependent.
- Bridge enforcement: if a backend advertises `timestamp_guarantee="text-only"` and runtime output lacks deterministic segment timestamps, the CLI bridge must raise a deterministic actionable error and stop before matching; timestamps must never be fabricated.

---


## 4. Forced alignment output (internal, normalized JSON)

Forced alignment uses a separate contract from ASR transcript generation.
`qwen3-asr` belongs to the ASR transcript contract (`ASRResult`) path. Forced-aligner model families (for example `qwen3-forced-aligner`) require this alignment contract/input and are not drop-in ASR replacements.

### Alignment input contract

```json
{
  "audio_path": "out/_tmp/audio/input_chunk_000.wav",
  "reference_spans": [
    {"ref_id": "M01_001", "text": "You don't get it."},
    {"ref_id": "M01_002", "text": "Yeah."}
  ]
}
```

### Alignment JSON schema

```json
{
  "transcript_text": "you don't get it",
  "spans": [
    {
      "span_id": "span_0001",
      "start": 12.34,
      "end": 15.56,
      "text": "you don't get it",
      "confidence": 0.94
    }
  ],
  "meta": {
    "backend": "forced-aligner",
    "version": "1.0",
    "device": "cpu"
  }
}
```

### Rules

- Input transcript text is caller-provided known text and must be echoed in `transcript_text`.
- `start` and `end` are absolute times (seconds) relative to original media.
- `start < end` must hold for every span.
- `confidence` is required and must be in `[0.0, 1.0]`.
- Alignment does not rewrite transcript/script text.

---

## 5. Matches output (`matches.csv`)

One row per ASR segment.

### Columns

- `segment_id`
- `start_time`
- `end_time`
- `asr_text`
- `matched_id`
- `matched_text`
- `score`

### Example

```json
segment_id,start_time,end_time,asr_text,matched_id,matched_text,score
seg_0001,12.34,15.56,you don't get it,M01_001,You don't get it.,0.93
seg_0002,16.10,16.80,yeah,M01_002,Yeah.,0.81
```

### Rules

- `score` is a float in range `[0.0, 1.0]`.
- Low scores are valid and must be exported.
- `matched_id` may be empty if no candidate exceeds minimum threshold.

---

## 6. Scene reconstruction (`scenes.json`)

Scenes are time-ordered groups of matched lines.

### Schema

```json
{
  "scenes": [
    {
      "scene_id": "scene_001",
      "start_time": 12.34,
      "end_time": 28.90,
      "lines": [
        {
          "segment_id": "seg_0001",
          "line_id": "M01_001"
        },
        {
          "segment_id": "seg_0002",
          "line_id": "M01_002"
        }
      ]
    }
  ]
}
```

### Rules

- Scene boundaries are derived from time gaps.
- Scene order is strictly chronological.
- Lines appear in spoken order, not script order.

---

## 7. QA subtitles (`subs_qa.srt`)

Used for review and correction.

### Format

```s
1
00:00:12,340 --> 00:00:15,560
[ORIG] You don't get it.
[TR] Ти не го разбираш.

2
00:00:16,100 --> 00:00:16,800
[ORIG] Yeah.
[TR] Да.
```

### Rules

- `[TR]` line is omitted if no translation column exists.
- Text is not normalized.
- Timestamps come directly from ASR/scene alignment.

---

## 8. Target subtitles (`subs_target.srt`)

Final subtitles in target language.

### Format

```s
1
00:00:12,340 --> 00:00:15,560
Ти не го разбираш.

2
00:00:16,100 --> 00:00:16,800
Да.
```

### Rules

- Generated only if `translation` column exists.
- Uses same timestamps as QA subtitles.
- No original text included.

---

## 9. Temporary files and caching

- Temporary files live under `<out>/_tmp/`
- The tool may clean this directory unless `--keep-wav` is set.
- No temporary files are required for correctness.

---

## 10. ASR model artifact contract (readiness preflight)

Runtime ASR backends that load local model snapshots must pass deterministic artifact-layout validation before inference.

- `qwen3-asr` required core artifacts: `config.json`, tokenizer asset set (`tokenizer.json` or `tokenizer.model` or `vocab.json`), `tokenizer_config.json`, and model weights (`model.safetensors` or `pytorch_model.bin` or sharded index variants).
- `qwen3-asr` optional artifacts: `processor_config.json` and `preprocessor_config.json` (accepted when present but not required for readiness pass/fail).

---

## 11. Versioning and metadata

All outputs may include optional metadata headers/comments indicating:

- tool version
- ASR backend
- configuration hash

This metadata must not affect parsing.

---

## Contract enforcement

- Invalid input formats → error
- Missing required fields → error
- Optional fields → best-effort passthrough
- The tool must never guess missing data

These contracts are authoritative.
