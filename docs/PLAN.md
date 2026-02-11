# Milestone 1 Implementation Plan (Core pipeline with mock ASR)

> This plan is for **Milestone 1 only**.

## Rules of engagement

- Implement only unchecked items from `docs/STATUS.md`.
- Update `docs/STATUS.md` when completing items.

## Milestone 1 Implementation Plan (Core pipeline with mock ASR)

Ordered, minimal, and mapped directly to unchecked `docs/STATUS.md` items.

1. **Lock the Milestone 1 execution path in CLI inputs/outputs first**: accept exactly one media input + required script + output dir, and define the run flow in this order: ingest → preprocess → optional chunking → ASR → normalization/matching → scene stitching → exports.  
   **STATUS items:** `Argument parsing`, `Help and version output`, `Exit codes`, `Verbose logging`, `Error handling`.

2. **Implement ffmpeg preflight and fail-fast behavior**: verify `ffmpeg` availability at startup and terminate clearly if unavailable.  
   **STATUS item:** `ffmpeg availability check`.

3. **Implement deterministic media preprocessing**: for video input, extract audio; for all inputs, convert to canonical WAV (`mono`, `PCM s16le`, `16kHz`) via subprocess-only ffmpeg calls.  
   **STATUS items:** `Audio extraction from video via ffmpeg`, `Audio normalization to canonical WAV`.

4. **Implement temp workspace layout under output directory**: route temporary artifacts to `<out>/_tmp/` (including canonical audio location), with deterministic naming.  
   **STATUS item:** `Temporary file management (out/_tmp/)`.

5. **Implement optional fixed chunking with absolute offsets**: split canonical audio into 300s, non-overlapping chunks; emit chunk metadata containing source filename, chunk index, absolute time offset.  
   **STATUS item:** `Chunking implementation (time-based)`.

6. **Implement script ingestion contract**: parse TSV/CSV; enforce required columns (`id`, `original`); preserve optional columns (`translation`, `file`, `mission`, `speaker`) for downstream passthrough.  
   **STATUS items:** `TSV/CSV parser`, `Required column validation (id, original)`, `Optional column passthrough (translation, file, etc.)`.

7. **Implement ASR abstraction for Milestone 1 + mock backend**: define backend interface and implement `MockASRBackend` consuming pre-generated JSON that conforms to the ASR contract.  
   **STATUS items:** `ASR backend interface`, `Mock ASR backend (JSON input)`.

8. **Implement ASR segment validation before matching**: validate required fields and timestamp invariants (`start`, `end`, `start < end`, absolute times), preserving optional speaker field.  
   **STATUS item:** `ASR segment validation (timestamps, text)`.

9. **Implement shared normalization pipeline once and reuse in both script + ASR paths**: lowercase, trim, collapse spaces, strip punctuation, and filler-token reduction for matching only (not export text mutation).  
   **STATUS items:** `Script normalization pipeline`, `Text normalization (shared ASR/script)`.

10. **Implement matching core with explicit ambiguity handling**: fuzzy-match each ASR segment against script lines, produce ranked candidates and confidence score in `[0,1]`, and flag low-confidence records without treating them as hard failures.  
    **STATUS items:** `Fuzzy matching implementation`, `Ranked candidate selection`, `Confidence scoring`, `Low-confidence flagging`.

11. **Implement deterministic scene reconstruction from matched timeline**: stitch by chronological time order, start a new scene on gap threshold (>10s default), generate stable scene IDs, and tolerate overlaps while preserving spoken order.  
    **STATUS items:** `Scene gap logic`, `Chronological stitching`, `Scene ID generation`, `Overlap tolerance`.

12. **Implement `matches.csv` writer exactly per contract**: one row per ASR segment with required columns and low-score rows retained.  
    **STATUS item:** ``matches.csv` writer`.

13. **Implement `scenes.json` writer exactly per contract**: output `scenes` array with `scene_id`, time bounds, and ordered `{segment_id, line_id}` line mappings.  
    **STATUS item:** ``scenes.json` writer`.

14. **Implement subtitle exporters with conditional target output**:  
    - `subs_qa.srt`: timestamps + `[ORIG]`, plus `[TR]` only when translation exists.  
    - `subs_target.srt`: generate only if translation column exists, same timestamps as QA.  
    **STATUS items:** ``subs_qa.srt` writer`, ``subs_target.srt` writer (optional)`.

15. **Close Milestone 1 with deterministic and contract checks wired into CLI exits**: ensure hard errors for missing required inputs/columns/ffmpeg/backend failures, warnings for low-confidence matches, and deterministic outputs for identical input+config+ASR JSON.  
    **STATUS items:** `Exit codes`, `Verbose logging`, `Error handling` (finalized behavior).
