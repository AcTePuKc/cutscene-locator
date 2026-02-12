# Integration Issues and Edge Cases

This document lists known edge cases and failure modes that must be handled explicitly.
The tool must surface uncertainty instead of hiding it.

---

## 1. ASR text does not match script 1:1

### Description

Spoken dialogue often deviates from the script:

- paraphrasing
- dropped words
- added filler
- contractions vs full forms

### Handling

- Use fuzzy matching with confidence scores.
- Always return ranked candidates.
- Never force a single match without a score.
- Allow multiple script lines to map to similar ASR text.

---

## 2. Very short lines

### Description

Lines such as:

- "Yeah."
- "Right."
- "Okay."
- "Sure."

Are highly ambiguous without context.

### Handling

- Penalize very short normalized strings.
- Require higher confidence thresholds.
- Prefer scene context (previous/next lines) when grouping.
- Flag as high-risk in output.

---

## 3. Repeated or identical lines

### Description

The same line text may appear multiple times:

- across different missions
- across different files
- spoken by different characters

### Handling

- Matching must be time-based, not text-only.
- Scene reconstruction must preserve temporal order.
- Do not deduplicate by text alone.

---

## 4. Fillers and interjections

### Description

Spoken audio may include:

- "uh", "um", "ah", "eh"
- repeated sounds
- stutters

These may not exist in the script.

### Handling

- Treat fillers as low-weight tokens.
- Normalization may remove them for matching.
- Do not remove them from displayed ASR text.
- Never alter script text.

---

## 5. Overlapping speech

### Description

Multiple speakers talking at once:

- interruptions
- cross-talk
- background NPC chatter

### Handling

- ASR timestamps may overlap.
- Scene grouping must tolerate overlaps.
- Do not assume strict sequential ordering by end time.
- Prefer start time ordering.

---

## 6. Music, radio, and sound effects

### Description

Game audio may include:

- background music
- radio stations
- ambient chatter
- combat noise

### Handling

- Non-speech segments should be filtered when possible.
- If ASR produces text for non-dialogue audio:
  - mark as unmatched
  - exclude from scene stitching
- Never attempt to classify music lyrics as dialogue.

---

## 7. Incorrect or missing timestamps

### Description

ASR backends may return:

- zero-length segments
- overlapping or inverted timestamps
- missing timestamps

### Handling

- Validate timestamps.
- Drop invalid segments with warnings.
- Never fabricate timestamps.

---

## 8. Speaker identity uncertainty

### Description

Speaker diarization may be:

- missing
- incorrect
- inconsistent across chunks

### Handling

- Speaker labels are optional.
- Do not use speaker identity as a hard constraint.
- Speaker info may be attached as metadata only.

---

## 9. Chunk boundaries

### Description

Dialogue may span chunk boundaries:

- sentence cut in half
- response in next chunk

### Handling

- Chunks must include absolute time offsets.
- Scene stitching must operate on global time.
- Do not assume chunk-local continuity.

---

## 10. Script structure mismatch

### Description

Script files may be organized by:

- internal engine logic
- triggers
- reuse across scenes

Which does not match actual spoken order.

### Handling

- Scene order is determined exclusively by audio timestamps.
- Script file ordering must not be trusted.
- Script metadata (file/mission) is informational only.

---

## 11. Low-confidence matches

### Description

Some matches will remain ambiguous.

### Handling

- Low-confidence matches are valid outputs.
- They must be:
  - flagged
  - exported
  - reviewable
- Never auto-correct or guess.

---

## 12. Determinism vs ASR variability

### Description

ASR models may evolve or produce slightly different text.

### Handling

- Determinism applies to pipeline logic, not ASR internals.
- Given the same ASR output, downstream results must be identical.
- Version ASR backend metadata in outputs.

---

## 13. Failure philosophy

- Errors should stop the pipeline.
- Ambiguity should not.
- Missing data should be reported, not invented.

The tool is a locator and alignment aid, not an oracle.


## 13. Windows CUDA native aborts in ASR backends

### Description

Some Windows + CUDA + ASR backend combinations may abort in native code (for example, process exit code `-1073740791`) even when Python-level CUDA checks pass.

### Handling

- For `faster-whisper` on Windows with `--device cuda`, run ASR in an isolated subprocess.
- Parent CLI must catch child non-zero exits and emit actionable guidance.
- Parent process must never silently disappear due to child native aborts.
- For CUDA transcription, avoid tqdm progress monitor threads in the ASR execution path.
- Never pass `progress=` into `WhisperModel.transcribe(...)`; `faster-whisper` 1.2.1 does not accept it.
- Add an ASR preflight print before transcription with `ctranslate2` version, CUDA device count, selected device, and selected compute type.
- If worker exits with `-1073740791` / `3221226505`, advise `--compute-type float32` first and verify torch/ctranslate2 CUDA wheel compatibility with the installed CUDA runtime/driver (see faster-whisper issue #1086).


## 14. Backend mode limits and support matrix

### Description

Backends do not all support the same mode. ASR transcript-generation and forced-alignment are separate paths and must not be mixed.

### Handling

- `vibevoice` supports ASR transcript-generation mode only.
- `qwen3-forced-aligner` supports forced-alignment mode only.
- CLI ASR mode (`--asr-backend`) must reject alignment-only backends with deterministic error text.
- Missing optional backend dependencies must produce deterministic install guidance (matching backend extra name).

## 15. Backend-specific CUDA preflight pitfalls (qwen3-asr / whisperx / vibevoice)

### Description

These backends rely on `torch` CUDA probing (`torch.cuda.is_available()`) during device preflight. CUDA install state can differ from backend import state.

### Handling

- Keep install checks and CUDA preflight checks separate and explicit:
  - install/readiness: optional dependency importability (`torch` + backend package),
  - runtime/readiness: CUDA probe reason from `torch` path.
- Surface deterministic preflight reason text for operator troubleshooting (for example, `torch CUDA probe reported unavailable`).
- Do not silently switch backend, model family, or mode if CUDA is unavailable.
- If the operator requested `--device cuda` and preflight fails, show explicit retry guidance: `--device cpu`.

### Windows-specific notes

- Windows users frequently hit CUDA wheel/runtime mismatches even when imports succeed.
- Preserve deterministic CPU rerun behavior per backend:
  - `qwen3-asr`: retry same command with `--device cpu`.
  - `whisperx`: retry same command with `--device cpu`.
  - `vibevoice`: retry same command with `--device cpu`.
- Do not perform automatic backend/device switching; require rerun with `--device cpu` as an explicit operator action.

