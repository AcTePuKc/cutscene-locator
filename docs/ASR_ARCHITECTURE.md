# ASR Architecture – cutscene-locator

This document locks the ASR layer architecture before real backend implementations.

The goal is to support multiple ASR backends in a deterministic, pluggable, and UI-ready way without coupling the core pipeline to any specific model.

---

## Design Principles

- Backend-agnostic core.
- No silent fallback between backends.
- Deterministic behavior given identical inputs.
- Explicit model selection.
- UI-ready (progress + cancellation support).
- No mandatory auto-download.

---

## 1. Backend Interface (Authoritative Contract)

All ASR backends must implement a unified interface.

### Required method

```py

transcribe(audio_path: str, config: ASRConfig) -> ASRResult

```

### ASRResult format

Must match the internal contract:

```py

{
segments: [
{
segment_id: str,
start: float,
end: float,
text: str,
speaker?: str
}
],
meta: {
backend: str,
model: str,
version: str,
device: str
}
}

```

Backends must NOT:

- Normalize text
- Modify timestamps
- Reorder segments
- Fabricate data

---

## 2. Backend Registry

Backends are registered centrally:

```bash

src/asr/registry.py

```

Responsibilities:

- Map backend name → backend class
- Validate backend existence
- Expose backend capability metadata

Example CLI usage:

```bash

--asr-backend faster-whisper

```

No implicit backend selection.

---

## 3. Backend Capability Metadata

Each backend declares capabilities:

```py

supports_word_timestamps: bool
supports_alignment: bool
supports_diarization: bool
max_audio_duration: Optional[int]

```

Core pipeline must not assume advanced capabilities.

---

## 4. Model Resolution Strategy

### A. Explicit model path (default)

```bash

--model-path <path>

```

User provides model location.

If missing → error.

---

### B. Optional model-id download

```bash

--model-id <repo-id> [--revision <rev>]

```

Policy:

- Uses `huggingface_hub.snapshot_download`.
- Caches in deterministic directory:
  `~/.cutscene-locator/models/<backend>/<sanitized_repo_id>/<revision_or_default>/`
- Backend validates required model files and fails with actionable errors.

### C. Optional auto-download

```bash

--auto-download <model-size>

```

Policy:

- Only downloads minimal model (e.g. tiny/base).
- Stores in deterministic cache directory:
  ~/.cutscene-locator/models/
- Logs clearly when downloading.
- Never triggers silently.

If download fails → error.

---

### D. models/ directory convention

If no explicit path:

- Look inside local `models/` folder.
- No recursion.
- Deterministic resolution.

No fallback to other backends.

---

## 5. Device Handling

CLI option:

```bash

--device cpu
--device cuda
--device auto

```

Rules:

- auto resolves deterministically (prefer cuda if available).
- If requested device unavailable → error.
- Device must be reported in ASRResult.meta.

---

## 6. No Silent Fallback Policy

If:

- backend fails
- model missing
- device unavailable

The tool must:

- fail loudly
- provide actionable error

Never:

- switch backend automatically
- switch model size automatically

---

## 7. UI Readiness Hooks

Backends must support:

### Progress callback (future-safe)

Optional parameter in config:

```py

progress_callback(percent: float)

```

No UI logic inside backend.

---

### Cancellation support (future-safe)

Backends should be structured to allow future interruption,
but no cancellation implementation required yet.

---

## 8. Determinism Guarantees

Given:

- same audio
- same backend
- same model
- same config

ASRResult must be identical (excluding floating precision variance).

No randomness allowed unless explicitly controlled.

---

## 9. Out of Scope (for now)

- Forced alignment
- Word-level timestamps
- Streaming ASR
- Diarization pipeline
- Multi-backend fallback chains

These belong to future milestones.

---

## 10. Implementation Order (Milestone 2)

1. Implement backend registry.
2. Implement faster-whisper backend.
3. Add device handling.
4. Add optional auto-download.
5. Validate deterministic behavior.
6. Benchmark memory and runtime.

Only after these are stable may additional backends be added.
