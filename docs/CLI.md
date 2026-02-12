# CLI Usage

`cutscene-locator` is a CLI-first tool. All functionality must be accessible from the command line.

UI frontends (e.g. PySide6) may be added later, but must call the same internal APIs.

---

## Basic usage

```bash

cutscene-locator --input <media> --script <script.tsv> --out <output_dir>

```

Example:

```bash

cutscene-locator 
--input mission_03.mp4 
--script gta_dialogues.tsv 
--out out/

```

---

## Required arguments

### `--input`

Path to input media file.

Accepted formats:

- Audio: `.wav`, `.flac`, `.mp3`, `.ogg`
- Video: `.mp4`, `.mkv`, `.avi`, `.mov`

If video is provided, audio is extracted using ffmpeg.

---

### `--script`

Path to script file.

Supported formats:

- `.tsv`
- `.csv`

Must contain at least:

- `id`
- `original`

---

### `--out`

Output directory.

- Created if it does not exist.
- Temporary files are written under:
  - `<out>/_tmp/`

---

## Optional arguments

### `--chunk <seconds>`

Chunk size in seconds.

- Default: `300`
- Use `0` to disable chunking.

Example:

```bash

--chunk 600

```

---

### `--scene-gap <seconds>`

Time gap threshold for starting a new scene.

- Default: `10`

---

### `--asr-backend <name>`

Select ASR backend.

Backends are split into two sets:

- **declared backends**: implemented in code.
- **enabled backends**: declared backends whose optional dependencies are installed.

Current declared backends (exact names):

- `mock` (always enabled)
- `faster-whisper` (enabled when optional runtime dependencies are installed)
- `qwen3-asr` (enabled only when optional runtime dependencies are installed)

Backend validation behavior is explicit:

- If you request an **unknown backend name** (not declared), the CLI returns the existing `Unknown ASR backend` error with available enabled backends.
- If you request a **declared but disabled backend**, the CLI returns an actionable error including exact missing optional dependencies and an install command for the matching extra (for example `pip install 'cutscene-locator[asr_qwen3]'`).

Install extras:

- `pip install 'cutscene-locator[asr_faster_whisper]'`
- `pip install 'cutscene-locator[asr_qwen3]'`

Backend dependency expectations:

- `mock`
  - No optional extra required.
- `faster-whisper`
  - Install: `pip install 'cutscene-locator[asr_faster_whisper]'`
  - Includes: `faster-whisper`, `huggingface_hub`.
- `qwen3-asr`
  - Install: `pip install 'cutscene-locator[asr_qwen3]'`
  - Includes: `torch`, `transformers`, `huggingface_hub`.

Default:

```bash

--asr-backend mock

```

---

### `--mock-asr <file>`

Path to mock ASR JSON output.

Required when `--asr-backend mock` is used.

Example:

```bash

--asr-backend mock --mock-asr tests/fixtures/asr.json

```

---


### `--model-path <path>`

Filesystem path to a local ASR model directory/file.

- Path must already exist.
- This flag is filesystem-only and never accepts a Hugging Face repo id.

### `--model-id <repo_id>`

Hugging Face repo id to download as a deterministic local snapshot cache.

- Example: `openai/whisper-tiny`
- Cached under: `<cache>/models/<backend>/<sanitized_repo_id>/<revision_or_default>/`
- On cache hit, model resolution returns the cached path immediately (no `snapshot_download` call).
- Requires `huggingface_hub` when downloading (cache hits do not re-download).

Backend-specific compatibility caveats:

- `mock`
  - `--model-id` is unsupported for practical use (mock backend reads `--mock-asr` JSON and does not load a model artifact).
- `faster-whisper`
  - `--model-id` should resolve to a faster-whisper CTranslate2 snapshot (for example, `Systran/faster-whisper-*`).
  - Standard Transformers checkpoints (for example, `openai/whisper-*`) are not compatible with faster-whisper runtime loading.
- `qwen3-asr`
  - `--model-id` must resolve to a full Transformers snapshot containing `config.json`, tokenizer assets, and model weights.
  - Incomplete local folders are rejected with deterministic validation errors.

### `--revision <revision>`

Optional Hugging Face revision when used with `--model-id`.

- Example: `--revision main`
- If omitted, cache folder uses `default`.

### `--device <cpu|cuda|auto>`

Selects ASR execution device.

### `--compute-type <float16|float32|auto>`

Controls faster-whisper compute precision passed to `WhisperModel(compute_type=...)`.

- `float16`: fastest on most CUDA setups
- `float32`: safer fallback for CUDA instability
- `auto`: backend default

When `--asr-backend faster-whisper` is selected, CLI prints an ASR preflight line before transcription that includes:

- detected `ctranslate2` version
- CUDA device count reported by `ctranslate2`
- selected resolved device
- selected compute type

If CUDA ASR aborts, first retry with `--compute-type float32`, then verify torch/ctranslate2 CUDA wheel compatibility (see faster-whisper issue #1086).

### `--progress <on|off>`

Controls progress behavior for ASR/model-resolution operations.

- For Hugging Face snapshot downloads, this controls progress bars as before.
- For faster-whisper transcription on CUDA, progress bars are disabled internally to avoid Windows tqdm monitor-thread crashes.

- Windows default: `off` (sets `HF_HUB_DISABLE_PROGRESS_BARS=1`)
- Non-Windows default: `on`
- Useful for avoiding native aborts in tqdm monitor threads on some Windows environments.

---

### `--match-threshold <float>`

Minimum confidence score for “high-confidence” matches.

- Default: `0.85`
- Lower scores are still exported but flagged.

### `--match-quick-threshold <float>`

Quick-filter minimum token-overlap score used before expensive fuzzy scoring.

- Default: `0.25`
- Set to `0.0` to disable quick-filter pruning.

### `--match-length-bucket-size <int>`

Token-count bucket size used by matching candidate indexes.

- Default: `4`

### `--match-max-length-bucket-delta <int>`

How many neighboring length buckets to include when searching candidates.

- Default: `3`

### `--match-monotonic-window <int>`

Optional monotonic alignment window in script-row indexes.

- Default: `0` (disabled)
- When > 0, later segments search only from previous best row forward by this window, which can reduce comparisons and enforce timeline consistency.

### `--match-progress-every <int>`

Verbose matching progress interval (segments).

- Default: `50`

### `--asr-vad-filter <on|off>`

Controls ASR backend VAD segmentation filter where supported.

- Default: `off`

### `--asr-merge-short-segments <seconds>`

Post-ASR deterministic merge threshold for short adjacent segments.

- Default: `0.0` (disabled)
- When > 0, segments shorter than threshold are merged into the previous segment to stabilize segment counts across runs/devices.

---

### `--keep-wav`

Do not delete intermediate WAV files.

Useful for debugging ffmpeg preprocessing.

---

### `--ffmpeg-path <path>`

Explicit path to ffmpeg binary.

If not provided:

- `ffmpeg` is resolved from PATH.

---

### `--verbose`

Enable verbose logging.

---

### `--version`

Print version and exit.

---

### `--help`

Print help and exit.

---

## Outputs

On success, the tool writes:

- `matches.csv`
- `scenes.json`
- `subs_qa.srt`

Additionally, if the script contains a `translation` column:

- `subs_target.srt`

All files are written under the directory specified by `--out`.

---

## Exit codes

- `0` – success
- `1` – fatal error (missing files, ffmpeg failure, invalid input)
- `2` – partial success (completed with warnings, low-confidence matches)
- `3` – ASR backend failure

---

## Logging rules

- Errors go to stderr.
- Normal progress goes to stdout.
- Warnings must be explicit and actionable.
- Never silently skip data.

---

## Determinism

Given the same:

- input media
- script
- ASR backend output
- configuration

The CLI must produce identical outputs.
