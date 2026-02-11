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

Supported values (initially):

- `mock`

Future values:

- `whisperx`
- `qwen`
- `vibevoice`

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
- Requires `huggingface_hub` only when downloading.

### `--revision <revision>`

Optional Hugging Face revision when used with `--model-id`.

- Example: `--revision main`
- If omitted, cache folder uses `default`.

### `--progress <on|off>`

Controls model download progress bars for Hugging Face snapshot downloads.

- Windows default: `off` (sets `HF_HUB_DISABLE_PROGRESS_BARS=1`)
- Non-Windows default: `on`
- Useful for avoiding native aborts in tqdm monitor threads on some Windows environments.

---

### `--confidence-threshold <float>`

Minimum confidence score for “high-confidence” matches.

- Default: `0.85`
- Lower scores are still exported but flagged.

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
