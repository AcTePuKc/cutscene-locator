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

- Default: none (required)
- Valid values: path to a single input media file
- Backend applicability: all backends (pre-ASR ingest step)

---

### `--script`

Path to script file.

Supported formats:

- `.tsv`
- `.csv`

Must contain at least:

- `id`
- `original`

- Default: none (required)
- Valid values: path to `.tsv` or `.csv` containing required columns
- Backend applicability: all backends (matching/alignment requires script rows)

---

### `--out`

Output directory.

- Created if it does not exist.
- Temporary files are written under:
  - `<out>/_tmp/`

- Default: none (required)
- Valid values: writable directory path
- Backend applicability: all backends (common export/output path)

---

## Optional arguments

### `--chunk <seconds>`

Chunk size in seconds.

- Default: `300`
- Use `0` to disable chunking.
- Valid range: integer `>= 0`
- Backend applicability: all backends (preprocessing before ASR/alignment)

Example:

```bash

--chunk 600

```

---

### `--scene-gap <seconds>`

Time gap threshold for starting a new scene.

- Default: `10`
- Valid range: integer (recommended `>= 0`)
- Backend applicability: all backends (scene reconstruction after matching)

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
- `whisperx` (enabled only when optional runtime dependencies are installed)
- `qwen3-forced-aligner` (declared alignment backend; rejected in ASR transcription mode)

#### Backend/model family support matrix

| backend key | model examples | mode | required extras | minimum artifacts | output contract type |
| --- | --- | --- | --- | --- | --- |
| `mock` | local fixture JSON (for example `tests/fixtures/mock_asr_valid.json`) | `asr` | none | mock ASR JSON with `segments[]` and `meta` | ASR transcript contract (`ASRResult`) |
| `faster-whisper` | `Systran/faster-whisper-tiny`, `Systran/faster-whisper-small` | `asr` | `asr_faster_whisper` | CTranslate2 snapshot (for example `model.bin` + tokenizer assets) | ASR transcript contract (`ASRResult`) |
| `qwen3-asr` | `Qwen/Qwen3-ASR-0.6B`, `Qwen/Qwen3-ASR-1.7B` | `asr` | `asr_qwen3` | Transformers ASR snapshot (`config.json`, tokenizer assets, `tokenizer_config.json`, model weights; `processor_config.json`/`preprocessor_config.json` optional) | ASR transcript contract (`ASRResult`) |

qwen3 model layout contract: `processor_config.json` and `preprocessor_config.json` are optional metadata files and are never required for snapshot validation pass.

#### qwen3-asr compatibility matrix (deterministic smoke checks)

The table below reflects deterministic backend smoke-check coverage (mocked pipeline, no heavy inference) that separates checkpoint artifact compatibility from runtime/API initialization issues.

| variant / checkpoint family | smoke status | what is validated | known caveats |
| --- | --- | --- | --- |
| `Qwen/Qwen3-ASR-0.6B` layout-compatible snapshots | Verified (mocked backend test path) | `transformers.pipeline` call shape (`task`, `model`, `device`, `trust_remote_code`), invocation with `return_timestamps=True`, and strict `chunks[]` + tuple `(start, end)` normalization assumptions | A model path may resolve and still fail at runtime if checkpoint internals are not pipeline-compatible for `automatic-speech-recognition` in your installed `transformers`/`torch` versions. |
| `Qwen/Qwen3-ASR-1.7B` layout-compatible snapshots | Verified (mocked backend test path) | Same deterministic smoke assertions as 0.6B path; same normalization contract for timestamp tuples | Requires a pipeline-compatible checkpoint layout/API. Artifact presence alone does not guarantee runtime pipeline initialization success. |

| `whisperx` | local CTranslate2 Whisper snapshots compatible with WhisperX | `asr` | `asr_whisperx` | CTranslate2 snapshot (for example `model.bin` + tokenizer assets) | ASR transcript contract (`ASRResult`) |
| `qwen3-forced-aligner` | `Qwen/Qwen3-ForcedAligner-0.6B` | `alignment` | `asr_qwen3` | canonical WAV + caller-provided `reference_spans[]` + aligner model snapshot | Alignment contract (`AlignmentResult`, `docs/Data-contracts.md`) |
| `vibevoice` | local VibeVoice checkpoints compatible with runtime | `asr` | `asr_vibevoice` | local model snapshot path (`--model-path` or resolved `--model-id`) | ASR transcript contract (`ASRResult`) |

Mode gating is explicit and deterministic:

- `--asr-backend` is ASR transcript-generation mode only.
- Alignment backends (currently `qwen3-forced-aligner`) are loaded only when the explicit alignment pipeline path is requested.
- There is no silent mode fallback (for example, no implicit switch from `qwen3-forced-aligner` to `qwen3-asr`); any device change must be a manual rerun with `--device cpu`.

Backend validation behavior is explicit:

- If you request an **unknown backend name** (not declared), the CLI returns the existing `Unknown ASR backend` error with available enabled backends.
- If you request a **declared but disabled backend**, the CLI returns an actionable error including exact missing optional dependencies and an install command for the matching extra (for example `pip install 'cutscene-locator[asr_qwen3]'` or `pip install 'cutscene-locator[asr_whisperx]'`).

Authoritative definition for diagnostics:

- **Declared but disabled backend** means the backend key is registered in code but currently not runnable in this environment. Causes include:
  - missing optional dependencies,
  - feature flag disabled,
  - experimental backend disabled by default.
- Dependency-gated errors include actionable install-extra guidance when disabled due to missing optional dependencies (for example: `Install with: pip install 'cutscene-locator[asr_qwen3]'`).

Install extras:

- `pip install 'cutscene-locator[asr_faster_whisper]'`
- `pip install 'cutscene-locator[asr_qwen3]'`
- `pip install 'cutscene-locator[asr_whisperx]'`
- `pip install 'cutscene-locator[asr_vibevoice]'`

Backend dependency expectations:

- `mock`
  - No optional extra required.
- `faster-whisper`
  - Install: `pip install 'cutscene-locator[asr_faster_whisper]'`
  - Includes: `faster-whisper`, `huggingface_hub`.
- `qwen3-asr`
  - Install: `pip install 'cutscene-locator[asr_qwen3]'`
  - Includes: `qwen-asr`, `torch`, `transformers`, `huggingface_hub`.
- `qwen3-forced-aligner`
  - Install: `pip install 'cutscene-locator[asr_qwen3]'`
  - Includes: `qwen-asr`, `torch`, `transformers`, `huggingface_hub`.
- `whisperx`
  - Install: `pip install 'cutscene-locator[asr_whisperx]'`
  - Includes: `whisperx`, `torch`, `huggingface_hub` (and WhisperX runtime preflight relies on `ctranslate2` CUDA probing).
- `vibevoice`
  - Install: `pip install 'cutscene-locator[asr_vibevoice]'`
  - Includes: `vibevoice`, `torch`.

#### Deterministic backend readiness matrix (authoritative CUDA probe mapping by backend runtime API)

Use `scripts/verify_backend_readiness.py` for a no-inference verification pass:

```bash
python scripts/verify_backend_readiness.py --json \
  --qwen3-model-path <path-to-qwen3-snapshot> \
  --whisperx-model-path <path-to-ctranslate2-whisper> \
  --vibevoice-model-path <path-to-vibevoice-model>
```

This checklist verifies, per backend:

- optional dependency importability (`find_spec`-based, deterministic),
- backend enabled/disabled state in registry,
- model artifact layout contract validation,
- device preflight reason using backend-specific CUDA probe label (`torch` or `ctranslate2`).

Readiness preconditions summary:

| backend | runtime API used by backend implementation | CUDA probe label in preflight (`device.cuda_probe_label`) | CPU rerun policy (no autoswitch) |
| --- | --- | --- | --- |
| `faster-whisper` | `faster_whisper.WhisperModel(...)` (CTranslate2 runtime path) | `ctranslate2` | Do not perform automatic backend/device switching; manual rerun with `--device cpu`. |
| `qwen3-asr` | `transformers.pipeline(...)` (torch runtime path) | `torch` | Do not perform automatic backend/device switching; manual rerun with `--device cpu`. |
| `whisperx` | `whisperx.load_model(..., device=...)` (CTranslate2 Whisper runtime path via WhisperX) | `ctranslate2` | Do not perform automatic backend/device switching; manual rerun with `--device cpu`. |
| `vibevoice` | `vibevoice.transcribe_file(..., device=...)` (torch runtime path) | `torch` | Do not perform automatic backend/device switching; manual rerun with `--device cpu`. |

Common deterministic failure messages:

- Declared but disabled backend: `ASR backend '<name>' is declared but currently disabled... missing optional dependencies ... Install with: pip install 'cutscene-locator[<extra>]'`.
- qwen3 layout error: `Resolved qwen3-asr model is missing required artifacts...` (required core artifacts: `config.json`, tokenizer assets, `tokenizer_config.json`, and model weights; `processor_config.json`/`preprocessor_config.json` optional).
- whisperx layout error: `Resolved whisperx model is missing required files...`.
- CUDA request failure: `Requested --device cuda, but CUDA is unavailable... Manual rerun with --device cpu.`.

Windows guidance:

- Prefer quoting extras in PowerShell/CMD exactly as shown above.
- If CUDA runtime/wheel compatibility is uncertain, start with `--device cpu` to confirm deterministic pipeline behavior first.
- If CUDA fails after install, keep backend/model fixed and retry only with `--device cpu` (do not change backend implicitly).

Default:

```bash

--asr-backend mock

```

- Default: `mock`
- Valid values: declared backend names from registry (`mock`, `faster-whisper`, `qwen3-asr`, `whisperx`, `vibevoice`, and alignment-only backends such as `qwen3-forced-aligner` that are rejected in ASR mode)
- Backend applicability: selector for backend-specific execution path

---

### `--asr-preflight-only`

Run backend readiness checks only, then exit before ingest/transcription/matching/exports.

This mode is designed for deterministic QA and CI diagnostics and performs only:

- backend availability/capability validation,
- model resolution via existing model-resolution rules,
- backend-specific device probing via the same device resolution logic used by runtime execution.

Stdout contract: preflight-only emits **exactly one line** containing one JSON object serialized with sorted keys and compact separators (`json.dumps(..., sort_keys=True, separators=(",", ":"))`). No additional stdout lines are allowed in this mode (including when `--verbose` is provided).

Example:

```bash
cutscene-locator --asr-preflight-only --asr-backend faster-whisper --model-path models/faster-whisper/tiny
```

Source checkout examples (same contract, different invocation path):

```powershell
py .\cli.py --asr-preflight-only --asr-backend faster-whisper --model-path models/faster-whisper/tiny
```

```bash
python ./cli.py --asr-preflight-only --asr-backend faster-whisper --model-path models/faster-whisper/tiny
```

Both the installed entrypoint (`cutscene-locator ...`) and source-checkout invocation (`py .\\cli.py ...` / `python ./cli.py ...`) must emit the same single-line JSON stdout contract in preflight-only mode.

Example output shape:

```json
{"backend":"faster-whisper","device":{"compute_type":"auto","cuda_probe_label":"ctranslate2","requested":"auto","resolution_reason":"--device auto selected cuda because ctranslate2 CUDA probe reported available"},"mode":"asr_preflight_only","model_resolution":{"requested":{"auto_download":null,"model_id":null,"model_path":"models/faster-whisper/tiny","revision":null},"resolved_model_path":"models/faster-whisper/tiny"}}
```

Behavior notes:

- Skips `--input`, `--script`, and `--out` requirements in preflight-only mode.
- Performs only backend availability/capability validation, model resolution, and device resolution.
- Does not run ffmpeg preflight, ingest/preprocess, transcription, matching, scene reconstruction, or exports.
- Still enforces backend/model/device validation failures with deterministic error messages.
- On Windows PowerShell, copied multiline commands can include continuation/copy artifacts (for example trailing `` ` `` or accidental extra newlines/characters); those are shell-level formatting artifacts, not additional tool output lines.

Qwen readiness/smoke checks (optional, env-gated):

- `tests/test_cli.py` includes a deterministic preflight QA assertion for `qwen3-asr` JSON payload shape (`mode/backend/model_resolution/device` fields) with single-line stdout enforcement.
- Optional loader-init smoke (`Qwen3ReadinessSmokeTests`) is gated to stay offline by default:
  - `CUTSCENE_QWEN3_INIT_SMOKE=1` enables the init-only class.
  - `CUTSCENE_QWEN3_MODEL_PATH=<local_qwen3_snapshot_dir>` must point to an existing local model snapshot.
  - This test validates loader initialization only (`Qwen3ASRModel.from_pretrained(...)`) and asserts `device` is **not** passed into `from_pretrained` (device transfer is post-load via `.to(...)`).
- Optional runtime smoke (`Qwen3RuntimeSmokeTests`) is additionally gated and disabled by default:
  - `CUTSCENE_QWEN3_RUNTIME_SMOKE=1` enables runtime smoke.
  - `CUTSCENE_QWEN3_MODEL_PATH=<local_qwen3_snapshot_dir>` must point to an existing local model snapshot.
  - `CUTSCENE_QWEN3_RUNTIME_AUDIO=<local_audio_file>` must point to an existing local audio file.
  - Expected output contract is standard ASR result shape (`meta.backend == "qwen3-asr"`, non-empty `segments` list); this smoke is intended for explicit local validation only, not default CI.

---

### `--mock-asr <file>`

Path to mock ASR JSON output.

Required when `--asr-backend mock` is used.

Example:

```bash

--asr-backend mock --mock-asr tests/fixtures/asr.json

```

- Default: none
- Valid values: path to deterministic mock ASR JSON contract fixture
- Backend applicability: required for `mock`; ignored by `faster-whisper`/`qwen3-asr`/`whisperx`/`vibevoice`

---


### `--model-path <path>`

Filesystem path to a local ASR model directory/file.

- Path must already exist.
- This flag is filesystem-only and never accepts a Hugging Face repo id.

Model loading policy is deterministic and loaded only when requested:

1. If `--model-path` is provided, that exact path is used (no search, no download).
2. Else if `--model-id` is provided, resolve deterministic cache path first and download only when cache is missing.
3. Else if `--auto-download` is provided, download the mapped backend/model-size artifact.
4. Else resolve via local `models/` directory convention.

If the selected backend/model artifacts are missing or invalid, the CLI fails with a clear error. It never silently changes backend, model family, or mode.

- Default: none
- Valid values: existing local filesystem path
- Backend applicability: model-loading backends (`faster-whisper`, `qwen3-asr`, `whisperx`, `vibevoice`); not used by `mock`

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
  - Accepted model-id examples include `Qwen/Qwen3-ASR-0.6B` and `Qwen/Qwen3-ASR-1.7B`.
  - `--model-id` must resolve to a full Transformers ASR snapshot containing `config.json`, tokenizer assets, `tokenizer_config.json`, and model weights.
  - `processor_config.json` / `preprocessor_config.json` are optional for `qwen3-asr` readiness validation.
  - Incomplete local folders are rejected with deterministic validation errors.
- `whisperx`
  - `--model-id` must resolve to a local CTranslate2 Whisper snapshot directory containing `config.json`, `model.bin`, and tokenizer/vocabulary assets.
  - Standard Transformers checkpoints are rejected by deterministic snapshot validation.

- Default: none
- Valid values: non-empty Hugging Face repo id string
- Backend applicability: model-loading backends (`faster-whisper`, `qwen3-asr`, `whisperx`, `vibevoice`); not useful for `mock`

### `--auto-download <tiny|base|small>`

Optional convenience selector for deterministic backend-specific model-id mapping.

- Default: none
- Valid choices: `tiny`, `base`, `small`
- Validation rule: mutually exclusive with `--model-path` and `--model-id`
- Backend applicability:
  - `faster-whisper`: supported (mapped to known faster-whisper repos)
  - `qwen3-asr` / `whisperx` / `vibevoice`: currently no documented size mapping; use `--model-id` or `--model-path`
  - `mock`: not applicable

### `--revision <revision>`

Optional Hugging Face revision when used with `--model-id`.

- Example: `--revision main`
- If omitted, cache folder uses `default`.

- Default: `default` cache key when omitted
- Valid values: non-empty revision string (branch/tag/commit)
- Backend applicability: applies only with `--model-id` model-resolution flow

### `--device <cpu|cuda|auto>`

Selects ASR execution device.

- Default: `auto`
- Valid choices: `cpu`, `cuda`, `auto`
- Backend applicability: runtime ASR backends (`faster-whisper`, `qwen3-asr`, `whisperx`, `vibevoice`); no effect for `mock`

### `--compute-type <float16|float32|auto>`

Controls faster-whisper compute precision passed to `WhisperModel(compute_type=...)`.

- `float16`: fastest on most CUDA setups
- `float32`: safer manual mitigation for CUDA instability
- `auto`: backend default

When `--asr-backend faster-whisper` is selected, CLI prints an ASR preflight line before transcription that includes:

- detected `ctranslate2` version
- CUDA device count reported by `ctranslate2`
- selected resolved device
- selected compute type

If CUDA ASR aborts, first retry with `--compute-type float32`, then verify torch/ctranslate2 CUDA wheel compatibility (see faster-whisper issue #1086).

- Default: `auto`
- Valid choices: `float16`, `float32`, `auto`
- Backend applicability: primarily `faster-whisper`; `whisperx` consumes resolved value; `qwen3-asr`/`vibevoice` ignore this setting

### `--asr-language <language>`

Optional ASR language hint.

- Default: backend/runtime default (`None` at CLI level)
- Valid values: language code/name string accepted by selected backend runtime
- Backend applicability:
  - `faster-whisper`: passed to `model.transcribe(language=...)`
  - `whisperx`: passed at model-load time
  - `vibevoice`: passed to runtime transcription call
  - `qwen3-asr`: currently not consumed by backend implementation
  - `mock`: not applicable

### `--asr-beam-size <int>`

Beam search width for ASR decoding where supported.

- Default: `1`
- Valid range: integer `>= 1`
- Backend applicability:
  - `faster-whisper`: consumed when value differs from default
  - `qwen3-asr` / `whisperx` / `vibevoice` / `mock`: currently not consumed

### `--asr-temperature <float>`

Sampling temperature for ASR decoding where supported.

- Default: `0.0`
- Valid range: float `>= 0.0`
- Backend applicability:
  - `faster-whisper`: always passed
  - `qwen3-asr` / `whisperx` / `vibevoice` / `mock`: currently not consumed

### `--asr-best-of <int>`

Number of sampling candidates for temperature-based decoding.

- Default: `1`
- Valid range: integer `>= 1`
- Deterministic guard: when `--asr-temperature` is `0.0`, `--asr-best-of` must remain `1`
- Backend applicability:
  - `faster-whisper`: passed only when `temperature > 0.0` and `best_of != 1`
  - `qwen3-asr` / `whisperx` / `vibevoice` / `mock`: currently not consumed

### `--asr-no-speech-threshold <float>`

Optional silence/no-speech detection threshold.

- Default: unset (`None`)
- Valid range: float in `[0.0, 1.0]`
- Backend applicability:
  - `faster-whisper`: forwarded when set
  - `qwen3-asr` / `whisperx` / `vibevoice` / `mock`: currently not consumed

### `--asr-logprob-threshold <float>`

Optional low-confidence token log-probability threshold.

- Default: unset (`None`)
- Valid range: float in `[0.0, 1.0]`
- Backend applicability:
  - `faster-whisper`: forwarded when set
  - `qwen3-asr` / `whisperx` / `vibevoice` / `mock`: currently not consumed

### `--progress <on|off>`

Controls progress behavior for ASR/model-resolution operations.

- For Hugging Face snapshot downloads, this controls progress bars as before.
- For faster-whisper transcription on CUDA, progress bars are disabled internally to avoid Windows tqdm monitor-thread crashes.

- Windows default: `off` (sets `HF_HUB_DISABLE_PROGRESS_BARS=1`)
- Non-Windows default: `on`
- Useful for avoiding native aborts in tqdm monitor threads on some Windows environments.
- Valid choices: `on`, `off`
- Backend applicability: model download progress + runtime guard behavior (not transcription quality)

---

### `--match-threshold <float>`

Minimum confidence score for “high-confidence” matches.

- Default: `0.85`
- Lower scores are still exported but flagged.
- Valid range: float (recommended `[0.0, 1.0]`)
- Backend applicability: all backends (post-ASR matching)

### `--match-quick-threshold <float>`

Quick-filter minimum token-overlap score used before expensive fuzzy scoring.

- Default: `0.25`
- Set to `0.0` to disable quick-filter pruning.
- Valid range: float (recommended `[0.0, 1.0]`)
- Backend applicability: all backends (post-ASR matching)

### `--match-length-bucket-size <int>`

Token-count bucket size used by matching candidate indexes.

- Default: `4`
- Valid range: integer `> 0`
- Backend applicability: all backends (post-ASR matching)

### `--match-max-length-bucket-delta <int>`

How many neighboring length buckets to include when searching candidates.

- Default: `3`
- Valid range: integer `>= 0`
- Backend applicability: all backends (post-ASR matching)

### `--match-monotonic-window <int>`

Optional monotonic alignment window in script-row indexes.

- Default: `0` (disabled)
- When > 0, later segments search only from previous best row forward by this window, which can reduce comparisons and enforce timeline consistency.
- Valid range: integer `>= 0`
- Backend applicability: all backends (post-ASR matching)

### `--match-progress-every <int>`

Verbose matching progress interval (segments).

- Default: `50`
- Valid range: integer `> 0`
- Backend applicability: all backends (verbose matching logs)

### `--asr-vad-filter <on|off>`

Controls ASR backend VAD segmentation filter where supported.

- Default: `off`
- Valid choices: `on`, `off`
- Backend applicability: `faster-whisper` currently consumes this setting; other backends currently ignore it

### `--asr-merge-short-segments <seconds>`

Post-ASR deterministic merge threshold for short adjacent segments.

- Default: `0.0` (disabled)
- When > 0, segments shorter than threshold are merged into the previous segment to stabilize segment counts across runs/devices.
- Valid range: float `>= 0.0`
- Backend applicability: currently applied in `faster-whisper` post-processing path

---

### `--keep-wav`

Do not delete intermediate WAV files.

Useful for debugging ffmpeg preprocessing.

- Default: disabled (`False`)
- Valid values: switch flag (present/absent)
- Backend applicability: all backends (preprocessing artifact retention)

---

### `--ffmpeg-path <path>`

Explicit path to ffmpeg binary.

If not provided:

- `ffmpeg` is resolved from PATH.

- Default: auto-resolve from `PATH`
- Valid values: executable path to `ffmpeg`
- Backend applicability: all backends (mandatory media preprocessing dependency)

---

### `--verbose`

Enable verbose logging.

- Default: disabled (`False`)
- Valid values: switch flag (present/absent)
- Backend applicability: all backends

---

### `--version`

Print version and exit.

- Default: disabled (`False`)
- Valid values: switch flag (present/absent)
- Backend applicability: global CLI behavior

---

### `--help`

Print help and exit.

- Default: disabled (`False`)
- Valid values: switch flag (present/absent)
- Backend applicability: global CLI behavior

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

- Default: `mock`
- Valid values: declared backend names from registry (`mock`, `faster-whisper`, `qwen3-asr`, `whisperx`, `vibevoice`, and alignment-only backends such as `qwen3-forced-aligner` that are rejected in ASR mode)
- Backend applicability: selector for backend-specific execution path

- Default: none
- Valid values: path to deterministic mock ASR JSON contract fixture
- Backend applicability: required for `mock`; ignored by `faster-whisper`/`qwen3-asr`/`whisperx`/`vibevoice`
