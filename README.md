# cutscene-locator

`cutscene-locator` is a CLI tool that reconstructs dialogue scenes from audio or video
by aligning spoken dialogue to an existing script (TSV/CSV) using ASR and time-based matching.

It is designed for localization, QA, and script verification workflows in games and other
media where dialogue is fragmented across multiple files but spoken as continuous scenes.

This tool does **not** translate text.  
It locates, aligns, and groups existing lines.

---

## What problem does this solve?

In many games, dialogue lines are:

- spread across multiple script files
- reused in different contexts
- difficult to map back to actual spoken scenes

Meanwhile, the audio contains the *true* dialogue order and timing.

`cutscene-locator` uses ASR timestamps as a **ground truth timeline** to:

- determine which lines are actually spoken
- reconstruct scenes as they occur in audio
- surface ambiguous or high-risk lines for review

---

## What this tool does

- Accepts **audio or video** input
- Uses `ffmpeg` to extract and normalize audio
- Runs ASR to obtain dialogue segments with timestamps
- Fuzzy-matches ASR text against a known script
- Reconstructs dialogue scenes based on time gaps
- Exports:
  - matched lines with confidence scores
  - reconstructed scenes
  - QA subtitles
  - target-language subtitles (if available)

---

## What this tool does NOT do

- No automatic translation
- No downloading of online content
- No OCR
- No rewriting or paraphrasing of dialogue
- No guessing when confidence is low

Ambiguity is surfaced, not hidden.

---

## Typical use cases

- Game localization and dubbing QA
- Verifying which script lines are actually spoken
- Reconstructing cutscenes from fragmented scripts
- Reviewing dialogue timing with subtitles
- Identifying high-risk translation lines due to context

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

## Outputs

Depending on configuration and script content, the tool produces:

- `matches.csv` — ASR segments matched to script lines with confidence scores
- `scenes.json` — reconstructed dialogue scenes with timeline ordering
- `subs_qa.srt` — QA subtitles (original + translation, if available)
- `subs_target.srt` — target-language subtitles (if translation exists)

---

## Requirements

- Python 3.10+
- `ffmpeg` available in PATH
- No GPU required for mock/testing workflows
- ASR backends may require GPU depending on implementation

---

## Project status

This project is currently CLI-first.

Planned (out of scope for initial milestones):

- Additional ASR backends
- Optional PySide6 UI frontend
- Advanced diarization support

---

## Philosophy

`cutscene-locator` is a **locator and alignment tool**, not an oracle.

- Audio timestamps are treated as factual data
- Matching is probabilistic but transparent
- Human review remains central to the workflow

The goal is to reduce manual searching and context loss — not to replace human judgment.
