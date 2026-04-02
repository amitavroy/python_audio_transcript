## What this program does

This project transcribes an audio file by running **Microsoft VibeVoice ASR** on **Modal** (GPU `A10G`), then writes the resulting transcription JSON to an output text file.

It uses the `transcribe_chunked()` flow:

- Splits the audio into overlapping chunks (defaults: 60s chunks, 1s overlap).
- Transcribes each chunk on the remote GPU.
- Shifts each segment’s timestamps to absolute time in the original audio.
- Performs a simple overlap de-duplication to reduce repeated lines near chunk boundaries.

Application and Modal code live in the **`transcript_app/`** package (`modal_app`, `transcription_service`, helpers). The repo root keeps `main.py` as the CLI entrypoint.

## Prerequisites

- **Python**: 3.12+ (see `pyproject.toml`)
- **uv**: used to run the project
- **Modal**: you must be authenticated (`modal token new`)

## How to run

From the repo root:

```bash
uv run modal run main.py --audio-path /path/to/audio.mp3
```

The program will print the transcription JSON to stdout and also write it to `output/output_YYYYMMDD_HHMMSS.txt`.

### Non-interactive / CI-friendly run

`main.py` asks for confirmation before processing unless you bypass it:

```bash
uv run modal run main.py --audio-path /path/to/audio.mp3 --confirm y
```

Or:

```bash
uv run modal run main.py --audio-path /path/to/audio.mp3 --yes
```

## Common options

- **Change chunking**:

```bash
uv run modal run main.py --audio-path /path/to/audio.mp3 --chunk-seconds 90 --overlap-seconds 3
```

- **Control generation length**:

```bash
uv run modal run main.py --audio-path /path/to/audio.mp3 --max-new-tokens 1024
```
