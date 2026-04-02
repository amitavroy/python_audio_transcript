## What this program does

This project transcribes an audio file by running **Microsoft VibeVoice ASR** on **Modal** (GPU `A10G`), then writes the resulting transcription JSON to an output text file.

It uses the `transcribe_chunked()` flow:

- Splits the audio into overlapping chunks (defaults: 60s chunks, 1s overlap).
- Transcribes each chunk on the remote GPU.
- Shifts each segment’s timestamps to absolute time in the original audio.
- Performs a simple overlap de-duplication to reduce repeated lines near chunk boundaries.

Note: per your request, this README intentionally **ignores** the standalone `transcribe()` function.

## Prerequisites

- **Python**: 3.12+ (see `pyproject.toml`)
- **uv**: used to run the project
- **Modal**: you must be authenticated (`modal token new`)

## How to run

From the repo root:

```bash
uv run modal run main.py --audio-path /path/to/audio.mp3
```

The program will print the transcription JSON to stdout and also write it to a file named like `output_YYYYMMDD_HHMMSS.txt` (unless you override the output path).

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

- **Choose output file**:

```bash
uv run modal run main.py --audio-path /path/to/audio.mp3 --output-path transcript.json.txt
```
