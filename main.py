from datetime import datetime
from pathlib import Path

from transcript_app.modal_app import app
from transcript_app.transcription_service import transcribe_chunked


@app.local_entrypoint()
def main(
    audio_path: str = "audio.mp3",
    chunk_seconds: int = 60,
    overlap_seconds: int = 1,
    max_new_tokens: int = 1024,
    yes: bool = False,
    confirm: str = "",
):
    import sys

    resolved_audio_path = str(Path(audio_path).expanduser().resolve())
    if not Path(resolved_audio_path).exists():
        raise FileNotFoundError(f"Audio file not found: {resolved_audio_path}")

    start = datetime.now()
    print(f"Start time: {start.strftime('%H:%M:%S')}")
    print(f"Using audio file: {resolved_audio_path}")
    if not yes:
        confirmation = confirm.strip().lower()
        if not confirmation:
            if sys.stdin.isatty():
                try:
                    confirmation = input("Proceed with this audio file? [y/N]: ").strip().lower()
                except EOFError:
                    confirmation = ""
            else:
                raise RuntimeError(
                    "Interactive confirmation is unavailable in this terminal. "
                    "Re-run with --confirm y (or --yes) to proceed."
                )
        if confirmation not in {"y", "yes"}:
            print("Cancelled by user.")
            return

    with open(resolved_audio_path, "rb") as f:
        result = transcribe_chunked.remote(
            f.read(),
            chunk_seconds=chunk_seconds,
            overlap_seconds=overlap_seconds,
            max_new_tokens=max_new_tokens,
        )

    end = datetime.now()
    elapsed = end - start
    print(f"End time:   {end.strftime('%H:%M:%S')}")
    print(f"Elapsed:    {str(elapsed).split('.')[0]}")
    print(result)
    output_dir = Path("output")
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"output_{timestamp}.txt"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(result)
    print(f"Wrote transcript to: {output_path}")
